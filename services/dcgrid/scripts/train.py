from __future__ import annotations
import os
import pickle
import pandas as pd
import numpy as np
import mlflow
from sqlalchemy import text
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

from dcgrid import config
from dcgrid.db import engine
from dcgrid.storage import put_bytes

def _load_features() -> pd.DataFrame:
    eng = engine()
    q = text("""
    SELECT ts_utc, demand_mw, wind_mw, solar_mw, renew_mw, renewable_share,
           price_eur_mwh, neg_price_freq_7d, price_spike_flag, ramp_rate_mw_h, renew_var_24h,
           dc_load_mw, stress_score
    FROM dcgrid.features_hourly
    WHERE region = :r
    ORDER BY ts_utc ASC
    """)
    df = pd.read_sql(q, eng, params={"r": config.REGION}, parse_dates=["ts_utc"])
    df = df.set_index("ts_utc")
    return df

def train_xgb(df: pd.DataFrame) -> dict:
    # Supervised transformation: predict next H hours (one-step ahead for simplicity)
    horizon = config.FORECAST_HORIZON_H
    # One-step ahead target (shift -1) for training; API will roll forward recursively
    y = df["dc_load_mw"].shift(-1)
    X = df.drop(columns=["dc_load_mw"])
    data = X.join(y.rename("y")).dropna()
    X = data.drop(columns=["y"])
    y = data["y"]

    # Time-series split
    tss = TimeSeriesSplit(n_splits=3)
    maes = []
    last_model = None

    for fold, (tr, te) in enumerate(tss.split(X), start=1):
        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=4,
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        mae = mean_absolute_error(y.iloc[te], pred)
        maes.append(mae)
        last_model = model
        print(f"fold={fold} MAE={mae:.2f} MW")

    artifact = {
        "type": "xgb",
        "region": config.REGION,
        "features": list(X.columns),
        "horizon_h": horizon,
        "mae_cv_mw": float(np.mean(maes)),
        "model": last_model,
    }
    return artifact

def train_tft_placeholder(df: pd.DataFrame) -> dict:
    # Full TFT implementation is included in dcgrid/models/tft.py; to keep runtime manageable
    # in constrained environments, we allow a placeholder training that can be expanded.
    from dcgrid.models.tft import train_tft
    return train_tft(df)

def main() -> None:
    df = _load_features()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(f"dcgrid-{config.REGION}")

    with mlflow.start_run(run_name=f"{config.MODEL_NAME}-{config.REGION}"):
        mlflow.log_param("region", config.REGION)
        mlflow.log_param("model", config.MODEL_NAME)
        mlflow.log_param("horizon_h", config.FORECAST_HORIZON_H)

        if config.MODEL_NAME.lower() == "tft":
            artifact = train_tft_placeholder(df)
        elif config.MODEL_NAME.lower() == "prophet":
            from dcgrid.models.prophet_baseline import train_prophet
            artifact = train_prophet(df)
        else:
            artifact = train_xgb(df)

        mlflow.log_metrics({k: v for k, v in artifact.items() if isinstance(v, (int, float))})

        blob = pickle.dumps(artifact)
        put_bytes(f"models/{config.REGION}/{config.MODEL_NAME}.pkl", blob, content_type="application/octet-stream")
        print("Saved model artifact to object storage:", f"models/{config.REGION}/{config.MODEL_NAME}.pkl")

if __name__ == "__main__":
    main()
