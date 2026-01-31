from __future__ import annotations

import os
import sys
import signal
import pickle
import traceback
from typing import Optional

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

# ---- MLflow run lifecycle helpers (fix "RUNNING forever" on SIGTERM) ----
_RUN_ACTIVE = False


def _safe_end_run(status: str) -> None:
    """End MLflow run safely without raising."""
    global _RUN_ACTIVE
    try:
        if _RUN_ACTIVE:
            mlflow.end_run(status=status)
    except Exception:
        # Don't mask the original failure / signal handling
        pass
    finally:
        _RUN_ACTIVE = False


def _handle_term(signum, frame) -> None:
    # Kubernetes sends SIGTERM on deletion/eviction/rolling updates.
    # Ensure run is closed so it doesn't remain RUNNING.
    _safe_end_run("KILLED")
    # 128 + 15 = 143 is a common convention for SIGTERM exit
    sys.exit(143)


# Register signal handlers early
signal.signal(signal.SIGTERM, _handle_term)
signal.signal(signal.SIGINT, _handle_term)


def _load_features() -> pd.DataFrame:
    eng = engine()
    q = text(
        """
        SELECT ts_utc, demand_mw, wind_mw, solar_mw, renew_mw, renewable_share,
               price_eur_mwh, neg_price_freq_7d, price_spike_flag, ramp_rate_mw_h, renew_var_24h,
               dc_load_mw, stress_score
        FROM dcgrid.features_hourly
        WHERE region = :r
        ORDER BY ts_utc ASC
        """
    )
    df = pd.read_sql(q, eng, params={"r": config.REGION}, parse_dates=["ts_utc"])
    if df.empty:
        raise RuntimeError(f"No features found for region={config.REGION}. Run ingest first.")
    df = df.set_index("ts_utc")
    return df


def train_xgb(df: pd.DataFrame) -> dict:
    # One-step ahead target; API can roll forward recursively
    horizon = int(config.FORECAST_HORIZON_H)

    if "dc_load_mw" not in df.columns:
        raise RuntimeError("dc_load_mw column missing from features table.")

    y = df["dc_load_mw"].shift(-1)
    X = df.drop(columns=["dc_load_mw"])

    data = X.join(y.rename("y")).dropna()
    if data.empty:
        raise RuntimeError("After shifting target and dropping NAs, no rows remain for training.")

    X = data.drop(columns=["y"])
    y = data["y"]

    # Time-series split
    tss = TimeSeriesSplit(n_splits=3)
    maes: list[float] = []
    last_model: Optional[XGBRegressor] = None

    n_jobs = int(os.getenv("XGB_N_JOBS", "4"))

    for fold, (tr, te) in enumerate(tss.split(X), start=1):
        model = XGBRegressor(
            n_estimators=800,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            n_jobs=n_jobs,
        )
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[te])
        mae = mean_absolute_error(y.iloc[te], pred)
        maes.append(float(mae))
        last_model = model
        print(f"fold={fold} MAE={mae:.2f} MW")

    if last_model is None:
        raise RuntimeError("Training produced no model (unexpected).")

    return {
        "type": "xgb",
        "region": config.REGION,
        "features": list(X.columns),
        "horizon_h": horizon,
        "mae_cv_mw": float(np.mean(maes)),
        "model": last_model,
    }


def train_tft_placeholder(df: pd.DataFrame) -> dict:
    from dcgrid.models.tft import train_tft
    return train_tft(df)


def main() -> None:
    # Tracking URI: in-cluster should be http://mlflow:5000; with port-forward you can set localhost.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(f"dcgrid-{config.REGION}")

    df = _load_features()

    global _RUN_ACTIVE
    run_name = f"{config.MODEL_NAME}-{config.REGION}"

    try:
        with mlflow.start_run(run_name=run_name):
            _RUN_ACTIVE = True

            mlflow.log_param("region", config.REGION)
            mlflow.log_param("model", config.MODEL_NAME)
            mlflow.log_param("horizon_h", int(config.FORECAST_HORIZON_H))
            mlflow.log_param("rows", int(len(df)))

            if config.MODEL_NAME.lower() == "tft":
                artifact = train_tft_placeholder(df)
            elif config.MODEL_NAME.lower() == "prophet":
                from dcgrid.models.prophet_baseline import train_prophet
                artifact = train_prophet(df)
            else:
                artifact = train_xgb(df)

            mlflow.log_metrics({k: v for k, v in artifact.items() if isinstance(v, (int, float))})

            blob = pickle.dumps(artifact)
            key = f"models/{config.REGION}/{config.MODEL_NAME}.pkl"
            put_bytes(key, blob, content_type="application/octet-stream")
            print("Saved model artifact to object storage:", key)

            _safe_end_run("FINISHED")

    except SystemExit:
        # signal handler may sys.exit(); MLflow already ended there
        raise
    except Exception as e:
        # Log traceback to MLflow so it’s visible in the UI
        tb = traceback.format_exc()
        try:
            mlflow.log_text(tb, "train_error.txt")
            mlflow.log_param("error", str(e)[:500])
        except Exception:
            pass

        _safe_end_run("FAILED")
        raise


if __name__ == "__main__":
    main()
