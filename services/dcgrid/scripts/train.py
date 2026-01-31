from __future__ import annotations

import json
import os
import signal
import sys
import traceback
import pickle
from typing import Any, Dict, Optional
import math

import numpy as np
import pandas as pd
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
        pass
    finally:
        _RUN_ACTIVE = False


def _handle_term(signum, frame) -> None:
    # Kubernetes sends SIGTERM on deletion/eviction/rolling updates.
    _safe_end_run("KILLED")
    sys.exit(143)


signal.signal(signal.SIGTERM, _handle_term)
signal.signal(signal.SIGINT, _handle_term)


def _to_jsonable(x: Any) -> Any:
    """Convert numpy/pandas types to JSON-serializable python types."""
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    # last resort
    return str(x)


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


def train_xgb(df: pd.DataFrame) -> Dict[str, Any]:
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


def train_tft(df: pd.DataFrame) -> Dict[str, Any]:
    from dcgrid.models.tft import train_tft as _train_tft
    return _train_tft(df)


def _persist_artifact(artifact: Dict[str, Any]) -> str:
    """
    Persist trained model to MinIO in a format that won't explode:
      - TFT: upload .ckpt and metadata .json (no pickling)
      - Others (xgb): pickle ok
    Returns the object-store key(s) summary string.
    """
    region = config.REGION
    model_name = config.MODEL_NAME.lower()
    prefix = f"models/{region}/{model_name}"

    art_type = (artifact.get("type") or model_name).lower()

    if art_type == "tft":
        ckpt_path = artifact.pop("checkpoint_path", None)
        if not ckpt_path or not os.path.exists(ckpt_path):
            raise RuntimeError(f"TFT checkpoint_path missing or does not exist: {ckpt_path}")

        ckpt_key = f"{prefix}.ckpt"
        meta_key = f"{prefix}.json"

        with open(ckpt_path, "rb") as f:
            put_bytes(ckpt_key, f.read(), content_type="application/octet-stream")

        meta = _to_jsonable(artifact)
        put_bytes(meta_key, json.dumps(meta, indent=2).encode("utf-8"), content_type="application/json")

        # Optional: also log to mlflow (best-effort)
        try:
            mlflow.log_artifact(ckpt_path, artifact_path="model")
            mlflow.log_dict(meta, "model/metadata.json")
        except Exception:
            pass

        return f"{ckpt_key} + {meta_key}"

    # default: pickle
    pkl_key = f"{prefix}.pkl"
    blob = pickle.dumps(artifact)
    put_bytes(pkl_key, blob, content_type="application/octet-stream")
    return pkl_key


def main() -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(f"dcgrid-{config.REGION}")

    df = _load_features()
    run_name = f"{config.MODEL_NAME}-{config.REGION}"

    global _RUN_ACTIVE
    run = None

    try:
        run = mlflow.start_run(run_name=run_name)
        _RUN_ACTIVE = True

        mlflow.log_param("region", config.REGION)
        mlflow.log_param("model", config.MODEL_NAME)
        mlflow.log_param("horizon_h", int(config.FORECAST_HORIZON_H))
        mlflow.log_param("rows", int(len(df)))

        if config.MODEL_NAME.lower() == "tft":
            artifact = train_tft(df)
        elif config.MODEL_NAME.lower() == "prophet":
            from dcgrid.models.prophet_baseline import train_prophet
            artifact = train_prophet(df)
        else:
            artifact = train_xgb(df)

        # log numeric metrics
       
        metrics = {}
        for k, v in artifact.items():
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                metrics[k] = float(v)
        mlflow.log_metrics(metrics)

        saved = _persist_artifact(artifact)
        mlflow.log_param("model_object_store", saved)
        print("Saved model artifact(s) to object storage:", saved)

        _safe_end_run("FINISHED")

    except SystemExit:
        # signal handler may sys.exit()
        raise
    except Exception as e:
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
