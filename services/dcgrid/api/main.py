from __future__ import annotations
import os
import pickle
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dcgrid import config
from dcgrid.db import engine
from dcgrid.storage import get_bytes

app = FastAPI(title="DCGrid Forecast API", version="0.1.0")

class ForecastPoint(BaseModel):
    ts_utc: datetime
    dc_load_mw: float
    stress_score: float | None = None

class ForecastResponse(BaseModel):
    region: str
    model: str
    generated_at_utc: datetime
    horizon_h: int
    points: List[ForecastPoint]

def _load_latest_features(n_hours: int = 24*30) -> pd.DataFrame:
    q = f"""SELECT ts_utc, demand_mw, wind_mw, solar_mw, renewable_share, price_eur_mwh,
                 neg_price_freq_7d, renew_var_24h, stress_score, dc_load_mw
              FROM dcgrid.features_hourly
              WHERE region = %s
              ORDER BY ts_utc DESC
              LIMIT {n_hours}
          """
    df = pd.read_sql(q, engine(), params=(config.REGION,), parse_dates=["ts_utc"])
    df = df.sort_values("ts_utc").set_index("ts_utc")
    return df

def _load_model() -> Dict[str, Any]:
    if config.MODEL_NAME.lower() == "tft":
        # Option B: TFT is saved as Lightning checkpoint + meta
        ckpt_blob = get_bytes(config.MODEL_OBJECT_KEY)
        meta_blob = get_bytes(config.MODEL_META_KEY)

        # TODO: load ckpt into TFT model here (we’ll wire this after train writes meta correctly)
        return {
            "type": "tft",
            "checkpoint_bytes": ckpt_blob,
            "meta": json.loads(meta_blob),
        }

    # legacy models
    blob = get_bytes(config.MODEL_OBJECT_KEY)
    return pickle.loads(blob)


@app.get("/healthz")
def healthz():
    return {"ok": True, "ts": datetime.now(timezone.utc).isoformat()}

@app.get("/forecast", response_model=ForecastResponse)
def forecast():
    df = _load_latest_features()
    if df.empty:
        raise HTTPException(404, "No features found - run ingest first.")

    model_art = _load_model()
    mtype = model_art.get("type")

    horizon = config.FORECAST_HORIZON_H
    last_ts = df.index.max()
    future_index = pd.date_range(last_ts + pd.Timedelta(hours=1), periods=horizon, freq="1H", tz="UTC")

    # For demo: use XGB recursive 1-step ahead on latest known features.
    if mtype == "xgb":
        model = model_art["model"]
        feat_cols = model_art["features"]
        cur = df.copy()
        preds = []
        for ts in future_index:
            x = cur.iloc[-1:][feat_cols].copy()
            yhat = float(model.predict(x)[0])
            # Build next row by carrying forward covariates (naive). In production, add proper covariate forecasting.
            next_row = cur.iloc[-1:].copy()
            next_row.index = [ts]
            next_row["dc_load_mw"] = yhat
            cur = pd.concat([cur, next_row], axis=0)
            preds.append(yhat)
        stress = cur.loc[future_index, "stress_score"].astype(float).fillna(method="ffill").fillna(0.0).tolist()
    elif mtype == "tft":
        # For TFT we provide a simple fallback: return persistence if no covariate forecasting implemented.
        last = float(df["dc_load_mw"].iloc[-1])
        preds = [last] * horizon
        stress = df["stress_score"].iloc[-1]
        stress = [float(stress) if stress == stress else 0.0] * horizon
    elif mtype == "prophet":
        # Return daily forecast expanded to hours (persistence within day)
        from prophet.serialize import model_to_json, model_from_json
        model = model_art["model"]
        future_days = pd.date_range(last_ts.date() + timedelta(days=1), periods=(horizon//24)+2, freq="1D", tz="UTC")
        fdf = pd.DataFrame({"ds": future_days.tz_convert(None)})
        fc = model.predict(fdf)
        daily = pd.Series(fc["yhat"].values, index=future_days)
        preds = []
        for ts in future_index:
            d = ts.normalize()
            preds.append(float(daily.get(d, daily.iloc[-1])))
        stress = [0.0] * horizon
    else:
        raise HTTPException(500, f"Unknown model type: {mtype}")

    points = [ForecastPoint(ts_utc=ts.to_pydatetime(), dc_load_mw=float(p), stress_score=float(s)) for ts, p, s in zip(future_index, preds, stress)]
    return ForecastResponse(
        region=config.REGION,
        model=config.MODEL_NAME,
        generated_at_utc=datetime.now(timezone.utc),
        horizon_h=horizon,
        points=points,
    )

@app.get("/stress/alerts")
def stress_alerts(threshold: float = 2.0):
    df = _load_latest_features()
    if df.empty:
        raise HTTPException(404, "No features found - run ingest first.")
    recent = df.tail(24*14).copy()
    hits = recent[recent["stress_score"] >= threshold].tail(100)
    return {
        "region": config.REGION,
        "threshold": threshold,
        "count": int(len(hits)),
        "events": [{"ts_utc": i.isoformat(), "stress_score": float(v)} for i, v in hits["stress_score"].items()],
    }
