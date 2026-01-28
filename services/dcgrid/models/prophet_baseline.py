from __future__ import annotations
import pickle
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error

def train_prophet(df: pd.DataFrame) -> dict:
    # Prophet works best on daily aggregates; we forecast daily DC load energy proxy.
    daily = df["dc_load_mw"].resample("1D").mean().to_frame("y").dropna()
    p = daily.reset_index().rename(columns={"ts_utc": "ds"}) if "ts_utc" in daily.columns else daily.reset_index().rename(columns={"index":"ds"})
    p = p.rename(columns={"index":"ds"}) if "index" in p.columns else p
    model = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(p[["ds","y"]])

    # Simple backtest: last 30 days
    train = p.iloc[:-30]
    test = p.iloc[-30:]
    m2 = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    m2.fit(train[["ds","y"]])
    fc = m2.predict(test[["ds"]])
    mae = mean_absolute_error(test["y"], fc["yhat"])

    return {
        "type": "prophet",
        "mae_30d": float(mae),
        "model": model,
    }
