from __future__ import annotations

import numpy as np
import pandas as pd


def train_tft(df: pd.DataFrame) -> dict:
    """
    Minimal, reproducible TFT training using PyTorch Forecasting.

    Key robustness points:
      - Cast categorical covariates to string (PyTorch Forecasting requires non-numeric dtype for categoricals).
      - Ensure all real-valued covariates are numeric (coerce errors to NaN).
      - Handle missing/empty dc_load_mw gracefully by falling back to demand_mw as a proxy target.
      - Drop rows with missing timestamps/targets and enforce sorted time.
    """
    try:
        import torch  # noqa: F401
        from pytorch_lightning import Trainer
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
    except Exception as e:
        raise RuntimeError(
            "TFT deps missing. Install torch, pytorch-lightning and pytorch-forecasting, "
            "or set MODEL_NAME=xgb."
        ) from e

    data = df.copy()

    # --- Ensure timestamp column exists and is datetime (UTC) ---
    if "ts_utc" in data.columns:
        ts_col = "ts_utc"
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index().rename(columns={"index": "ts_utc"})
        ts_col = "ts_utc"
    else:
        # best-effort fallback: take first column as time
        ts_col = data.columns[0]

    data = data.rename(columns={ts_col: "time"})
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    data = data.dropna(subset=["time"]).sort_values("time")

    # --- Create time index in hours ---
    data["time_idx"] = ((data["time"] - data["time"].min()).dt.total_seconds() // 3600).astype(int)

    # Single series demo; for multi-series use region or other IDs
    if "region" in data.columns:
        data["series"] = data["region"].astype(str)
    else:
        data["series"] = "dcgrid"

    # --- Target: prefer dc_load_mw; fallback to demand_mw if dc_load_mw is empty ---
    if "dc_load_mw" in data.columns:
        data["dc_load_mw"] = pd.to_numeric(data["dc_load_mw"], errors="coerce")
    else:
        data["dc_load_mw"] = np.nan

    if data["dc_load_mw"].notna().any():
        data["y"] = data["dc_load_mw"]
        target_note = "dc_load_mw"
    else:
        if "demand_mw" not in data.columns:
            raise RuntimeError(
                "No usable target: dc_load_mw is empty/missing and demand_mw not present."
            )
        data["demand_mw"] = pd.to_numeric(data["demand_mw"], errors="coerce")
        data["y"] = data["demand_mw"]
        target_note = "demand_mw_fallback"

    data = data.dropna(subset=["y"])

    # --- Known covariates: calendar features (categorical) ---
    data["hour"] = data["time"].dt.hour.astype("Int64").astype(str)
    data["dow"] = data["time"].dt.dayofweek.astype("Int64").astype(str)
    data["month"] = data["time"].dt.month.astype("Int64").astype(str)

    # --- Unknown reals (observed at prediction time only historically) ---
    # Keep only columns that exist; coerce to numeric for model stability.
    candidate_unknown_reals = [
        "y",
        "demand_mw",
        "wind_mw",
        "solar_mw",
        "renewable_share",
        "price_eur_mwh",
        "neg_price_freq_7d",
        "renew_var_24h",
        "stress_score",
    ]
    time_varying_unknown_reals: list[str] = []
    for c in candidate_unknown_reals:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
            time_varying_unknown_reals.append(c)

    # Remove rows that are completely unusable (all unknown reals NaN besides y already dropped)
    if "y" not in time_varying_unknown_reals:
        time_varying_unknown_reals = ["y"] + time_varying_unknown_reals

    # --- Dataset lengths ---
    max_encoder_length = 336  # 14 days
    max_prediction_length = 168  # 7 days

    # Need enough history for at least one prediction window
    if data["time_idx"].max() < (max_encoder_length + max_prediction_length + 1):
        raise RuntimeError(
            "Not enough data to train TFT: "
            f"need at least ~{max_encoder_length + max_prediction_length + 1} hourly rows, "
            f"got {int(data['time_idx'].max()) + 1}."
        )

    training_cutoff = data["time_idx"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        data[data["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target="y",
        group_ids=["series"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
        static_categoricals=["series"],
        time_varying_known_categoricals=["hour", "dow", "month"],
        time_varying_unknown_reals=time_varying_unknown_reals,
        target_normalizer=GroupNormalizer(groups=["series"]),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)

    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=16,
        attention_head_size=2,
        dropout=0.1,
        hidden_continuous_size=8,
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    trainer = Trainer(
        max_epochs=8,
        accelerator="cpu",
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Metric: MAE using quantile 0.5 on the prediction window
    raw_predictions, x = tft.predict(val_loader, mode="raw", return_x=True)
    # QuantileLoss default quantiles include 0.1, 0.5, 0.9 -> index 1 is median
    yhat = raw_predictions["prediction"][:, :, 1].detach().cpu().numpy()
    ytrue = x["decoder_target"].detach().cpu().numpy()
    mae = float(np.mean(np.abs(ytrue - yhat)))

    return {
        "type": "tft",
        "mae_val": mae,
        "model": tft,
        "target_used": target_note,
        "dataset_meta": {
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "known_categoricals": ["hour", "dow", "month"],
            "unknown_reals": time_varying_unknown_reals,
        },
    }
