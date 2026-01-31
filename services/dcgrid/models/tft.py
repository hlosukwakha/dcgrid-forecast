from __future__ import annotations

import numpy as np
import pandas as pd


def train_tft(df: pd.DataFrame) -> dict:
    """
    Robust TFT training for messy real-world feature tables.

    Key points:
      - PyTorch Forecasting does NOT allow NA/inf values in real-valued covariates.
      - We therefore:
          1) auto-drop covariates that are all-NA (or mostly NA),
          2) replace inf -> NaN,
          3) fill remaining NaNs (0) and add *_isna indicator columns.
      - Categorical covariates must be non-numeric dtype -> cast to string.
      - Target uses dc_load_mw if present, otherwise falls back to demand_mw.
    """
    try:
        from lightning.pytorch import Trainer
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
    except Exception as e:
        raise RuntimeError(
            "TFT deps missing. Install torch, pytorch-lightning and pytorch-forecasting, "
            "or set MODEL_NAME=xgb."
        ) from e

    data = df.copy()

    # --- Timestamp ---
    if "ts_utc" in data.columns:
        ts_col = "ts_utc"
    elif isinstance(data.index, pd.DatetimeIndex):
        data = data.reset_index().rename(columns={"index": "ts_utc"})
        ts_col = "ts_utc"
    else:
        ts_col = data.columns[0]

    data = data.rename(columns={ts_col: "time"})
    data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
    data = data.dropna(subset=["time"]).sort_values("time")

    # --- time_idx in hours ---
    data["time_idx"] = ((data["time"] - data["time"].min()).dt.total_seconds() // 3600).astype(int)

    # --- group id ---
    if "region" in data.columns:
        data["series"] = data["region"].astype(str)
    else:
        data["series"] = "dcgrid"

    # --- target ---
    if "dc_load_mw" in data.columns:
        data["dc_load_mw"] = pd.to_numeric(data["dc_load_mw"], errors="coerce")
    else:
        data["dc_load_mw"] = np.nan

    if data["dc_load_mw"].notna().any():
        data["y"] = data["dc_load_mw"]
        target_note = "dc_load_mw"
    else:
        if "demand_mw" not in data.columns:
            raise RuntimeError("No usable target: dc_load_mw missing and demand_mw not present.")
        data["demand_mw"] = pd.to_numeric(data["demand_mw"], errors="coerce")
        data["y"] = data["demand_mw"]
        target_note = "demand_mw_fallback"

    data = data.dropna(subset=["y"])

    # --- known categoricals (cast to string as required) ---
    data["hour"] = data["time"].dt.hour.astype("Int64").astype(str)
    data["dow"] = data["time"].dt.dayofweek.astype("Int64").astype(str)
    data["month"] = data["time"].dt.month.astype("Int64").astype(str)

    known_categoricals = ["hour", "dow", "month"]

    # --- candidate unknown reals ---
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

    # Make sure all numeric columns are numeric, and replace inf -> NaN
    for c in candidate_unknown_reals:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
            data[c] = data[c].replace([np.inf, -np.inf], np.nan)

    # Drop features that are all-NA (or too sparse)
    # With optional signals (prices/DC), this prevents TFT from crashing.
    min_non_na_frac = 0.20  # keep if at least 20% values present
    unknown_reals: list[str] = ["y"]

    dropped: list[tuple[str, float]] = []
    kept: list[tuple[str, float]] = []

    for c in candidate_unknown_reals:
        if c == "y":
            continue
        if c not in data.columns:
            continue
        frac = float(data[c].notna().mean())
        if frac <= 0.0 or frac < min_non_na_frac:
            dropped.append((c, frac))
            continue
        kept.append((c, frac))
        unknown_reals.append(c)

    # For remaining unknown reals, fill missing values + add missing indicators
    # (PyTorch Forecasting requires finite values for reals).
    fill_cols = [c for c in unknown_reals if c != "y"]
    for c in fill_cols:
        ind = f"{c}_isna"
        data[ind] = data[c].isna().astype("float")
        data[c] = data[c].fillna(0.0)

    # Add the indicator columns as additional unknown reals
    for c in fill_cols:
        unknown_reals.append(f"{c}_isna")

    # --- Dataset lengths ---
    max_encoder_length = 336  # 14d
    max_prediction_length = 168  # 7d

    if data["time_idx"].max() < (max_encoder_length + max_prediction_length + 1):
        raise RuntimeError(
            "Not enough data to train TFT: "
            f"need at least ~{max_encoder_length + max_prediction_length + 1} hourly rows."
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
        time_varying_known_categoricals=known_categoricals,
        time_varying_unknown_reals=unknown_reals,
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

    raw_predictions, x = tft.predict(val_loader, mode="raw", return_x=True)
    yhat = raw_predictions["prediction"][:, :, 1].detach().cpu().numpy()  # median
    ytrue = x["decoder_target"].detach().cpu().numpy()
    mae = float(np.mean(np.abs(ytrue - yhat)))

    return {
        "type": "tft",
        "mae_val": mae,
        "model": tft,
        "target_used": target_note,
        "dropped_features": [{"name": n, "non_na_frac": f} for n, f in dropped],
        "kept_features": [{"name": n, "non_na_frac": f} for n, f in kept],
        "dataset_meta": {
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "known_categoricals": known_categoricals,
            "unknown_reals": unknown_reals,
            "min_non_na_frac": min_non_na_frac,
        },
    }
