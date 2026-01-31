from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def train_tft(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Robust TFT training for messy real-world feature tables.

    Returns ONLY serializable metadata + a checkpoint_path (no model object),
    so callers can persist the checkpoint bytes (MinIO) and avoid pickle issues.

    Key points:
      - PyTorch Forecasting does NOT allow NA/inf values in real covariates.
      - We:
          1) replace inf -> NaN,
          2) drop very sparse covariates,
          3) fill remaining NaNs (0) and add *_isna indicator columns,
          4) cast known categoricals to strings.
    """
    try:
        from lightning.pytorch import Trainer
        from lightning.pytorch.callbacks import ModelCheckpoint
        from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
        from pytorch_forecasting.data import GroupNormalizer
        from pytorch_forecasting.metrics import QuantileLoss
    except Exception as e:
        raise RuntimeError(
            "TFT deps missing. Install torch, lightning and pytorch-forecasting, "
            "or set MODEL_NAME=xgb."
        ) from e

    data = df.copy()

    # -----------------------
    # Timestamp normalization
    # -----------------------
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

    # time_idx in hours since start
    data["time_idx"] = (
        (data["time"] - data["time"].min()).dt.total_seconds() // 3600
    ).astype(int)

    # -----------------------
    # Group id
    # -----------------------
    if "region" in data.columns:
        data["series"] = data["region"].astype(str)
    else:
        data["series"] = "dcgrid"

    # -----------------------
    # Target selection
    # -----------------------
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
    if data.empty:
        raise RuntimeError("No training rows after target selection (all targets are NA).")

    # -----------------------
    # Known categoricals (string)
    # -----------------------
    data["hour"] = data["time"].dt.hour.astype("Int64").astype(str)
    data["dow"] = data["time"].dt.dayofweek.astype("Int64").astype(str)
    data["month"] = data["time"].dt.month.astype("Int64").astype(str)
    known_categoricals: List[str] = ["hour", "dow", "month"]

    # -----------------------
    # Candidate unknown reals
    # -----------------------
    candidate_unknown_reals: List[str] = [
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

    # Coerce numeric and replace inf
    for c in candidate_unknown_reals:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")
            data[c] = data[c].replace([np.inf, -np.inf], np.nan)

    # Drop sparse signals
    min_non_na_frac = float(os.getenv("TFT_MIN_NON_NA_FRAC", "0.20"))
    unknown_reals: List[str] = ["y"]

    dropped: List[Tuple[str, float]] = []
    kept: List[Tuple[str, float]] = []

    for c in candidate_unknown_reals:
        if c == "y" or c not in data.columns:
            continue
        frac = float(data[c].notna().mean())
        if frac < min_non_na_frac:
            dropped.append((c, frac))
            continue
        kept.append((c, frac))
        unknown_reals.append(c)

    # Fill remaining NA for reals + indicators
    fill_cols = [c for c in unknown_reals if c != "y"]
    for c in fill_cols:
        ind = f"{c}_isna"
        data[ind] = data[c].isna().astype("float32")
        data[c] = data[c].fillna(0.0)

    for c in fill_cols:
        unknown_reals.append(f"{c}_isna")

    # -----------------------
    # Dataset lengths
    # -----------------------
    max_encoder_length = int(os.getenv("TFT_ENCODER_LEN", "336"))  # 14d
    max_prediction_length = int(os.getenv("TFT_PRED_LEN", "168"))  # 7d

    if int(data["time_idx"].max()) < (max_encoder_length + max_prediction_length + 1):
        raise RuntimeError(
            "Not enough data to train TFT: "
            f"need at least ~{max_encoder_length + max_prediction_length + 1} hourly rows."
        )

    training_cutoff = int(data["time_idx"].max()) - max_prediction_length

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

    num_workers = int(os.getenv("DATALOADER_WORKERS", "2"))
    batch_size = int(os.getenv("TFT_BATCH_SIZE", "64"))

    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_loader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    # -----------------------
    # Model
    # -----------------------
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=float(os.getenv("TFT_LR", "1e-3")),
        hidden_size=int(os.getenv("TFT_HIDDEN_SIZE", "16")),
        attention_head_size=int(os.getenv("TFT_ATTENTION_HEADS", "2")),
        dropout=float(os.getenv("TFT_DROPOUT", "0.1")),
        hidden_continuous_size=int(os.getenv("TFT_HIDDEN_CONT_SIZE", "8")),
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # -----------------------
    # Training + checkpoint
    # -----------------------
    ckpt_dir = os.getenv("TFT_CKPT_DIR", tempfile.gettempdir())

    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        # safer filename (avoid formatting errors if metric missing at init)
        filename="tft-{epoch}",
        monitor="val_loss",
        mode="min",
        save_last=True,
        save_top_k=1,
    )

    trainer = Trainer(
        max_epochs=int(os.getenv("TFT_MAX_EPOCHS", "8")),
        accelerator="cpu",
        callbacks=[ckpt_cb],
        logger=False,
        enable_checkpointing=True,
    )
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

    checkpoint_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if not checkpoint_path:
        checkpoint_path = os.path.join(ckpt_dir, "tft-last.ckpt")
        trainer.save_checkpoint(checkpoint_path)

    # -----------------------
    # Optional MAE preview (never fail job)
    # -----------------------
    mae = float("nan")
    try:
        # Prefer prediction mode + return_y when available (most stable)
        try:
            pred, y = tft.predict(val_loader, mode="prediction", return_y=True)
        except TypeError:
            # older versions: no return_y
            pred = tft.predict(val_loader, mode="prediction")
            y = None

        if y is None:
            raise RuntimeError("predict(return_y=True) not supported; MAE preview skipped")

        pred_np = pred.detach().cpu().numpy() if hasattr(pred, "detach") else np.asarray(pred)
        y_np = y.detach().cpu().numpy() if hasattr(y, "detach") else np.asarray(y)

        mae = float(np.mean(np.abs(y_np - pred_np)))

    except Exception as e:
        print(f"WARN: TFT MAE preview failed (continuing anyway): {e}")

    # IMPORTANT: return NO model object
    return {
        "type": "tft",
        "mae_val": mae,
        "checkpoint_path": checkpoint_path,
        "target_used": target_note,
        "dropped_features": [{"name": n, "non_na_frac": f} for n, f in dropped],
        "kept_features": [{"name": n, "non_na_frac": f} for n, f in kept],
        "dataset_meta": {
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "known_categoricals": known_categoricals,
            "unknown_reals": unknown_reals,
            "min_non_na_frac": min_non_na_frac,
            "batch_size": batch_size,
            "num_workers": num_workers,
        },
    }
