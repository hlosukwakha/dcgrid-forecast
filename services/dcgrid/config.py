from __future__ import annotations
import os

def env(key: str, default: str | None = None) -> str:
    v = os.getenv(key, default)
    if v is None:
        raise RuntimeError(f"Missing env var: {key}")
    return v

REGION = env("REGION", "IE")
TZ = env("TZ", "Europe/Dublin")

EIRGRID_SYSTEM_DATA_URL = env(
    "EIRGRID_SYSTEM_DATA_URL",
    "https://cms.eirgrid.ie/sites/default/files/publications/System-Data-Qtr-Hourly-2025-v11.xlsx",
)

POSTGRES_HOST = env("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(env("POSTGRES_PORT", "5432"))
POSTGRES_DB = env("POSTGRES_DB", "dcgrid")
POSTGRES_USER = env("POSTGRES_USER", "dcgrid")
POSTGRES_PASSWORD = env("POSTGRES_PASSWORD", "dcgrid")

MINIO_ENDPOINT = env("MINIO_ENDPOINT", "http://minio:9000")
MINIO_ACCESS_KEY = env("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = env("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = env("MINIO_BUCKET", "dcgrid")

MODEL_NAME = env("MODEL_NAME", "tft")  # tft | xgb | prophet
FORECAST_HORIZON_H = int(env("FORECAST_HORIZON_H", "168"))      # 7 days
CONTEXT_LENGTH_H = int(env("CONTEXT_LENGTH_H", "336"))         # 14 days

# --- Model artifact keys in MinIO (Option B: TFT checkpoints, not pickle) ---

MODEL_NAME_L = MODEL_NAME.lower()

MODEL_OBJECT_KEY = os.getenv("MODEL_OBJECT_KEY") or (
    f"models/{REGION}/tft.ckpt"
    if MODEL_NAME_L == "tft"
    else f"models/{REGION}/{MODEL_NAME}.pkl"
)

MODEL_META_KEY = os.getenv("MODEL_META_KEY") or (
    f"models/{REGION}/tft.meta.json"
    if MODEL_NAME_L == "tft"
    else ""
)
