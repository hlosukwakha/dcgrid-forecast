from __future__ import annotations
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from .config import POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD

def pg_url() -> str:
    return f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"

def engine() -> Engine:
    return create_engine(pg_url(), pool_pre_ping=True)

def ensure_schema() -> None:
    eng = engine()
    with eng.begin() as c:
        c.execute(text("CREATE SCHEMA IF NOT EXISTS dcgrid;"))
        c.execute(text("""
        CREATE TABLE IF NOT EXISTS dcgrid.features_hourly (
            ts_utc TIMESTAMPTZ PRIMARY KEY,
            region TEXT NOT NULL,
            demand_mw DOUBLE PRECISION,
            wind_mw DOUBLE PRECISION,
            solar_mw DOUBLE PRECISION,
            renew_mw DOUBLE PRECISION,
            renewable_share DOUBLE PRECISION,
            price_eur_mwh DOUBLE PRECISION,
            neg_price_flag INTEGER,
            neg_price_freq_7d DOUBLE PRECISION,
            price_spike_flag INTEGER,
            ramp_rate_mw_h DOUBLE PRECISION,
            renew_var_24h DOUBLE PRECISION,
            dc_share DOUBLE PRECISION,
            dc_load_mw DOUBLE PRECISION,
            stress_score DOUBLE PRECISION
        );
        """))
