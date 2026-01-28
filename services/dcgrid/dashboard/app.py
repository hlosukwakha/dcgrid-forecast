from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from sqlalchemy import text

from dcgrid.db import engine

st.set_page_config(page_title="DCGrid Forecast Dashboard", layout="wide")

st.title("DCGrid Forecast: Data-centre load + grid stress (Ireland-first)")

region = os.getenv("REGION", "IE")
eng = engine()


@st.cache_data(ttl=60)
def load_features(n: int = 24 * 90) -> pd.DataFrame:
    q = text(
        f"""
        SELECT
            ts_utc,
            region,
            demand_mw,
            wind_mw,
            solar_mw,
            renewable_share,
            price_eur_mwh,
            neg_price_freq_7d,
            renew_var_24h,
            dc_load_mw,
            stress_score
        FROM dcgrid.features_hourly
        WHERE region = :r
        ORDER BY ts_utc DESC
        LIMIT {n}
        """
    )
    df = pd.read_sql(q, eng, params={"r": region}, parse_dates=["ts_utc"])
    return df.sort_values("ts_utc")


df = load_features()

if df.empty:
    st.warning("No data yet. Run the ingest job to load features into Postgres.")
    st.stop()

# Normalize timestamps + numeric types to avoid Streamlit mixed-type chart errors.
df = df.copy()
df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")

numeric_cols = [
    "demand_mw",
    "wind_mw",
    "solar_mw",
    "renewable_share",
    "price_eur_mwh",
    "neg_price_freq_7d",
    "renew_var_24h",
    "dc_load_mw",
    "stress_score",
]
for c in numeric_cols:
    if c not in df.columns:
        df[c] = pd.NA
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    n = st.slider("Lookback window (hours)", min_value=24, max_value=24 * 365, value=24 * 90, step=24)
    if n != 24 * 90:
        df = load_features(n=n)
        df = df.copy()
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
        for c in numeric_cols:
            if c not in df.columns:
                df[c] = pd.NA
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["ts_utc"]).sort_values("ts_utc")

    st.caption(f"Region: **{region}**")

# Layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demand vs Data-centre load proxy (MW)")

    plot_df = df.set_index("ts_utc")

    # Only plot dc_load_mw if there is at least some data
    if plot_df["dc_load_mw"].notna().any():
        st.line_chart(plot_df, y=["demand_mw", "dc_load_mw"])
    else:
        st.info("DC load series not available yet (CSO not ingested). Showing demand only.")
        st.line_chart(plot_df, y=["demand_mw"])

with col2:
    st.subheader("Stress score (composite)")

    plot_df = df.set_index("ts_utc")
    st.line_chart(plot_df, y=["stress_score"])

st.subheader("Drivers")

drivers = df.set_index("ts_utc")

driver_cols = ["price_eur_mwh", "neg_price_freq_7d", "renew_var_24h"]
available = [c for c in driver_cols if c in drivers.columns and drivers[c].notna().any()]

if not available:
    st.info("No driver signals available yet (prices not ingested; renew variability may be missing).")
else:
    st.line_chart(drivers, y=available)

with st.expander("Raw (latest rows)"):
    st.dataframe(df.tail(200), use_container_width=True)

st.caption(
    "Note: DC load is a proxy; with CSO enabled it can be derived from quarterly DC consumption "
    "shares applied to system demand. Prices are optional (Ember ingestion)."
)
