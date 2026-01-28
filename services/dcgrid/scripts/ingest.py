from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from dcgrid import config
from dcgrid.db import engine, ensure_schema
from dcgrid.data.eirgrid_system_data import load_system_qh
from dcgrid.data.cso_datacentres import load_quarterly_gwh_from_keyfindings
from dcgrid.features import to_hourly, build_features


def _align_to_db_schema(features: pd.DataFrame) -> pd.DataFrame:
    """
    Align the feature dataframe to the existing Postgres table schema:
      dcgrid.features_hourly columns (16):
        ts_utc, region, demand_mw, wind_mw, solar_mw, renew_mw, renewable_share,
        price_eur_mwh, neg_price_flag, neg_price_freq_7d, price_spike_flag,
        ramp_rate_mw_h, renew_var_24h, dc_share, dc_load_mw, stress_score
    """
    df = features.copy()

    # --- Column renames to match DB ---
    rename_map = {
        "renew_share": "renewable_share",
        "demand_ramp_mw_h": "ramp_rate_mw_h",
        # if your pipeline produces 24h neg-price frequency, map it to the DB field
        # (semantics differ; ideally compute 7d in features.py, but this unblocks ingestion)
        "neg_price_freq_24h": "neg_price_freq_7d",
    }
    df = df.rename(columns=rename_map)

    # --- Ensure required columns exist (create from available signals where possible) ---
    # renewable_share
    if "renewable_share" not in df.columns:
        if "renew_mw" in df.columns and "demand_mw" in df.columns:
            df["renewable_share"] = (df["renew_mw"] / df["demand_mw"]).replace([pd.NA, pd.NaT], pd.NA)
        else:
            df["renewable_share"] = pd.NA

    # neg_price_flag (DB expects integer)
    if "neg_price_flag" not in df.columns:
        if "price_eur_mwh" in df.columns:
            df["neg_price_flag"] = (pd.to_numeric(df["price_eur_mwh"], errors="coerce") < 0).astype("Int64")
        else:
            df["neg_price_flag"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    else:
        df["neg_price_flag"] = pd.to_numeric(df["neg_price_flag"], errors="coerce").astype("Int64")

    # neg_price_freq_7d
    if "neg_price_freq_7d" not in df.columns:
        # If you have neg_price_flag, compute a 7-day rolling mean (168 hours) as a proper 7d frequency.
        if "neg_price_flag" in df.columns:
            df["neg_price_freq_7d"] = pd.to_numeric(df["neg_price_flag"], errors="coerce").rolling(168).mean()
        else:
            df["neg_price_freq_7d"] = pd.NA

    # price_spike_flag (DB expects integer)
    if "price_spike_flag" not in df.columns:
        # If you have a z-score style column, derive a flag from it.
        if "price_spike_z" in df.columns:
            z = pd.to_numeric(df["price_spike_z"], errors="coerce")
            df["price_spike_flag"] = (z > 2.0).astype("Int64")
        else:
            df["price_spike_flag"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
    else:
        df["price_spike_flag"] = pd.to_numeric(df["price_spike_flag"], errors="coerce").astype("Int64")

    # ramp_rate_mw_h (if missing but demand_mw exists)
    if "ramp_rate_mw_h" not in df.columns:
        if "demand_mw" in df.columns:
            df["ramp_rate_mw_h"] = pd.to_numeric(df["demand_mw"], errors="coerce").diff()
        else:
            df["ramp_rate_mw_h"] = pd.NA

    # dc_share, dc_load_mw (optional; keep nullable)
    if "dc_share" not in df.columns:
        df["dc_share"] = pd.NA
    if "dc_load_mw" not in df.columns:
        df["dc_load_mw"] = pd.NA

    # Ensure numeric types for float columns where possible
    float_cols = [
        "demand_mw",
        "wind_mw",
        "solar_mw",
        "renew_mw",
        "renewable_share",
        "price_eur_mwh",
        "neg_price_freq_7d",
        "ramp_rate_mw_h",
        "renew_var_24h",
        "dc_share",
        "dc_load_mw",
        "stress_score",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Final column selection/order to match DB exactly ---
    db_cols = [
        "ts_utc",
        "region",
        "demand_mw",
        "wind_mw",
        "solar_mw",
        "renew_mw",
        "renewable_share",
        "price_eur_mwh",
        "neg_price_flag",
        "neg_price_freq_7d",
        "price_spike_flag",
        "ramp_rate_mw_h",
        "renew_var_24h",
        "dc_share",
        "dc_load_mw",
        "stress_score",
    ]

    missing = [c for c in db_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Internal error: missing required DB columns after alignment: {missing}")

    return df[db_cols]


def main() -> None:
    ensure_schema()

    # --- Required: EirGrid system data (qh -> hourly) ---
    if not getattr(config, "EIRGRID_SYSTEM_DATA_URL", None):
        raise RuntimeError("EIRGRID_SYSTEM_DATA_URL is required (set it in .env).")

    system_qh = load_system_qh(config.EIRGRID_SYSTEM_DATA_URL)
    system_h = to_hourly(system_qh)

    # --- Optional: CSO quarterly DC consumption ---
    dc_q = None
    cso_keyfindings_url = getattr(config, "CSO_DC_TABLE2_XLS_URL", None)
    if cso_keyfindings_url:
        try:
            dc_q = load_quarterly_gwh_from_keyfindings(cso_keyfindings_url)
        except Exception as e:
            print(f"WARNING: CSO ingest failed; continuing without CSO series. Reason: {e}")
            dc_q = None
    else:
        print("CSO_DC_TABLE2_XLS_URL not set; skipping CSO data-centre series")

    # --- Optional: prices (Ember) ---
    prices = None
    ember_zip_url = getattr(config, "EMBER_DA_PRICE_ZIP_URL", None)
    if ember_zip_url:
        try:
            from dcgrid.data.ember_prices import download_ember_zip, load_ember_prices_zip

            zip_bytes = download_ember_zip(ember_zip_url)
            prices = load_ember_prices_zip(zip_bytes, country_code="Ireland")
        except Exception as e:
            print(f"WARNING: Ember price ingest failed; continuing without prices. Reason: {e}")
            prices = None
    else:
        print("EMBER_DA_PRICE_ZIP_URL not set; skipping prices")

    # --- Build features ---
    features = build_features(config.REGION, system_h, dc_q, prices_h=prices)

    # Ensure timestamp column is clean and unique
    features = features.reset_index(names="ts_utc")
    features["ts_utc"] = pd.to_datetime(features["ts_utc"], utc=True)
    features = features.drop_duplicates(subset=["region", "ts_utc"]).sort_values(["region", "ts_utc"])

    # Align to DB schema (prevents to_sql failures)
    features_db = _align_to_db_schema(features)

    # --- Persist ---
    eng = engine()
    with eng.begin() as c:
        # Delete only if the table exists (first-run friendly)
        c.execute(
            text(
                """
                DO $$
                BEGIN
                    IF to_regclass('dcgrid.features_hourly') IS NOT NULL THEN
                        DELETE FROM dcgrid.features_hourly WHERE region = :r;
                    END IF;
                END $$;
                """
            ),
            {"r": config.REGION},
        )

        features_db.to_sql(
            "features_hourly",
            con=c,
            schema="dcgrid",
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )

    print(
        f"Ingested {len(features_db):,} hourly rows into dcgrid.features_hourly for region={config.REGION}"
    )


if __name__ == "__main__":
    main()
