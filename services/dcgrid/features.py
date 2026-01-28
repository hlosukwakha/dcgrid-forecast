from __future__ import annotations

import re
import pandas as pd


def to_hourly(df_qh: pd.DataFrame) -> pd.DataFrame:
    """
    Convert quarter-hourly data to hourly via mean aggregation.
    """
    if not isinstance(df_qh.index, pd.DatetimeIndex):
        raise ValueError("to_hourly expects df_qh indexed by a DatetimeIndex")
    # Pandas deprecates 'H' in favor of 'h'
    return df_qh.resample("1h").mean()


def _norm(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = s.replace("-", "_")
    return s


def _pick_series(df: pd.DataFrame, name: str) -> pd.Series | None:
    """
    Return a single Series for a column name.
    If df[name] yields multiple columns (duplicate names), collapse to one:
      - prefer numeric columns
      - use row-wise sum across duplicates (robust)
    """
    if name not in df.columns:
        return None

    obj = df[name]
    # If duplicate column names exist, df[name] returns a DataFrame
    if isinstance(obj, pd.DataFrame):
        # keep numeric-looking columns
        obj_num = obj.apply(pd.to_numeric, errors="coerce")
        # Sum duplicates (row-wise), ignoring NaNs
        return obj_num.sum(axis=1, min_count=1)
    # Series case
    return pd.to_numeric(obj, errors="coerce")


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common EirGrid column naming variants to canonical names:
      demand_mw, wind_mw, solar_mw
    Handles duplicate columns by later collapsing in _pick_series.
    """
    # normalize labels but keep duplicates (renaming can create more duplicates)
    new_cols = [_norm(c) for c in df.columns]
    df = df.copy()
    df.columns = new_cols

    # Map likely variants -> canonical
    rename_map = {}

    for c in df.columns:
        # demand
        if re.search(r"\b(demand|load)\b", c) and ("mw" in c or "demand" in c or "load" in c):
            # avoid mapping "wind" demand etc.
            if "wind" not in c and "solar" not in c:
                rename_map[c] = "demand_mw"

        # wind
        if re.search(r"\bwind\b", c):
            rename_map[c] = "wind_mw"

        # solar
        if re.search(r"\bsolar\b", c) or re.search(r"\bpv\b", c):
            rename_map[c] = "solar_mw"

    # Apply rename (may create duplicates intentionally; handled later)
    df = df.rename(columns=rename_map)
    return df


def build_features(
    region: str,
    system_h: pd.DataFrame,
    dc_q: pd.DataFrame | None,
    prices_h: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build an hourly feature table.
    Expected system_h columns (after standardization): demand_mw, wind_mw, solar_mw (some may be missing).
    """
    if not isinstance(system_h.index, pd.DatetimeIndex):
        raise ValueError("system_h must be indexed by a DatetimeIndex")

    df = _standardize_columns(system_h)

    # Pull canonical series safely (collapsing duplicates if needed)
    demand = _pick_series(df, "demand_mw")
    wind = _pick_series(df, "wind_mw")
    solar = _pick_series(df, "solar_mw")

    out = pd.DataFrame(index=df.index)
    out["region"] = region

    if demand is not None:
        out["demand_mw"] = demand
    else:
        out["demand_mw"] = pd.NA

    out["wind_mw"] = wind if wind is not None else 0.0
    out["solar_mw"] = solar if solar is not None else 0.0

    # Renewable total (this is where your crash happened)
    out["renew_mw"] = out["wind_mw"].fillna(0) + out["solar_mw"].fillna(0)
    out["renew_share"] = (out["renew_mw"] / out["demand_mw"]).replace([pd.NA, pd.NaT], pd.NA)

    # Variability / ramps (simple, transparent)
    out["demand_ramp_mw_h"] = out["demand_mw"].diff()
    out["renew_ramp_mw_h"] = out["renew_mw"].diff()
    out["renew_var_24h"] = out["renew_mw"].rolling(24).std()
    out["demand_var_24h"] = out["demand_mw"].rolling(24).std()

    # Optional: merge quarterly DC proxy (forward fill within quarter)
    if dc_q is not None and not dc_q.empty:
        # dc_q index is quarter-end timestamps (UTC), column dc_gwh
        dc = dc_q.copy()
        dc = dc.rename(columns={"dc_gwh": "dc_gwh_q"})
        # align by timestamp, forward-fill to hourly
        dc = dc.reindex(out.index.union(dc.index)).sort_index().ffill()
        out["dc_gwh_q"] = dc.loc[out.index, "dc_gwh_q"]
    else:
        out["dc_gwh_q"] = pd.NA

    # Optional: prices (expects columns like price_eur_mwh)
    if prices_h is not None and not prices_h.empty:
        p = prices_h.copy()
        if not isinstance(p.index, pd.DatetimeIndex):
            # if it's a column, try to fix
            if "ts_utc" in p.columns:
                p["ts_utc"] = pd.to_datetime(p["ts_utc"], utc=True)
                p = p.set_index("ts_utc")
        # normalize
        p.columns = [_norm(c) for c in p.columns]
        # pick a price column
        price_col = None
        for c in p.columns:
            if "price" in c and ("eur" in c or "euro" in c or "mwh" in c):
                price_col = c
                break
        if price_col:
            out["price_eur_mwh"] = pd.to_numeric(p[price_col], errors="coerce").reindex(out.index)
            out["neg_price_flag"] = (out["price_eur_mwh"] < 0).astype("float")
            out["neg_price_freq_24h"] = out["neg_price_flag"].rolling(24).mean()
            out["price_spike_z"] = (out["price_eur_mwh"] - out["price_eur_mwh"].rolling(168).mean()) / out[
                "price_eur_mwh"
            ].rolling(168).std()
        else:
            out["price_eur_mwh"] = pd.NA
            out["neg_price_freq_24h"] = pd.NA
            out["price_spike_z"] = pd.NA
    else:
        out["price_eur_mwh"] = pd.NA
        out["neg_price_freq_24h"] = pd.NA
        out["price_spike_z"] = pd.NA

    # Stress score (simple composite, works even if prices/DC missing)
    def z(x: pd.Series, win: int) -> pd.Series:
        mu = x.rolling(win).mean()
        sd = x.rolling(win).std()
        return (x - mu) / sd

    out["z_demand_ramp"] = z(out["demand_ramp_mw_h"], 168)
    out["z_renew_var"] = z(out["renew_var_24h"], 168)
    out["z_price_spike"] = z(out["price_eur_mwh"], 168) if out["price_eur_mwh"].notna().any() else pd.NA

    # Weighted sum; missing components ignored via fillna(0)
    price_spike_z = pd.to_numeric(out["price_spike_z"], errors="coerce")
    out["stress_score"] = (
        out["z_demand_ramp"].fillna(0) * 0.4
        + out["z_renew_var"].fillna(0) * 0.4
        + price_spike_z.fillna(0) * 0.2
    )

    return out
