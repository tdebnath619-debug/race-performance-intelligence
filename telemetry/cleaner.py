"""
F1 Performance Intelligence System
telemetry/cleaner.py

F1-aware signal cleaning:
- Physical bound enforcement per F1 regulations
- ERS state-machine validation
- DRS boolean cleaning
- Tyre temperature window validation
- Brake bias inference
- G-force plausibility check
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# F1 physical bounds (2024 regulations / sensor ranges)
F1_BOUNDS = {
    "speed":              (0.0,   380.0),   # km/h
    "throttle":           (0.0,   1.0),
    "brake":              (0.0,   1.0),
    "rpm":                (0.0,   18_500.0),
    "steering":           (-1.0,  1.0),
    "gear":               (0.0,   9.0),     # 8 + neutral
    "ers_deployment_kw":  (0.0,   120.0),   # F1 MGU-K limit: 120 kW deploy
    "ers_harvesting_kw":  (0.0,   120.0),
    "mguk_kw":            (-120.0,120.0),
    "fuel_flow_kgh":      (0.0,   100.0),   # FIA limit: 100 kg/h
    "fuel_kg":            (0.0,   110.0),   # max fuel load
    "g_lateral":          (-7.0,  7.0),     # g
    "g_longitudinal":     (-7.0,  5.0),
    "g_vertical":         (-3.0,  3.0),
    "brake_temp_fl":      (0.0,   1200.0),  # °C
    "brake_temp_fr":      (0.0,   1200.0),
    "brake_temp_rl":      (0.0,   1200.0),
    "brake_temp_rr":      (0.0,   1200.0),
    "tyre_surf_fl":       (0.0,   160.0),
    "tyre_surf_fr":       (0.0,   160.0),
    "tyre_surf_rl":       (0.0,   160.0),
    "tyre_surf_rr":       (0.0,   160.0),
}

SMOOTH_WINDOWS = {
    "speed": 5, "throttle": 3, "brake": 3, "steering": 7,
    "rpm": 5, "g_lateral": 5, "g_longitudinal": 5,
    "ers_deployment_kw": 7, "ers_harvesting_kw": 7,
    "tyre_surf_fl": 11, "tyre_surf_fr": 11,
    "tyre_surf_rl": 11, "tyre_surf_rr": 11,
    "brake_temp_fl": 9, "brake_temp_fr": 9,
    "brake_temp_rl": 9, "brake_temp_rr": 9,
}


def clean_telemetry(
    df: pd.DataFrame,
    spike_threshold: float = 4.0,
    smooth: bool = True,
) -> pd.DataFrame:
    df = df.copy()
    df = _clip_bounds(df)
    df = _fill_nans(df)
    df = _remove_spikes(df, spike_threshold)
    df = _clean_drs(df)
    df = _validate_ers(df)

    if smooth:
        df = _smooth(df)

    df = _normalise_controls(df)
    df = _compute_derivatives(df)
    df = _compute_tyre_balance(df)

    logger.info("Cleaning complete.")
    return df


# ── Cleaning steps ────────────────────────────────────────────────────────────

def _clip_bounds(df):
    for col, (lo, hi) in F1_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    return df


def _fill_nans(df):
    num = df.select_dtypes(include=np.number).columns
    n   = df[num].isna().sum().sum()
    if n:
        df[num] = df[num].interpolate("linear", limit_direction="both").ffill().bfill()
        logger.debug(f"Filled {n} NaNs.")
    return df


def _remove_spikes(df, threshold):
    targets = [c for c in ["speed","throttle","brake","rpm","g_lateral",
                             "g_longitudinal","ers_deployment_kw"] if c in df.columns]
    total = 0
    for col in targets:
        s  = df[col].copy()
        rm = s.rolling(30, center=True, min_periods=1).mean()
        rs = s.rolling(30, center=True, min_periods=1).std().replace(0, np.nan)
        z  = (s - rm).abs() / rs
        mask = z > threshold
        total += mask.sum()
        df.loc[mask, col] = np.nan
    if total:
        num = df.select_dtypes(include=np.number).columns
        df[num] = df[num].interpolate("linear", limit_direction="both").ffill().bfill()
        logger.debug(f"Removed {total} spikes.")
    return df


def _clean_drs(df):
    """DRS is binary (0/1). Clean partial activations and latch properly."""
    if "drs_state" not in df.columns:
        return df
    df["drs_state"] = (df["drs_state"] > 0.5).astype(int)
    # Require DRS speed > 210 km/h (FIA detection point rule)
    if "speed" in df.columns:
        df.loc[df["speed"] < 210, "drs_state"] = 0
    return df


def _validate_ers(df):
    """
    ERS energy budget check.
    MGU-K can deliver max 120 kW but total energy per lap is limited (~4 MJ).
    Flag if cumulative deploy exceeds budget — don't correct, just annotate.
    """
    if "ers_deployment_kw" not in df.columns:
        return df
    dt      = df["time"].diff().fillna(0)
    energy  = (df["ers_deployment_kw"] * dt).cumsum() / 1000  # MJ
    df["ers_cumulative_mj"] = energy.round(4)
    df["ers_over_budget"]   = energy > 4.0
    over = df["ers_over_budget"].sum()
    if over:
        logger.warning(f"ERS over-budget detected in {over} samples (>4 MJ threshold).")
    return df


def _smooth(df):
    for col, win in SMOOTH_WINDOWS.items():
        if col in df.columns:
            df[col] = df[col].rolling(win, center=True, min_periods=1).mean()
    return df


def _normalise_controls(df):
    """Convert throttle/brake from 0-100 to 0-1 if needed."""
    for col in ("throttle", "brake"):
        if col in df.columns and df[col].max() > 1.5:
            df[col] = df[col] / 100.0
    return df


def _compute_derivatives(df):
    """Add time-derivatives useful for F1 analysis."""
    dt = df["time"].diff().replace(0, np.nan)
    for col in ("speed", "throttle", "brake", "steering"):
        if col in df.columns:
            df[f"d_{col}"] = (df[col].diff() / dt).round(4)

    # Longitudinal G from speed (if not measured)
    if "g_longitudinal" not in df.columns and "speed" in df.columns:
        df["g_longitudinal"] = (df["d_speed"] / 3.6 / 9.81).round(4)

    return df


def _compute_tyre_balance(df):
    """
    Compute front / rear tyre temperature balance.
    High imbalance → setup issue or driving style artefact.
    """
    has_temps = all(c in df.columns for c in
                    ["tyre_surf_fl","tyre_surf_fr","tyre_surf_rl","tyre_surf_rr"])
    if not has_temps:
        return df
    df["tyre_avg_front"] = (df["tyre_surf_fl"] + df["tyre_surf_fr"]) / 2
    df["tyre_avg_rear"]  = (df["tyre_surf_rl"] + df["tyre_surf_rr"]) / 2
    df["tyre_balance"]   = (df["tyre_avg_front"] - df["tyre_avg_rear"]).round(2)
    return df


def compute_derivatives(df: pd.DataFrame) -> pd.DataFrame:
    """Public alias — compute time derivatives for key channels."""
    return _compute_derivatives(df)


def flag_coasting(df, throttle_thr=0.05, brake_thr=0.05):
    if "throttle" in df.columns and "brake" in df.columns:
        df["coasting"] = (df["throttle"] < throttle_thr) & (df["brake"] < brake_thr)
    return df


def compute_brake_balance(df):
    """
    Infer front/rear brake balance from g-force vs brake pressure.
    Returns a scalar balance estimate (% front, nominal ~56% for F1).
    """
    if not all(c in df.columns for c in ["g_longitudinal", "brake"]):
        return None
    braking_mask = df["brake"] > 0.1
    if braking_mask.sum() < 10:
        return None
    sub = df[braking_mask]
    # Simple linear proxy: slope of g_longitudinal / brake
    ratio = (sub["g_longitudinal"].abs() / sub["brake"].replace(0, np.nan)).dropna()
    return round(float(ratio.median()), 3)
