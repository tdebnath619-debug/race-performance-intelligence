"""
F1 Performance Intelligence System
telemetry/loader.py

Loads and standardises F1 telemetry from:
  - FastF1 API  (live / historical official F1 data)
  - CSV export  (MoTeC, Pi Toolbox, custom logger)

F1-specific channels beyond sim-racing:
  - ERS deployment (harvesting / delivery kW)
  - DRS state
  - tyre compound + age
  - fuel load + flow rate
  - MGU-K / MGU-H power split
  - g-forces (lateral / longitudinal)
  - suspension travel
  - brake temp (FL, FR, RL, RR)
  - tyre surface / carcass temp
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ── Column alias map (MoTeC / Pi / iRacing → internal) ───────────────────────
COLUMN_ALIASES = {
    # Core
    "timestamp": "time", "time_s": "time", "elapsed": "time",
    "dist": "distance", "lap_dist": "distance", "track_pos": "distance",
    "speed_kmh": "speed", "velocity": "speed", "car_speed": "speed",
    "gas": "throttle", "throttle_input": "throttle",
    "brake_input": "brake", "brake_pct": "brake",
    "current_gear": "gear", "engine_rpm": "rpm",
    "steer": "steering", "steer_angle": "steering",
    "current_lap_time": "lap_time", "laptime": "lap_time",
    # F1 Power Unit
    "ers_deploy": "ers_deployment_kw",
    "ers_harvest": "ers_harvesting_kw",
    "mguk_power": "mguk_kw",
    "mguh_power": "mguh_kw",
    "fuel_flow": "fuel_flow_kgh",
    "fuel_remaining": "fuel_kg",
    # DRS
    "drs": "drs_state",
    "drs_active": "drs_state",
    # G-forces
    "g_lat": "g_lateral",
    "g_lon": "g_longitudinal",
    "g_vert": "g_vertical",
    # Brake temps
    "brake_temp_fl": "brake_temp_fl",
    "brake_temp_fr": "brake_temp_fr",
    "brake_temp_rl": "brake_temp_rl",
    "brake_temp_rr": "brake_temp_rr",
    # Tyre temps (surface)
    "tyre_temp_fl": "tyre_surf_fl",
    "tyre_temp_fr": "tyre_surf_fr",
    "tyre_temp_rl": "tyre_surf_rl",
    "tyre_temp_rr": "tyre_surf_rr",
    # Suspension
    "susp_fl": "susp_travel_fl",
    "susp_fr": "susp_travel_fr",
    "susp_rl": "susp_travel_rl",
    "susp_rr": "susp_travel_rr",
}

REQUIRED_COLUMNS = ["time", "speed"]

F1_CHANNELS = [
    "distance", "throttle", "brake", "gear", "rpm", "steering", "lap_time",
    "drs_state", "ers_deployment_kw", "ers_harvesting_kw", "mguk_kw", "mguh_kw",
    "fuel_flow_kgh", "fuel_kg",
    "g_lateral", "g_longitudinal", "g_vertical",
    "brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr",
    "tyre_surf_fl", "tyre_surf_fr", "tyre_surf_rl", "tyre_surf_rr",
    "susp_travel_fl", "susp_travel_fr", "susp_travel_rl", "susp_travel_rr",
]


def load_telemetry(
    file_path: str | Path,
    lap_number: int | None = None,
    resample_hz: float | None = None,
) -> pd.DataFrame:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading: {file_path.name}")
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.rename(columns=COLUMN_ALIASES)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    df = _normalise_time(df)

    if "distance" not in df.columns:
        speed_ms = df["speed"] / 3.6
        dt = df["time"].diff().fillna(0)
        df["distance"] = (speed_ms * dt).cumsum()

    if "lap_number" not in df.columns:
        df = _infer_lap_number(df)

    if lap_number is not None:
        df = df[df["lap_number"] == lap_number].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"Lap {lap_number} not found.")

    if resample_hz:
        df = _resample(df, resample_hz)

    df = df.sort_values("time").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} samples | laps: {sorted(df['lap_number'].unique())}")
    return df


def load_fastf1(session_year: int, gp: str, session: str = "R",
                driver: str = "VER") -> pd.DataFrame:
    """
    Load official F1 telemetry via FastF1.
    Requires: pip install fastf1
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache("data/cache")
        sess = fastf1.get_session(session_year, gp, session)
        sess.load()
        lap = sess.laps.pick_driver(driver).pick_fastest()
        tel = lap.get_telemetry()

        df = pd.DataFrame({
            "time":              tel["Time"].dt.total_seconds(),
            "distance":          tel["Distance"],
            "speed":             tel["Speed"],
            "throttle":          tel["Throttle"] / 100.0,
            "brake":             tel["Brake"].astype(float),
            "gear":              tel["nGear"],
            "rpm":               tel["RPM"],
            "drs_state":         tel["DRS"],
            "x":                 tel["X"],
            "y":                 tel["Y"],
            "z":                 tel["Z"],
        })
        df["lap_number"] = 0
        df["driver"]     = driver
        df["session"]    = f"{session_year} {gp} {session}"
        logger.info(f"FastF1: {driver} {session_year} {gp} — {len(df)} samples")
        return df
    except ImportError:
        logger.error("fastf1 not installed. Run: pip install fastf1")
        raise


def list_laps(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lap, g in df.groupby("lap_number"):
        rows.append({
            "lap":          lap,
            "samples":      len(g),
            "duration_s":   round(g["time"].max() - g["time"].min(), 3),
            "distance_m":   round(g["distance"].max() - g["distance"].min(), 1),
            "max_speed":    round(g["speed"].max(), 1),
            "avg_throttle": round(g["throttle"].mean() * 100, 1) if "throttle" in g else None,
        })
    return pd.DataFrame(rows)


def _normalise_time(df):
    t = df["time"]
    if t.dtype == object:
        try:
            df["time"] = pd.to_timedelta(t).dt.total_seconds()
            return df
        except Exception:
            pass
    if t.max() > 10_000 and t.min() >= 0:
        df["time"] = t / 1000.0
    return df


def _infer_lap_number(df):
    if "lap_time" in df.columns:
        reset = df["lap_time"].diff() < -5.0
    else:
        reset = df["distance"].diff() < -50.0
    df["lap_number"] = reset.cumsum().astype(int)
    return df


def _resample(df, hz):
    t0, t1 = df["time"].min(), df["time"].max()
    t_new  = np.arange(t0, t1, 1.0 / hz)
    out    = {}
    for col in df.select_dtypes(include=np.number).columns:
        out[col] = np.interp(t_new, df["time"], df[col])
    return pd.DataFrame(out)
