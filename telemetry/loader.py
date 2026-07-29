"""
telemetry/loader.py — Telemetry ingestion and standardisation.
Supports CSV (MoTeC, Pi Toolbox, ACC, iRacing) and FastF1 (via loader_fastf1.py).
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

REQUIRED_SIGNALS = ["time", "speed"]
RECOMMENDED_SIGNALS = ["distance","throttle","brake","steering","gear","rpm"]
OPTIONAL_SIGNALS = ["lateral_g","longitudinal_g","tyre_temp_fl","tyre_temp_fr",
                    "tyre_temp_rl","tyre_temp_rr","brake_temp_fl","brake_temp_fr",
                    "brake_temp_rl","brake_temp_rr","fuel_kg","ers_deployment_kw",
                    "ers_harvesting_kw","drs_state","susp_fl","susp_fr","susp_rl","susp_rr"]

_ALIASES = {
    "timestamp":"time","time_s":"time","elapsed_s":"time","t":"time",
    "dist":"distance","lap_dist":"distance","track_pos":"distance",
    "speed_kmh":"speed","velocity":"speed","car_speed":"speed","v":"speed",
    "gas":"throttle","throttle_input":"throttle","throttle_pct":"throttle","tps":"throttle",
    "brake_input":"brake","brake_pct":"brake","bps":"brake",
    "steer":"steering","steer_angle":"steering","steering_wheel":"steering","sw":"steering",
    "current_gear":"gear","ngear":"gear",
    "engine_rpm":"rpm","n":"rpm",
    "g_lat":"lateral_g","lat_g":"lateral_g",
    "g_lon":"longitudinal_g","lon_g":"longitudinal_g","g_long":"longitudinal_g",
    "ers_deploy":"ers_deployment_kw","ers_harvest":"ers_harvesting_kw",
    "fuel_remaining":"fuel_kg","drs":"drs_state",
}

_BOUNDS = {
    "speed":(0,380),"throttle":(0,1),"brake":(0,1),"steering":(-1,1),
    "gear":(0,9),"rpm":(0,20000),"lateral_g":(-7,7),"longitudinal_g":(-7,5),
    "tyre_temp_fl":(0,200),"tyre_temp_fr":(0,200),"tyre_temp_rl":(0,200),"tyre_temp_rr":(0,200),
    "brake_temp_fl":(0,1200),"brake_temp_fr":(0,1200),"brake_temp_rl":(0,1200),"brake_temp_rr":(0,1200),
    "fuel_kg":(0,110),"ers_deployment_kw":(0,120),"ers_harvesting_kw":(0,120),
}


@dataclass
class TelemetryLap:
    data:             pd.DataFrame
    lap_number:       int
    driver:           str
    session:          str
    source_file:      str
    sample_rate_hz:   float
    signals_present:  list
    warnings:         list = field(default_factory=list)

    @property
    def lap_time_s(self):
        return float(self.data["time"].max() - self.data["time"].min())

    @property
    def distance_m(self):
        return float(self.data["distance"].max() - self.data["distance"].min())

    @property
    def n_samples(self): return len(self.data)

    def has_signal(self, name): return name in self.data.columns

    def __repr__(self):
        return (f"TelemetryLap(driver='{self.driver}', lap={self.lap_number}, "
                f"time={self.lap_time_s:.3f}s, dist={self.distance_m:.0f}m, "
                f"Hz={self.sample_rate_hz:.0f}, signals={len(self.signals_present)})")


def load_csv(file_path, driver="UNKNOWN", session="UNKNOWN", lap_number=None):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    log.info("Loading %s", file_path.name)
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[\s\-]+","_",regex=True)
    df = df.rename(columns=_ALIASES)

    missing = [s for s in REQUIRED_SIGNALS if s not in df.columns]
    if missing:
        raise ValueError(f"Required signals missing: {missing}\nAvailable: {list(df.columns)}")

    df = _normalise_time(df)
    df = _normalise_controls(df)
    df = _normalise_steering(df)
    df = _compute_distance(df)
    df = _infer_lap_number(df)

    laps_found = sorted(df["_lap"].unique())
    log.info("Laps detected: %s", laps_found)

    if lap_number is not None and lap_number not in laps_found:
        raise ValueError(f"Lap {lap_number} not found. Available: {laps_found}")

    results = []
    for lap_idx in laps_found:
        if lap_number is not None and lap_idx != lap_number:
            continue
        lap_df = df[df["_lap"]==lap_idx].copy().drop(columns=["_lap"]).reset_index(drop=True)
        lap_df["time"]     = lap_df["time"]     - lap_df["time"].iloc[0]
        lap_df["distance"] = lap_df["distance"] - lap_df["distance"].iloc[0]

        hz   = _estimate_hz(lap_df)
        sigs = [c for c in lap_df.columns if c in REQUIRED_SIGNALS+RECOMMENDED_SIGNALS+OPTIONAL_SIGNALS]
        warn = _validate_bounds(lap_df)

        tl = TelemetryLap(data=lap_df, lap_number=lap_idx, driver=driver,
                          session=session, source_file=file_path.name,
                          sample_rate_hz=hz, signals_present=sigs, warnings=warn)
        results.append(tl)
        log.info("Loaded: %s", tl)
        for w in warn: log.warning("  [WARN] %s", w)

    return results


def lap_summary(laps):
    rows = []
    for lap in laps:
        rows.append({"driver":lap.driver,"session":lap.session,"lap_number":lap.lap_number,
                     "lap_time_s":round(lap.lap_time_s,3),"distance_m":round(lap.distance_m,0),
                     "sample_rate":round(lap.sample_rate_hz,1),"n_samples":lap.n_samples,
                     "signals":len(lap.signals_present),"warnings":len(lap.warnings),"source":lap.source_file})
    return pd.DataFrame(rows)


def _normalise_time(df):
    t = df["time"]
    if t.dtype == object:
        try: df["time"] = pd.to_timedelta(t).dt.total_seconds(); return df
        except Exception: pass
    if t.max() > 10000 and t.min() >= 0:
        log.info("Time appears milliseconds — converting."); df["time"] = t/1000.0
    return df

def _normalise_controls(df):
    for col in ("throttle","brake"):
        if col in df.columns and df[col].max() > 1.5:
            log.info("Normalising '%s' 0-100 to 0-1.", col); df[col] = df[col]/100.0
    return df

def _normalise_steering(df):
    if "steering" not in df.columns: return df
    if df["steering"].abs().max() > 1.5:
        scale = df["steering"].abs().max()
        log.info("Normalising steering ±%.0f to ±1.", scale)
        df["steering"] = df["steering"]/scale
    return df

def _compute_distance(df):
    if "distance" not in df.columns:
        log.info("Computing distance from speed integration.")
        df["distance"] = (df["speed"]/3.6 * df["time"].diff().fillna(0)).cumsum()
    return df

def _infer_lap_number(df):
    if "lap_number" in df.columns:
        df["_lap"] = df["lap_number"].astype(int); return df
    if "lap_time" in df.columns:
        reset = df["lap_time"].diff() < -5.0
    else:
        reset = df["distance"].diff() < -50.0
    df["_lap"] = reset.cumsum().astype(int)
    return df

def _estimate_hz(df):
    dt = df["time"].diff().dropna(); dt = dt[dt>0]
    return round(1.0/float(dt.median()),1) if not dt.empty else 0.0

def _validate_bounds(df):
    warnings = []
    for col,(lo,hi) in _BOUNDS.items():
        if col not in df.columns: continue
        n = int(((df[col]<lo)|(df[col]>hi)).sum())
        if n: warnings.append(f"'{col}': {n} samples outside [{lo},{hi}].")
    return warnings
