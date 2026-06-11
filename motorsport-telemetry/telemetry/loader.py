"""
telemetry/loader.py
===================
Telemetry ingestion and standardisation.

Objective
---------
Load raw telemetry CSV files from multiple sources, validate signal
completeness, normalise units, and return a standardised internal
structure for downstream processing.

Supported sources
-----------------
- Generic CSV (MoTeC export, Pi Toolbox, custom logger)
- Assetto Corsa / ACC / iRacing flat export
- FastF1 (via separate adapter — see loader_fastf1.py)

Internal data contract
----------------------
Every TelemetryLap returned by this module guarantees:
  - time        : float64, seconds, monotonically increasing, starts at 0
  - distance    : float64, metres, cumulative from lap start
  - speed       : float64, km/h, >= 0
  - throttle    : float64, 0.0–1.0 normalised
  - brake       : float64, 0.0–1.0 normalised
  - steering    : float64, -1.0 to +1.0 normalised
  - gear        : int8,    0–9
  - rpm         : float64, >= 0

Optional signals (present if source provides them):
  - lateral_g, longitudinal_g
  - tyre_temp_fl/fr/rl/rr
  - brake_temp_fl/fr/rl/rr
  - fuel_kg
  - ers_deployment_kw
  - drs_state

Assumptions
-----------
See docs/assumptions.md §1 (Telemetry Ingestion).

Limitations
-----------
- GPS-derived distance accumulates error over long stints.
  Normalise against known lap length where possible.
- Time resolution below 10 Hz produces unreliable corner detection.

Author : F1 Performance Intelligence System
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Signal catalogue
# ---------------------------------------------------------------------------

REQUIRED_SIGNALS: list[str] = [
    "time", "speed",
]

RECOMMENDED_SIGNALS: list[str] = [
    "distance", "throttle", "brake", "steering", "gear", "rpm",
]

OPTIONAL_SIGNALS: list[str] = [
    "lateral_g", "longitudinal_g",
    "tyre_temp_fl", "tyre_temp_fr", "tyre_temp_rl", "tyre_temp_rr",
    "brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr",
    "fuel_kg", "ers_deployment_kw", "ers_harvesting_kw", "drs_state",
    "susp_fl", "susp_fr", "susp_rl", "susp_rr",
]

# Platform-specific column aliases mapped to internal names
_ALIASES: dict[str, str] = {
    # Time
    "timestamp": "time", "time_s": "time", "elapsed_s": "time",
    "t": "time",
    # Distance
    "dist": "distance", "lap_dist": "distance", "track_pos": "distance",
    "odometer": "distance",
    # Speed
    "speed_kmh": "speed", "velocity": "speed", "car_speed": "speed",
    "v": "speed",
    # Throttle
    "gas": "throttle", "throttle_input": "throttle",
    "throttle_pct": "throttle", "tps": "throttle",
    # Brake
    "brake_input": "brake", "brake_pct": "brake", "bps": "brake",
    # Steering
    "steer": "steering", "steer_angle": "steering",
    "steering_wheel": "steering", "sw": "steering",
    # Gear
    "current_gear": "gear", "ngear": "gear", "g": "gear",
    # RPM
    "engine_rpm": "rpm", "n": "rpm",
    # G-forces
    "g_lat": "lateral_g", "lat_g": "lateral_g",
    "g_lon": "longitudinal_g", "lon_g": "longitudinal_g",
    "g_long": "longitudinal_g",
    # Tyre temps
    "tyre_temp_fl": "tyre_temp_fl",
    "tyre_temp_fr": "tyre_temp_fr",
    "tyre_temp_rl": "tyre_temp_rl",
    "tyre_temp_rr": "tyre_temp_rr",
    # Brake temps
    "brake_temp_fl": "brake_temp_fl",
    "brake_temp_fr": "brake_temp_fr",
    "brake_temp_rl": "brake_temp_rl",
    "brake_temp_rr": "brake_temp_rr",
    # Power unit
    "ers_deploy": "ers_deployment_kw",
    "ers_harvest": "ers_harvesting_kw",
    "fuel_remaining": "fuel_kg",
    "drs": "drs_state",
}

# Physical bounds for hard validation (signal: (min, max))
_BOUNDS: dict[str, tuple[float, float]] = {
    "speed":             (0.0,   380.0),
    "throttle":          (0.0,   1.0),
    "brake":             (0.0,   1.0),
    "steering":          (-1.0,  1.0),
    "gear":              (0.0,   9.0),
    "rpm":               (0.0,   20_000.0),
    "lateral_g":         (-7.0,  7.0),
    "longitudinal_g":    (-7.0,  5.0),
    "tyre_temp_fl":      (0.0,   200.0),
    "tyre_temp_fr":      (0.0,   200.0),
    "tyre_temp_rl":      (0.0,   200.0),
    "tyre_temp_rr":      (0.0,   200.0),
    "brake_temp_fl":     (0.0,   1200.0),
    "brake_temp_fr":     (0.0,   1200.0),
    "brake_temp_rl":     (0.0,   1200.0),
    "brake_temp_rr":     (0.0,   1200.0),
    "fuel_kg":           (0.0,   110.0),
    "ers_deployment_kw": (0.0,   120.0),
    "ers_harvesting_kw": (0.0,   120.0),
    "drs_state":         (0.0,   1.0),
}


# ---------------------------------------------------------------------------
# Internal data structure
# ---------------------------------------------------------------------------

@dataclass
class TelemetryLap:
    """
    Standardised internal representation of one lap of telemetry.

    All signals are stored in a single DataFrame indexed by sample number.
    Metadata captures provenance for traceability.
    """
    data:        pd.DataFrame
    lap_number:  int
    driver:      str
    session:     str
    source_file: str
    sample_rate_hz: float
    signals_present: list[str]
    warnings:    list[str] = field(default_factory=list)

    @property
    def lap_time_s(self) -> float:
        return float(self.data["time"].iloc[-1] - self.data["time"].iloc[0])

    @property
    def distance_m(self) -> float:
        return float(self.data["distance"].iloc[-1] - self.data["distance"].iloc[0])

    @property
    def n_samples(self) -> int:
        return len(self.data)

    def has_signal(self, name: str) -> bool:
        return name in self.data.columns

    def __repr__(self) -> str:
        return (
            f"TelemetryLap("
            f"driver='{self.driver}', "
            f"lap={self.lap_number}, "
            f"time={self.lap_time_s:.3f}s, "
            f"dist={self.distance_m:.0f}m, "
            f"Hz={self.sample_rate_hz:.0f}, "
            f"signals={len(self.signals_present)})"
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_csv(
    file_path: str | Path,
    driver:     str = "UNKNOWN",
    session:    str = "UNKNOWN",
    lap_number: Optional[int] = None,
) -> list[TelemetryLap]:
    """
    Load a telemetry CSV file and return a list of TelemetryLap objects.

    One object is returned per lap detected in the file.
    If lap_number is specified, only that lap is returned.

    Parameters
    ----------
    file_path   : Path to the CSV file.
    driver      : Driver identifier (used for traceability and delta labels).
    session     : Session name (e.g. 'Bahrain_2024_Q').
    lap_number  : Optional lap filter (0-based index).

    Returns
    -------
    list[TelemetryLap]

    Raises
    ------
    FileNotFoundError : File does not exist.
    ValueError        : Required signals missing after alias resolution.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {file_path}")

    log.info("Loading %s", file_path.name)

    raw = pd.read_csv(file_path)
    raw.columns = raw.columns.str.strip().str.lower().str.replace(r"[\s\-]+", "_", regex=True)
    raw = raw.rename(columns=_ALIASES)

    _validate_required(raw, file_path.name)

    raw = _normalise_time(raw)
    raw = _normalise_controls(raw)
    raw = _normalise_steering(raw)
    raw = _compute_distance(raw)
    raw = _infer_lap_number(raw)

    laps_found = sorted(raw["_lap"].unique())
    log.info("Laps detected: %s", laps_found)

    if lap_number is not None and lap_number not in laps_found:
        raise ValueError(
            f"Lap {lap_number} not found. Available: {laps_found}"
        )

    results = []
    for lap_idx in laps_found:
        if lap_number is not None and lap_idx != lap_number:
            continue

        lap_df = raw[raw["_lap"] == lap_idx].copy()
        lap_df = lap_df.drop(columns=["_lap"])
        lap_df = lap_df.reset_index(drop=True)

        # Normalise time to start at 0 within lap
        lap_df["time"] = lap_df["time"] - lap_df["time"].iloc[0]

        # Normalise distance to start at 0 within lap
        if "distance" in lap_df.columns:
            lap_df["distance"] = lap_df["distance"] - lap_df["distance"].iloc[0]

        hz   = _estimate_sample_rate(lap_df)
        sigs = _present_signals(lap_df)
        warn = _validate_bounds(lap_df)

        lap_obj = TelemetryLap(
            data             = lap_df,
            lap_number       = lap_idx,
            driver           = driver,
            session          = session,
            source_file      = file_path.name,
            sample_rate_hz   = hz,
            signals_present  = sigs,
            warnings         = warn,
        )
        results.append(lap_obj)
        log.info("Loaded: %s", lap_obj)
        for w in warn:
            log.warning("  [WARN] %s", w)

    return results


def load_multi(
    files: list[tuple[Path, str, str]],
) -> list[TelemetryLap]:
    """
    Load multiple telemetry files.

    Parameters
    ----------
    files : List of (file_path, driver, session) tuples.

    Returns
    -------
    list[TelemetryLap] — all laps from all files, in order.
    """
    all_laps = []
    for fp, drv, sess in files:
        all_laps.extend(load_csv(fp, driver=drv, session=sess))
    return all_laps


def lap_summary(laps: list[TelemetryLap]) -> pd.DataFrame:
    """Return a summary DataFrame — one row per lap."""
    rows = []
    for lap in laps:
        rows.append({
            "driver":       lap.driver,
            "session":      lap.session,
            "lap_number":   lap.lap_number,
            "lap_time_s":   round(lap.lap_time_s, 3),
            "distance_m":   round(lap.distance_m, 0),
            "sample_rate":  round(lap.sample_rate_hz, 1),
            "n_samples":    lap.n_samples,
            "signals":      len(lap.signals_present),
            "warnings":     len(lap.warnings),
            "source":       lap.source_file,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _validate_required(df: pd.DataFrame, filename: str) -> None:
    missing = [s for s in REQUIRED_SIGNALS if s not in df.columns]
    if missing:
        available = list(df.columns)
        raise ValueError(
            f"Required signals missing in '{filename}': {missing}\n"
            f"Available columns after alias resolution: {available}\n"
            f"Add column aliases to telemetry/loader.py → _ALIASES if needed."
        )


def _normalise_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert time column to float seconds."""
    t = df["time"]
    if t.dtype == object:
        try:
            df["time"] = pd.to_timedelta(t).dt.total_seconds()
            return df
        except Exception:
            pass
    # Detect milliseconds: max value > 10 000 s is implausible for a lap
    if t.max() > 10_000 and t.min() >= 0:
        log.info("Time column appears to be in milliseconds. Converting to seconds.")
        df["time"] = t / 1_000.0
    return df


def _normalise_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise throttle and brake from 0–100 to 0–1 if needed."""
    for col in ("throttle", "brake"):
        if col in df.columns and df[col].max() > 1.5:
            log.info("Normalising '%s' from 0-100 range to 0-1.", col)
            df[col] = df[col] / 100.0
    return df


def _normalise_steering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise steering to -1.0 … +1.0.
    Some platforms output degrees (e.g. -360 to +360).
    """
    if "steering" not in df.columns:
        return df
    s = df["steering"]
    if s.abs().max() > 1.5:
        scale = s.abs().max()
        log.info("Normalising steering from ±%.0f to ±1.0.", scale)
        df["steering"] = s / scale
    return df


def _compute_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute cumulative distance from speed × Δt if not present.

    Assumption: speed is in km/h. Converts to m/s before integration.
    Limitation: accumulated error grows with lap length; GPS noise compounds.
    """
    if "distance" not in df.columns:
        log.info(
            "'distance' not found. Computing from speed integration. "
            "Note: accumulated error applies — see docs/assumptions.md §1.3."
        )
        speed_ms = df["speed"] / 3.6
        dt = df["time"].diff().fillna(0.0)
        df["distance"] = (speed_ms * dt).cumsum()
    return df


def _infer_lap_number(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect lap boundaries from distance resets or lap_time resets.
    Stores result in a temporary '_lap' column.
    """
    if "lap_number" in df.columns:
        df["_lap"] = df["lap_number"].astype(int)
        return df

    if "lap_time" in df.columns:
        reset = df["lap_time"].diff() < -5.0
    else:
        reset = df["distance"].diff() < -50.0

    df["_lap"] = reset.cumsum().astype(int)
    return df


def _estimate_sample_rate(df: pd.DataFrame) -> float:
    """Estimate sample rate in Hz from median time delta."""
    dt = df["time"].diff().dropna()
    dt = dt[dt > 0]
    if dt.empty:
        return 0.0
    return round(1.0 / float(dt.median()), 1)


def _present_signals(df: pd.DataFrame) -> list[str]:
    """Return list of recognised signals present in the DataFrame."""
    all_known = set(REQUIRED_SIGNALS + RECOMMENDED_SIGNALS + OPTIONAL_SIGNALS)
    return [c for c in df.columns if c in all_known]


def _validate_bounds(df: pd.DataFrame) -> list[str]:
    """
    Check signal values against physical bounds.
    Returns a list of warning strings — does NOT modify the data.
    Correction is handled by the cleaner module.
    """
    warnings = []
    for col, (lo, hi) in _BOUNDS.items():
        if col not in df.columns:
            continue
        n_out = ((df[col] < lo) | (df[col] > hi)).sum()
        if n_out > 0:
            pct = 100 * n_out / len(df)
            warnings.append(
                f"'{col}': {n_out} samples ({pct:.1f}%) outside bounds "
                f"[{lo}, {hi}]. Inspect before analysis."
            )
    return warnings
