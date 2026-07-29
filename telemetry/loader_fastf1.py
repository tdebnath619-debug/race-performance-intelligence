"""
telemetry/loader_fastf1.py
===========================
FastF1 adapter — loads official F1 timing and telemetry data.

Converts FastF1 channel names to the internal TelemetryLap format,
so all downstream modules (cleaner, segmentation, delta) work unchanged.

Usage
-----
    from loader_fastf1 import load_fastf1

    lap = load_fastf1(2024, "Bahrain", "Q", driver="VER")
    lap2 = load_fastf1(2024, "Bahrain", "Q", driver="LEC")

Requirements
------------
    pip install fastf1

Cache
-----
FastF1 caches session data in data/cache/ to avoid repeated downloads.
First load of a session takes 30-60 seconds. Subsequent loads are instant.

Supported sessions
------------------
    "Q"  — Qualifying
    "R"  — Race
    "FP1", "FP2", "FP3" — Practice

Notes on data quality
---------------------
FastF1 provides 240 Hz telemetry from official FIA timing.
Speed is in km/h, throttle 0-100 (normalised to 0-1 by this adapter).
Distance is measured from the pit exit loop — consistent across laps.
DRS state is integer (0=closed, 10/12/14=open stages).

See docs/methodology.md §5 for FastF1 integration notes.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def load_fastf1(
    year:     int,
    gp:       str,
    session:  str  = "Q",
    driver:   str  = "VER",
    lap_type: str  = "fastest",
    cache_dir: str = "data/cache",
) -> object:
    """
    Load official F1 telemetry via FastF1 and return a TelemetryLap.

    Parameters
    ----------
    year      : Season year (e.g. 2024).
    gp        : Grand Prix name or round number (e.g. "Bahrain" or 1).
    session   : Session type — "Q", "R", "FP1", "FP2", "FP3".
    driver    : Three-letter driver code (e.g. "VER", "LEC", "HAM").
    lap_type  : "fastest" for fastest lap; "all" returns list of all laps.
    cache_dir : Local cache directory. Created if it does not exist.

    Returns
    -------
    TelemetryLap  (same format as load_csv output)

    Raises
    ------
    ImportError   : fastf1 not installed.
    ValueError    : Driver or session not found.
    """
    try:
        import fastf1
    except ImportError:
        raise ImportError(
            "fastf1 is not installed. Run: pip install fastf1"
        )

    # Enable cache
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_path))

    log.info("Loading FastF1: %d %s %s — %s", year, gp, session, driver)

    sess = fastf1.get_session(year, gp, session)
    sess.load(telemetry=True, laps=True, weather=False, messages=False)

    # Get driver laps
    try:
        driver_laps = sess.laps.pick_driver(driver)
    except Exception:
        available = sorted(sess.laps["Driver"].unique().tolist())
        raise ValueError(
            f"Driver '{driver}' not found in session. "
            f"Available: {available}"
        )

    if lap_type == "fastest":
        lap = driver_laps.pick_fastest()
        tel = lap.get_telemetry()
        return _build_telemetry_lap(tel, lap, driver, f"{year}_{gp}_{session}", lap_number=0)
    else:
        results = []
        for i, (_, lap_row) in enumerate(driver_laps.iterrows()):
            try:
                tel = lap_row.get_telemetry()
                tl  = _build_telemetry_lap(tel, lap_row, driver,
                                            f"{year}_{gp}_{session}", lap_number=i)
                results.append(tl)
            except Exception as e:
                log.debug("Lap %d skipped: %s", i, e)
        log.info("Loaded %d laps for %s", len(results), driver)
        return results


def load_fastf1_comparison(
    year:     int,
    gp:       str,
    session:  str = "Q",
    driver_a: str = "VER",
    driver_b: str = "LEC",
    cache_dir: str = "data/cache",
) -> tuple:
    """
    Load fastest laps for two drivers in one call.
    Returns (lap_a, lap_b) as TelemetryLap objects.
    Convenience wrapper for the most common use case.
    """
    lap_a = load_fastf1(year, gp, session, driver_a, cache_dir=cache_dir)
    lap_b = load_fastf1(year, gp, session, driver_b, cache_dir=cache_dir)
    log.info(
        "Loaded: %s %.3fs | %s %.3fs",
        driver_a, lap_a.lap_time_s,
        driver_b, lap_b.lap_time_s,
    )
    return lap_a, lap_b


# ---------------------------------------------------------------------------
# Internal builder
# ---------------------------------------------------------------------------

def _build_telemetry_lap(tel, lap_row, driver, session_name, lap_number):
    """Convert FastF1 telemetry to internal TelemetryLap format."""

    # Import here to avoid circular import
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from loader import TelemetryLap

    df = pd.DataFrame({
        "time":        tel["Time"].dt.total_seconds(),
        "distance":    tel["Distance"].astype(float),
        "speed":       tel["Speed"].astype(float),
        "throttle":    (tel["Throttle"] / 100.0).clip(0, 1),
        "brake":       tel["Brake"].astype(float).clip(0, 1),
        "gear":        tel["nGear"].astype(int),
        "rpm":         tel["RPM"].astype(float),
        "drs_state":   (tel["DRS"] > 9).astype(int),   # DRS open = 10/12/14
    })

    # Add track XY if available (useful for track map visualisation)
    if "X" in tel.columns and "Y" in tel.columns:
        df["track_x"] = tel["X"].astype(float)
        df["track_y"] = tel["Y"].astype(float)

    # Normalise time to start at 0
    df["time"]     = df["time"] - df["time"].iloc[0]
    df["distance"] = df["distance"] - df["distance"].iloc[0]

    # Sort and reset
    df = df.sort_values("time").reset_index(drop=True)

    # Estimate sample rate
    dt  = df["time"].diff().dropna()
    hz  = round(1.0 / float(dt[dt > 0].median()), 1) if not dt.empty else 0.0

    signals = [c for c in df.columns
               if c in ["time","distance","speed","throttle","brake",
                        "gear","rpm","drs_state","track_x","track_y"]]

    # Lap time from FastF1 timing
    try:
        lap_time_s = lap_row["LapTime"].total_seconds()
    except Exception:
        lap_time_s = float(df["time"].max())

    tl = TelemetryLap(
        data             = df,
        lap_number       = lap_number,
        driver           = driver,
        session          = session_name,
        source_file      = f"fastf1:{session_name}:{driver}",
        sample_rate_hz   = hz,
        signals_present  = signals,
        warnings         = [],
    )
    # Attach lap time directly
    tl._fastf1_lap_time_s = round(lap_time_s, 3)

    log.info(
        "FastF1 lap: %s | %.3f s | %.0f m | %.0f Hz | %d signals",
        driver, lap_time_s, df["distance"].max(), hz, len(signals)
    )
    return tl


def session_info(year: int, gp: str, session: str = "Q",
                 cache_dir: str = "data/cache") -> pd.DataFrame:
    """
    Return a summary DataFrame of all drivers in a session.
    Useful for picking comparison targets.
    """
    try:
        import fastf1
        fastf1.Cache.enable_cache(cache_dir)
        sess = fastf1.get_session(year, gp, session)
        sess.load(telemetry=False, laps=True, weather=False, messages=False)
        summary = (
            sess.laps
            .groupby("Driver")
            .apply(lambda g: g.loc[g["LapTime"].idxmin()])
            [["Driver", "LapTime", "Compound", "TyreLife"]]
            .sort_values("LapTime")
            .reset_index(drop=True)
        )
        summary["LapTime_s"] = summary["LapTime"].dt.total_seconds().round(3)
        summary["Gap_s"]     = (summary["LapTime_s"] - summary["LapTime_s"].iloc[0]).round(3)
        return summary[["Driver","LapTime_s","Gap_s","Compound","TyreLife"]]
    except Exception as e:
        log.error("session_info failed: %s", e)
        return pd.DataFrame()
