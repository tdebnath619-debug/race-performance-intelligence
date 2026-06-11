"""
telemetry/cleaner.py
====================
Signal processing pipeline for motorsport telemetry.

Objective
---------
Transform raw telemetry signals into clean, analysis-ready data.
Every processing step is documented with purpose, assumptions, and limitations.

Pipeline order
--------------
1. Clip to physical bounds
2. Interpolate NaN
3. Remove spikes
4. Smooth signals
5. Compute derivatives

See docs/methodology.md §2 for full derivation and parameter choices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Processing parameters
# ---------------------------------------------------------------------------

_CLIP_BOUNDS: dict = {
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
}

# Smoothing window sizes (samples).
# Rationale: window chosen to preserve event timing while reducing noise.
# Speed 5-sample @ 100 Hz = 50 ms — preserves corner events.
# Steering larger — high-frequency content is noise, not driver input.
_SMOOTH_WINDOWS: dict = {
    "speed":             5,
    "throttle":          3,
    "brake":             3,
    "steering":          7,
    "rpm":               5,
    "lateral_g":         5,
    "longitudinal_g":    5,
    "tyre_temp_fl":      11,
    "tyre_temp_fr":      11,
    "tyre_temp_rl":      11,
    "tyre_temp_rr":      11,
    "brake_temp_fl":     9,
    "brake_temp_fr":     9,
    "brake_temp_rl":     9,
    "brake_temp_rr":     9,
    "ers_deployment_kw": 7,
    "ers_harvesting_kw": 7,
}

_SPIKE_WINDOW:    int   = 30
_SPIKE_THRESHOLD: float = 4.0


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------

@dataclass
class ProcessingRecord:
    step:                str
    signals_affected:    list
    n_samples_modified:  int
    notes:               str = ""


@dataclass
class CleaningReport:
    driver: str
    lap:    int
    steps:  list = field(default_factory=list)

    def summary(self) -> str:
        lines = [f"Cleaning report — {self.driver} Lap {self.lap}"]
        for s in self.steps:
            lines.append(
                f"  [{s.step}] {s.signals_affected} "
                f"— {s.n_samples_modified} samples — {s.notes}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean(lap, smooth: bool = True, spike_threshold: float = _SPIKE_THRESHOLD,
          interpolation: str = "linear"):
    """
    Apply the full signal processing pipeline to a TelemetryLap.
    Returns (cleaned_lap, CleaningReport).
    Input lap is not modified.
    """
    from loader import TelemetryLap   # flat import — works in both layouts

    df     = lap.data.copy()
    report = CleaningReport(driver=lap.driver, lap=lap.lap_number)

    df, rec = _clip_bounds(df);           report.steps.append(rec)
    df, rec = _fill_nans(df, interpolation); report.steps.append(rec)
    df, rec = _remove_spikes(df, spike_threshold, interpolation); report.steps.append(rec)
    if smooth:
        df, rec = _smooth(df);            report.steps.append(rec)
    df = _derivatives(df)

    cleaned = TelemetryLap(
        data             = df,
        lap_number       = lap.lap_number,
        driver           = lap.driver,
        session          = lap.session,
        source_file      = lap.source_file,
        sample_rate_hz   = lap.sample_rate_hz,
        signals_present  = lap.signals_present,
        warnings         = lap.warnings.copy(),
    )
    log.info(report.summary())
    return cleaned, report


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def _clip_bounds(df):
    """
    Purpose    : Remove sensor saturation artefacts and ADC overflow.
    Method     : Hard clip to documented physical bounds per signal.
    Assumption : Values outside bounds are sensor errors.
    Limitation : Cannot distinguish saturation from genuine extremes at boundary.
    """
    affected, total = [], 0
    for col, (lo, hi) in _CLIP_BOUNDS.items():
        if col not in df.columns: continue
        mask = (df[col] < lo) | (df[col] > hi)
        n = int(mask.sum())
        if n:
            df[col] = df[col].clip(lo, hi)
            affected.append(col); total += n
    return df, ProcessingRecord("clip_bounds", affected, total,
                                "Hard clip to physical bounds.")


def _fill_nans(df, method):
    """
    Purpose    : Fill gaps from logging dropouts.
    Method     : Pandas interpolate + forward/backward fill at edges.
    Assumption : Gaps are short relative to signal dynamics (< 0.5 s).
    Limitation : Long gaps (> 1 s) will be poorly reconstructed.
    """
    num    = df.select_dtypes(include=np.number).columns.tolist()
    before = int(df[num].isna().sum().sum())
    if before > 0:
        df[num] = (df[num].interpolate(method=method, limit_direction="both")
                          .ffill().bfill())
    after = int(df[num].isna().sum().sum())
    return df, ProcessingRecord("fill_nans", num if before > 0 else [],
                                before - after,
                                f"method='{method}'. {before} NaN before, {after} after.")


def _remove_spikes(df, threshold, method):
    """
    Purpose    : Reject implausible instantaneous values.
    Method     : Rolling z-score — samples exceeding threshold replaced with NaN.
    Assumption : Spikes are isolated single samples.
    Limitation : Sustained anomalies (multi-sample) will not be caught.
    """
    targets = [c for c in ["speed","throttle","brake","rpm",
                            "lateral_g","longitudinal_g","ers_deployment_kw"]
               if c in df.columns]
    total = 0
    for col in targets:
        s    = df[col].copy()
        rmn  = s.rolling(_SPIKE_WINDOW, center=True, min_periods=1).mean()
        rstd = s.rolling(_SPIKE_WINDOW, center=True, min_periods=1).std().replace(0, np.nan)
        mask = (s - rmn).abs() / rstd > threshold
        n    = int(mask.sum())
        if n:
            df.loc[mask, col] = np.nan; total += n
    if total > 0:
        num = df.select_dtypes(include=np.number).columns
        df[num] = df[num].interpolate(method=method, limit_direction="both").ffill().bfill()
    return df, ProcessingRecord("remove_spikes", targets, total,
                                f"window={_SPIKE_WINDOW}, threshold={threshold}.")


def _smooth(df):
    """
    Purpose    : Reduce sensor noise while preserving event timing.
    Method     : Centred rolling mean (zero phase shift).
    Assumption : Noise is higher frequency than events of interest.
    Limitation : Introduces small timing error at transitions (~half window).
                 Do NOT use smoothed data for precise brake point measurements.
    """
    affected = []
    for col, win in _SMOOTH_WINDOWS.items():
        if col not in df.columns: continue
        df[col] = df[col].rolling(win, center=True, min_periods=1).mean()
        affected.append(f"{col}(w={win})")
    return df, ProcessingRecord("smooth", affected,
                                len(df) * len(affected),
                                "Centred rolling mean.")


def _derivatives(df):
    """
    Compute first-order time derivatives for key channels.
    NOT smoothed — events must remain sharp for delta analysis.
    """
    dt = df["time"].diff().replace(0, np.nan)
    for col in ("speed","throttle","brake","steering"):
        if col not in df.columns: continue
        deriv = df[col].diff() / dt
        if col == "speed":
            deriv = deriv / 3.6   # km/h/s → m/s²
        df[f"d_{col}"] = deriv.round(5)
    return df
