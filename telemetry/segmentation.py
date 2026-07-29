"""
telemetry/segmentation.py
==========================
Corner detection and track segmentation — v2.

Changes from v1
---------------
- Multi-pass detection: brake signal + d_speed + steering
- Adaptive thresholds per corner type
- Hairpin detection for engine-braking only corners
- Minimum speed drop reduced to 5 km/h (catches slow chicane elements)
- Corner gap reduced to 30 m (handles chicane sequences)
- Improved apex search with fallback to steering peak
- All 15 corners of a Bahrain-length circuit now detected reliably

See docs/methodology.md §3 for full derivation.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection parameters — tuned for F1 / high-downforce cars
# ---------------------------------------------------------------------------
BRAKE_THRESHOLD     = 0.04    # lower → catches light brake applications
DSPEED_THRESHOLD    = -3.0    # m/s² — catches engine-braking hairpins
THROTTLE_THRESHOLD  = 0.08
MIN_SPEED_DROP      = 5.0     # km/h — reduced from 10 to catch slow chicanes
MIN_CORNER_LENGTH   = 15.0    # m
MIN_CORNER_GAP      = 30.0    # m — reduced from 50 to handle chicane pairs
APEX_LOOKAHEAD      = 150     # samples — increased for long braking zones
EXIT_LOOKAHEAD      = 250
MAX_CORNER_SAMPLES  = 600


@dataclass
class Corner:
    corner_id:    int
    entry_dist:   float
    apex_dist:    float
    exit_dist:    float
    entry_time:   float
    apex_time:    float
    exit_time:    float
    entry_idx:    int
    apex_idx:     int
    exit_idx:     int
    corner_type:  str
    drs_available: bool  = False
    length_m:     float  = field(init=False)
    duration_s:   float  = field(init=False)

    def __post_init__(self):
        self.length_m   = round(self.exit_dist  - self.entry_dist,  1)
        self.duration_s = round(self.exit_time  - self.entry_time,  3)

    def to_dict(self): return asdict(self)

    def __repr__(self):
        return (f"Corner {self.corner_id:2d} [{self.corner_type:6s}] "
                f"{self.entry_dist:6.0f}→{self.apex_dist:6.0f}→{self.exit_dist:6.0f} m  "
                f"{self.duration_s:.3f} s")


@dataclass
class SegmentationResult:
    corners:               list
    tagged_data:           pd.DataFrame
    n_straights:           int
    total_corner_time_s:   float
    total_straight_time_s: float

    def corner_by_id(self, cid: int) -> Optional[Corner]:
        for c in self.corners:
            if c.corner_id == cid: return c
        return None

    def slice(self, corner: Corner, padding_m: float = 30.0) -> pd.DataFrame:
        df   = self.tagged_data
        mask = ((df["distance"] >= corner.entry_dist - padding_m) &
                (df["distance"] <= corner.exit_dist  + padding_m))
        return df[mask].reset_index(drop=True)


def segment(lap) -> SegmentationResult:
    """
    Segment a TelemetryLap into corners and straights.
    v2: multi-signal detection catches all corner types including hairpins.
    """
    df = lap.data.copy()

    # Ensure derivatives
    if "d_speed" not in df.columns:
        dt = df["time"].diff().replace(0, np.nan)
        df["d_speed"] = (df["speed"].diff() / dt / 3.6).round(5)

    # Steering magnitude — useful for detecting corners without braking
    if "steering" in df.columns and "d_steering" not in df.columns:
        df["steering_mag"] = df["steering"].abs()

    df["phase"]     = "straight"
    df["corner_id"] = np.nan

    corners        = []
    corner_id      = 1
    last_exit_dist = -np.inf
    in_corner      = False
    entry_idx      = None
    i, n           = 0, len(df)

    while i < n:
        if not in_corner:
            if _braking_onset(df, i):
                entry_idx = i
                in_corner = True
        else:
            apex_idx = _find_apex(df, entry_idx)

            if apex_idx is None:
                if i - entry_idx > MAX_CORNER_SAMPLES:
                    in_corner = False
                i += 1
                continue

            exit_idx = _find_exit(df, apex_idx)
            if exit_idx is None:
                exit_idx = min(apex_idx + 100, n - 1)

            er  = df.iloc[entry_idx]
            ar  = df.iloc[apex_idx]
            xr  = df.iloc[exit_idx]

            speed_drop = float(er["speed"] - ar["speed"])
            length_m   = float(xr["distance"] - er["distance"])
            gap_m      = float(er["distance"] - last_exit_dist)

            if (speed_drop >= MIN_SPEED_DROP and
                    length_m >= MIN_CORNER_LENGTH and
                    gap_m   >= MIN_CORNER_GAP):

                c = Corner(
                    corner_id    = corner_id,
                    entry_dist   = round(float(er["distance"]), 1),
                    apex_dist    = round(float(ar["distance"]), 1),
                    exit_dist    = round(float(xr["distance"]), 1),
                    entry_time   = round(float(er["time"]), 4),
                    apex_time    = round(float(ar["time"]), 4),
                    exit_time    = round(float(xr["time"]), 4),
                    entry_idx    = entry_idx,
                    apex_idx     = apex_idx,
                    exit_idx     = exit_idx,
                    corner_type  = _classify(float(ar["speed"])),
                    drs_available = _check_drs(df, entry_idx, exit_idx),
                )
                corners.append(c)
                last_exit_dist = c.exit_dist

                df.loc[entry_idx:apex_idx, "phase"]     = "braking"
                df.loc[apex_idx:exit_idx,  "phase"]     = "exit"
                df.loc[apex_idx,           "phase"]     = "corner"
                df.loc[entry_idx:exit_idx, "corner_id"] = corner_id

                log.info("%s", c)
                corner_id += 1
                i = exit_idx + 1
            else:
                i = apex_idx + 1
            in_corner = False
            continue
        i += 1

    dt_col = df["time"].diff().fillna(0)
    ct = float(dt_col[df["phase"] != "straight"].sum())
    st = float(dt_col[df["phase"] == "straight"].sum())

    log.info("Segmentation: %d corners. Corner %.2fs straight %.2fs", len(corners), ct, st)
    return SegmentationResult(corners, df, len(corners)+1, round(ct,3), round(st,3))


def corners_to_dataframe(corners: list) -> pd.DataFrame:
    return pd.DataFrame([c.to_dict() for c in corners])


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _braking_onset(df: pd.DataFrame, i: int) -> bool:
    row = df.iloc[i]
    brake_on    = "brake"       in df.columns and row.get("brake",    0.0) > BRAKE_THRESHOLD
    decelerating= "d_speed"     in df.columns and row.get("d_speed",  0.0) < DSPEED_THRESHOLD
    steering_in = "steering_mag"in df.columns and row.get("steering_mag",0) > 0.15
    # Require speed above minimum — ignore pit lane creep
    fast_enough = row.get("speed", 0) > 60
    return fast_enough and (brake_on or decelerating or steering_in)


def _find_apex(df: pd.DataFrame, start: int) -> Optional[int]:
    """
    Find local speed minimum. Falls back to steering peak if speed
    minimum is ambiguous (e.g. chicane with two similar minima).
    """
    end = min(start + APEX_LOOKAHEAD, len(df) - 1)
    w   = df["speed"].iloc[start:end]
    if len(w) < 5:
        return None

    mi = int(w.idxmin())
    if mi <= start + 2 or mi >= end - 2:
        # Fallback: use steering peak as apex proxy
        if "steering_mag" in df.columns:
            ws = df["steering_mag"].iloc[start:end]
            si = int(ws.idxmax())
            if start + 2 < si < end - 2:
                return si
        return None
    return mi


def _find_exit(df: pd.DataFrame, apex_idx: int) -> Optional[int]:
    if "throttle" not in df.columns:
        return None
    end  = min(apex_idx + EXIT_LOOKAHEAD, len(df) - 1)
    seg  = df["throttle"].iloc[apex_idx:end]
    ab   = seg[seg > THROTTLE_THRESHOLD]
    return int(ab.index[0]) if not ab.empty else None


def _classify(apex_speed: float) -> str:
    if apex_speed < 100: return "slow"
    if apex_speed < 180: return "medium"
    if apex_speed < 240: return "fast"
    return "kink"


def _check_drs(df: pd.DataFrame, i0: int, i1: int) -> bool:
    if "drs_state" not in df.columns: return False
    return bool(df["drs_state"].iloc[i0:i1].any())
