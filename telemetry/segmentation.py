"""
F1 Performance Intelligence System
telemetry/segmentation.py

F1-specific track segmentation:
- Corner detection (braking / apex / exit)
- Mini-sector generation (3 per sector × 3 sectors = 9 mini-sectors)
- DRS zone detection
- Straight identification
- Traction zone tagging (high-throttle, low-speed exit)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class Corner:
    corner_id:       int
    entry_distance:  float
    apex_distance:   float
    exit_distance:   float
    entry_time:      float
    apex_time:       float
    exit_time:       float
    entry_idx:       int
    apex_idx:        int
    exit_idx:        int
    drs_available:   bool  = False
    corner_type:     str   = "medium"   # slow / medium / fast / chicane

    corner_length_m: float = field(init=False)
    corner_time_s:   float = field(init=False)

    def __post_init__(self):
        self.corner_length_m = round(self.exit_distance - self.entry_distance, 1)
        self.corner_time_s   = round(self.exit_time - self.entry_time, 3)

    def to_dict(self):
        return asdict(self)


@dataclass
class MiniSector:
    mini_sector_id:  int
    start_distance:  float
    end_distance:    float
    avg_speed:       float
    time_s:          float
    contains_corner: bool


@dataclass
class DRSZone:
    zone_id:         int
    detection_dist:  float
    activation_dist: float
    end_dist:        float


def segment_corners(
    df: pd.DataFrame,
    brake_threshold:     float = 0.05,
    speed_drop_kmh:      float = 10.0,
    min_corner_length_m: float = 20.0,
    min_gap_m:           float = 50.0,
    throttle_threshold:  float = 0.10,
) -> tuple[list[Corner], pd.DataFrame]:

    df = df.copy()
    df["phase"]     = "straight"
    df["corner_id"] = np.nan

    if "d_speed" not in df.columns:
        dt = df["time"].diff().replace(0, np.nan)
        df["d_speed"] = df["speed"].diff() / dt

    corners: list[Corner] = []
    corner_id = 1
    last_exit_dist = -np.inf
    in_corner  = False
    entry_idx  = None
    i = 0
    n = len(df)

    while i < n:
        row = df.iloc[i]
        if not in_corner:
            if _braking_onset(df, i, brake_threshold):
                entry_idx = i
                in_corner = True
        else:
            apex_idx = _find_apex(df, entry_idx, i, lookahead=100)
            if apex_idx is None:
                if i - entry_idx > 400:
                    in_corner = False
                i += 1
                continue

            exit_idx = _find_exit(df, apex_idx, throttle_threshold, lookahead=200)
            if exit_idx is None:
                exit_idx = min(apex_idx + 100, n - 1)

            er = df.iloc[entry_idx]
            ar = df.iloc[apex_idx]
            xr = df.iloc[exit_idx]

            speed_drop = er["speed"] - ar["speed"]
            length_m   = xr["distance"] - er["distance"]
            gap_m      = er["distance"] - last_exit_dist

            if speed_drop >= speed_drop_kmh and length_m >= min_corner_length_m and gap_m >= min_gap_m:
                ctype = _classify_corner(float(ar["speed"]))
                drs   = bool(df.iloc[entry_idx:exit_idx]["drs_state"].any()) \
                        if "drs_state" in df.columns else False

                c = Corner(
                    corner_id      = corner_id,
                    entry_distance = round(float(er["distance"]), 1),
                    apex_distance  = round(float(ar["distance"]), 1),
                    exit_distance  = round(float(xr["distance"]), 1),
                    entry_time     = round(float(er["time"]), 3),
                    apex_time      = round(float(ar["time"]), 3),
                    exit_time      = round(float(xr["time"]), 3),
                    entry_idx      = entry_idx,
                    apex_idx       = apex_idx,
                    exit_idx       = exit_idx,
                    drs_available  = drs,
                    corner_type    = ctype,
                )
                corners.append(c)
                last_exit_dist = c.exit_distance

                df.loc[entry_idx:apex_idx, "phase"]     = "braking"
                df.loc[apex_idx:exit_idx,  "phase"]     = "exit"
                df.loc[apex_idx,           "phase"]     = "corner"
                df.loc[entry_idx:exit_idx, "corner_id"] = corner_id

                logger.info(
                    f"Corner {corner_id:2d} [{ctype:6s}] | "
                    f"{c.entry_distance:6.0f}→{c.apex_distance:6.0f}→{c.exit_distance:6.0f} m | "
                    f"apex {ar['speed']:.0f} km/h | Δv {speed_drop:.0f} km/h"
                )
                corner_id += 1
                i = exit_idx + 1
            else:
                i = apex_idx + 1
            in_corner = False
            continue
        i += 1

    logger.info(f"Segmentation: {len(corners)} corners detected.")
    return corners, df


def generate_mini_sectors(
    df: pd.DataFrame,
    n_sectors: int = 3,
    per_sector: int = 3,
) -> list[MiniSector]:
    """
    Divide the lap into n_sectors × per_sector mini-sectors by distance.
    Returns list of MiniSector with avg speed and time for each.
    """
    total_ms = n_sectors * per_sector
    d_min, d_max = df["distance"].min(), df["distance"].max()
    edges = np.linspace(d_min, d_max, total_ms + 1)

    mini_sectors = []
    for idx in range(total_ms):
        lo, hi = edges[idx], edges[idx + 1]
        mask   = (df["distance"] >= lo) & (df["distance"] < hi)
        seg    = df[mask]
        if seg.empty:
            continue
        dt       = seg["time"].diff().fillna(0)
        time_s   = float(dt.sum())
        avg_spd  = float(seg["speed"].mean())
        has_corn = "corner_id" in seg.columns and seg["corner_id"].notna().any()

        mini_sectors.append(MiniSector(
            mini_sector_id  = idx + 1,
            start_distance  = round(lo, 1),
            end_distance    = round(hi, 1),
            avg_speed       = round(avg_spd, 2),
            time_s          = round(time_s, 4),
            contains_corner = bool(has_corn),
        ))
    return mini_sectors


def detect_drs_zones(df: pd.DataFrame, min_zone_length_m: float = 400.0) -> list[DRSZone]:
    """Identify DRS zones from the drs_state channel."""
    if "drs_state" not in df.columns:
        return []
    zones = []
    zone_id = 1
    in_zone = False
    zone_start = 0.0

    for _, row in df.iterrows():
        if not in_zone and row["drs_state"] == 1:
            in_zone    = True
            zone_start = row["distance"]
        elif in_zone and row["drs_state"] == 0:
            zone_len = row["distance"] - zone_start
            if zone_len >= min_zone_length_m:
                zones.append(DRSZone(
                    zone_id         = zone_id,
                    detection_dist  = round(zone_start - 100, 0),
                    activation_dist = round(zone_start, 0),
                    end_dist        = round(row["distance"], 0),
                ))
                zone_id += 1
            in_zone = False

    logger.info(f"DRS zones detected: {len(zones)}")
    return zones


def corners_to_dataframe(corners: list[Corner]) -> pd.DataFrame:
    return pd.DataFrame([c.to_dict() for c in corners])


def extract_corner_slice(df: pd.DataFrame, corner: Corner,
                          padding_m: float = 30.0) -> pd.DataFrame:
    mask = (
        (df["distance"] >= corner.entry_distance - padding_m) &
        (df["distance"] <= corner.exit_distance  + padding_m)
    )
    return df[mask].reset_index(drop=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _braking_onset(df, i, threshold):
    row = df.iloc[i]
    return (("brake" in df.columns and row.get("brake", 0) > threshold)
            or row.get("d_speed", 0) < -5.0)


def _find_apex(df, start, current, lookahead):
    end = min(start + lookahead, len(df) - 1)
    w   = df["speed"].iloc[start:end]
    if len(w) < 5:
        return None
    mi = int(w.idxmin())
    if mi <= start + 2 or mi >= end - 2:
        return None
    return mi


def _find_exit(df, apex_idx, throttle_threshold, lookahead):
    if "throttle" not in df.columns:
        return None
    end  = min(apex_idx + lookahead, len(df) - 1)
    w    = df["throttle"].iloc[apex_idx:end]
    ab   = w[w > throttle_threshold]
    return int(ab.index[0]) if not ab.empty else None


def _classify_corner(apex_speed_kmh: float) -> str:
    if apex_speed_kmh < 100:
        return "slow"
    elif apex_speed_kmh < 180:
        return "medium"
    elif apex_speed_kmh < 240:
        return "fast"
    else:
        return "kink"
