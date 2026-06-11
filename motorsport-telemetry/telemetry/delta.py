"""
telemetry/delta.py
==================
Lap delta analysis engine.

Objective
---------
Determine where lap time is gained or lost between two laps and explain
the engineering reason for each delta.

Methodology
-----------
1. Align both laps on a common distance grid (not time grid).
   Rationale: time-based alignment phase-shifts signals when one driver
   is faster in a section, making brake point comparison meaningless.

2. Cumulative time delta:
   ΔT(d) = ∫₀ᵈ (1/v_B − 1/v_A) dx
   Positive → Lap A faster at distance d.

3. Per-corner metrics: time delta, entry/apex/exit speed, brake point,
   throttle pickup, ERS delta.

4. Engineering narrative: rule-based attribution of delta to primary cause.

See docs/methodology.md §4 for full derivation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

GRID_POINTS       = 5000
MAX_APEX_MISMATCH = 200.0   # metres


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CornerDelta:
    corner_id:              int
    corner_type:            str
    apex_dist_m:            float
    time_delta_s:           float   # positive → A faster
    entry_speed_delta:      float   # km/h  A − B
    min_speed_delta:        float
    exit_speed_delta:       float
    brake_point_delta_m:    float   # positive → A brakes later
    throttle_point_delta_m: float   # positive → A picks up earlier
    ers_delta_kj:           float = 0.0
    lateral_g_delta:        float = 0.0
    primary_cause:          str   = ""
    narrative:              str   = ""

    def __post_init__(self):
        if not self.primary_cause:
            self.primary_cause, self.narrative = _narrative(self)

    def to_dict(self): return asdict(self)

    def __repr__(self):
        return (f"T{self.corner_id} [{self.corner_type}]  "
                f"Δt={self.time_delta_s:+.3f}s  "
                f"ΔvEntry={self.entry_speed_delta:+.1f}  "
                f"ΔvApex={self.min_speed_delta:+.1f}  "
                f"ΔvExit={self.exit_speed_delta:+.1f} km/h")


@dataclass
class ComparisonReport:
    driver_a:       str
    driver_b:       str
    lap_a:          int
    lap_b:          int
    session:        str
    total_delta_s:  float
    corner_deltas:  list = field(default_factory=list)
    key_findings:   list = field(default_factory=list)
    unmatched_a:    list = field(default_factory=list)
    unmatched_b:    list = field(default_factory=list)
    _common_dist:   object = field(default=None, repr=False)
    _cum_delta:     object = field(default=None, repr=False)

    def summary(self) -> str:
        faster = self.driver_a if self.total_delta_s > 0 else self.driver_b
        return (f"{faster} faster by {abs(self.total_delta_s):.3f} s  |  "
                f"{len(self.corner_deltas)} corners  |  {self.session}")

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_common_dist", None); d.pop("_cum_delta", None)
        d["corner_deltas"] = [c.to_dict() for c in self.corner_deltas]
        return d

    def to_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        class _Enc(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)): return int(o)
                if isinstance(o, (np.floating,)): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                if isinstance(o, (np.bool_,)): return bool(o)
                return super().default(o)

        path.write_text(json.dumps(self.to_dict(), indent=2, cls=_Enc))
        log.info("Delta report → %s", path)
        return path

    def print_report(self):
        sep = "═" * 72
        print(f"\n{sep}")
        print(f"  DELTA: {self.driver_a} (A) vs {self.driver_b} (B)")
        print(f"  {self.summary()}")
        print(f"{sep}")

        if self.key_findings:
            print("\n  KEY FINDINGS")
            print("  " + "─" * 60)
            for f in self.key_findings:
                print(f"  • {f}")

        print("\n  CORNER-BY-CORNER")
        print("  " + "─" * 60)
        print(f"  {'T':<4} {'Type':<8} {'Δt(s)':>8} {'ΔvEntry':>9} "
              f"{'ΔvApex':>9} {'ΔvExit':>9}")
        print("  " + "─" * 60)
        for cd in self.corner_deltas:
            print(f"  T{cd.corner_id:<3} {cd.corner_type:<8} "
                  f"{cd.time_delta_s:>+8.3f} "
                  f"{cd.entry_speed_delta:>+9.1f} "
                  f"{cd.min_speed_delta:>+9.1f} "
                  f"{cd.exit_speed_delta:>+9.1f}")
            print(f"          → {cd.narrative}")

        if self.unmatched_a:
            print(f"\n  Unmatched in {self.driver_a}: corners {self.unmatched_a}")
        if self.unmatched_b:
            print(f"\n  Unmatched in {self.driver_b}: corners {self.unmatched_b}")
        print(sep + "\n")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compare(lap_a, lap_b, seg_a, seg_b) -> ComparisonReport:
    """
    Full delta analysis between two TelemetryLap / SegmentationResult pairs.
    Returns ComparisonReport with per-corner deltas and cumulative trace.
    """
    df_a, df_b = lap_a.data, lap_b.data

    common_dist, _, _ = _align(df_a, df_b)
    cum_delta         = _cum_delta(df_a, df_b, common_dist)
    total             = round(float(cum_delta[-1]), 4)

    matched, unmatched_a, unmatched_b = _match_corners(seg_a.corners, seg_b.corners)

    corner_deltas = []
    for ca, cb in matched:
        cd = _corner_delta(df_a, df_b, ca, cb, common_dist, cum_delta)
        corner_deltas.append(cd)
        log.info("%s  →  %s", cd, cd.narrative)

    findings = _key_findings(corner_deltas, total, lap_a.driver, lap_b.driver)

    report = ComparisonReport(
        driver_a      = lap_a.driver,
        driver_b      = lap_b.driver,
        lap_a         = lap_a.lap_number,
        lap_b         = lap_b.lap_number,
        session       = lap_a.session,
        total_delta_s = total,
        corner_deltas = corner_deltas,
        key_findings  = findings,
        unmatched_a   = [c.corner_id for c in unmatched_a],
        unmatched_b   = [c.corner_id for c in unmatched_b],
        _common_dist  = common_dist,
        _cum_delta    = cum_delta,
    )
    log.info("Delta complete: %s", report.summary())
    return report


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _align(df_a, df_b, n=GRID_POINTS):
    d_min = max(df_a["distance"].min(), df_b["distance"].min())
    d_max = min(df_a["distance"].max(), df_b["distance"].max())
    if d_max <= d_min:
        raise ValueError("Laps have no overlapping distance range.")
    cd = np.linspace(d_min, d_max, n)
    def _interp(df):
        s = {}
        for col in ("speed","throttle","brake","time","ers_deployment_kw","lateral_g"):
            if col in df.columns:
                s[col] = np.interp(cd, df["distance"].values, df[col].values)
        return s
    return cd, _interp(df_a), _interp(df_b)


def _cum_delta(df_a, df_b, common_dist):
    va = np.interp(common_dist, df_a["distance"].values, df_a["speed"].values) / 3.6
    vb = np.interp(common_dist, df_b["distance"].values, df_b["speed"].values) / 3.6
    va = np.where(va > 0, va, np.nan)
    vb = np.where(vb > 0, vb, np.nan)
    dd = np.diff(common_dist, prepend=common_dist[0])
    return np.nancumsum(np.where(~np.isnan(vb), dd/vb, 0) -
                        np.where(~np.isnan(va), dd/va, 0))


def _match_corners(corners_a, corners_b):
    matched, used, unmatched_a = [], set(), []
    for ca in corners_a:
        best, bd = None, np.inf
        for cb in corners_b:
            if id(cb) in used: continue
            d = abs(ca.apex_dist - cb.apex_dist)
            if d < bd and d < MAX_APEX_MISMATCH:
                bd, best = d, cb
        if best:
            matched.append((ca, best)); used.add(id(best))
        else:
            unmatched_a.append(ca)
    unmatched_b = [c for c in corners_b if id(c) not in used]
    return matched, unmatched_a, unmatched_b


def _spd(df, idx):
    idx = max(0, min(idx, len(df)-1))
    return round(float(df.iloc[idx]["speed"]), 2)


def _ers_kj(df, i0, i1):
    if "ers_deployment_kw" not in df.columns: return 0.0
    seg = df.iloc[i0:i1]
    return float((seg["ers_deployment_kw"] * seg["time"].diff().fillna(0)).sum()) / 1000.0


def _latg(df, i0, i1):
    if "lateral_g" not in df.columns: return 0.0
    return float(df.iloc[i0:i1]["lateral_g"].abs().max())


def _corner_delta(df_a, df_b, ca, cb, common_dist, cum_delta):
    exit_d = (ca.exit_dist + cb.exit_dist) / 2
    dt_val = float(np.interp(exit_d, common_dist, cum_delta))
    return CornerDelta(
        corner_id              = ca.corner_id,
        corner_type            = ca.corner_type,
        apex_dist_m            = round((ca.apex_dist + cb.apex_dist) / 2, 1),
        time_delta_s           = round(dt_val, 4),
        entry_speed_delta      = round(_spd(df_a, ca.entry_idx) - _spd(df_b, cb.entry_idx), 2),
        min_speed_delta        = round(_spd(df_a, ca.apex_idx)  - _spd(df_b, cb.apex_idx),  2),
        exit_speed_delta       = round(_spd(df_a, ca.exit_idx)  - _spd(df_b, cb.exit_idx),  2),
        brake_point_delta_m    = round(ca.entry_dist - cb.entry_dist, 1),
        throttle_point_delta_m = round(ca.apex_dist  - cb.apex_dist,  1),
        ers_delta_kj           = round(_ers_kj(df_a,ca.entry_idx,ca.exit_idx) -
                                       _ers_kj(df_b,cb.entry_idx,cb.exit_idx), 3),
        lateral_g_delta        = round(_latg(df_a,ca.entry_idx,ca.exit_idx) -
                                       _latg(df_b,cb.entry_idx,cb.exit_idx), 3),
    )


def _narrative(cd):
    faster = "A" if cd.time_delta_s > 0 else "B"
    if abs(cd.time_delta_s) < 0.005:
        return "none", "No significant delta — laps equivalent through this corner."
    if abs(cd.entry_speed_delta) >= 5.0:
        high = "A" if cd.entry_speed_delta > 0 else "B"
        return "braking_zone", (
            f"Driver {high} carries {abs(cd.entry_speed_delta):.1f} km/h more entry speed "
            f"(brakes {abs(cd.brake_point_delta_m):.0f} m later). "
            f"Driver {faster} gains {abs(cd.time_delta_s):.3f} s.")
    if abs(cd.min_speed_delta) >= 3.0:
        high = "A" if cd.min_speed_delta > 0 else "B"
        return "corner_speed", (
            f"Driver {high} achieves {abs(cd.min_speed_delta):.1f} km/h higher minimum speed. "
            f"Likely: mechanical grip, aero balance, or line. "
            f"Driver {faster} gains {abs(cd.time_delta_s):.3f} s.")
    if abs(cd.throttle_point_delta_m) >= 10.0:
        early = "A" if cd.throttle_point_delta_m > 0 else "B"
        return "throttle_application", (
            f"Driver {early} picks up throttle {abs(cd.throttle_point_delta_m):.0f} m earlier. "
            f"Suggests higher traction confidence or better car balance on exit. "
            f"Driver {faster} gains {abs(cd.time_delta_s):.3f} s.")
    if abs(cd.exit_speed_delta) >= 5.0:
        high = "A" if cd.exit_speed_delta > 0 else "B"
        return "corner_exit", (
            f"Driver {high} exits {abs(cd.exit_speed_delta):.1f} km/h faster. "
            f"Check ERS deployment and rear downforce. "
            f"Driver {faster} gains {abs(cd.time_delta_s):.3f} s.")
    return "unclassified", (
        f"Driver {faster} gains {abs(cd.time_delta_s):.3f} s. "
        f"No single dominant cause — review full trace.")


def _key_findings(deltas, total, driver_a, driver_b):
    if not deltas: return []
    findings = []
    faster = driver_a if total > 0 else driver_b
    worst  = max(deltas, key=lambda d: abs(d.time_delta_s))
    findings.append(
        f"Largest delta: T{worst.corner_id} ({worst.corner_type}) — "
        f"{abs(worst.time_delta_s):.3f} s — {worst.primary_cause.replace('_',' ')}.")
    late = [d for d in deltas if d.brake_point_delta_m < -10]
    if late:
        findings.append(
            f"{driver_a} brakes later at {len(late)} corner(s): {[d.corner_id for d in late]}.")
    early_t = [d for d in deltas if d.throttle_point_delta_m > 10]
    if early_t:
        findings.append(
            f"{driver_a} picks up throttle earlier at {len(early_t)} corner(s): "
            f"{[d.corner_id for d in early_t]}.")
    apex_low = [d for d in deltas if d.min_speed_delta < -3]
    if apex_low:
        findings.append(
            f"{driver_a} has lower apex speed at {len(apex_low)} corner(s) — "
            f"check mechanical balance or aero setup.")
    return findings
