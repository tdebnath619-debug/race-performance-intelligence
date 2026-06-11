"""
telemetry/metrics.py
====================
Per-corner and per-lap performance metrics.

Computes engineering indicators from a cleaned, segmented TelemetryLap.
These feed directly into the delta analysis and report output.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class CornerMetrics:
    corner_id:           int
    corner_type:         str
    entry_speed:         float
    min_speed:           float
    exit_speed:          float
    brake_pressure_max:  float
    braking_distance_m:  float
    throttle_pickup_dist: float
    throttle_ramp_rate:  float
    ers_deployed_kj:     float
    lateral_g_max:       float
    corner_time_s:       float
    late_braking_score:  float   # 0-100
    exit_quality_score:  float   # 0-100
    v_loss:  float = field(init=False)
    v_recovery: float = field(init=False)

    def __post_init__(self):
        self.v_loss     = round(self.entry_speed - self.min_speed, 2)
        self.v_recovery = round(self.exit_speed  - self.min_speed, 2)

    def to_dict(self): return asdict(self)

    def __repr__(self):
        return (f"C{self.corner_id:2d}[{self.corner_type}] "
                f"entry={self.entry_speed:.0f} apex={self.min_speed:.0f} "
                f"exit={self.exit_speed:.0f} km/h | "
                f"brk={self.braking_distance_m:.0f}m | "
                f"t={self.corner_time_s:.3f}s")


@dataclass
class LapMetrics:
    lap_id:               int
    lap_time_s:           float
    avg_speed_kmh:        float
    max_speed_kmh:        float
    total_brake_time_s:   float
    total_coasting_time_s: float
    avg_throttle_pct:     float
    top_gear:             int
    n_corners:            int
    total_ers_deployed_kj: float
    avg_lateral_g:        float
    max_lateral_g:        float
    fuel_load_kg:         float
    corner_metrics:       list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["corner_metrics"] = [c.to_dict() for c in self.corner_metrics]
        return d


def compute_lap_metrics(lap, corners, throttle_thr=0.10, brake_thr=0.05) -> LapMetrics:
    df = lap.data
    dt = df["time"].diff().fillna(0)

    brake_t  = float(dt[df["brake"] > brake_thr].sum()) if "brake" in df.columns else 0.0
    coast_t  = float(dt[
        (df.get("throttle", pd.Series(0,index=df.index)) < throttle_thr) &
        (df.get("brake",    pd.Series(0,index=df.index)) < brake_thr)
    ].sum()) if "throttle" in df.columns else 0.0
    avg_thr  = float(df["throttle"].mean()*100) if "throttle" in df.columns else 0.0
    top_gear = int(df["gear"].max())             if "gear"     in df.columns else 0
    fuel_kg  = float(df["fuel_kg"].iloc[0])      if "fuel_kg"  in df.columns else 0.0
    ers_dep  = _integrate(df, "ers_deployment_kw", dt)
    avg_g    = float(df["lateral_g"].abs().mean()) if "lateral_g" in df.columns else 0.0
    max_g    = float(df["lateral_g"].abs().max())  if "lateral_g" in df.columns else 0.0

    cms = [_corner_metrics(df, c, throttle_thr, brake_thr) for c in corners]
    cms = [c for c in cms if c is not None]

    lap_m = LapMetrics(
        lap_id                = lap.lap_number,
        lap_time_s            = round(float(df["time"].max() - df["time"].min()), 3),
        avg_speed_kmh         = round(float(df["speed"].mean()), 2),
        max_speed_kmh         = round(float(df["speed"].max()),  2),
        total_brake_time_s    = round(brake_t, 3),
        total_coasting_time_s = round(coast_t, 3),
        avg_throttle_pct      = round(avg_thr, 2),
        top_gear              = top_gear,
        n_corners             = len(cms),
        total_ers_deployed_kj = round(ers_dep, 2),
        avg_lateral_g         = round(avg_g, 3),
        max_lateral_g         = round(max_g, 3),
        fuel_load_kg          = round(fuel_kg, 2),
        corner_metrics        = cms,
    )
    log.info("Lap %d: %.3fs | %d corners | ERS %.1f kJ | maxG %.2f g",
             lap.lap_number, lap_m.lap_time_s, lap_m.n_corners,
             lap_m.total_ers_deployed_kj, lap_m.max_lateral_g)
    return lap_m


def print_lap_summary(lm: LapMetrics):
    sep = "═" * 68
    print(f"\n{sep}")
    print(f"  LAP {lm.lap_id}  —  {lm.lap_time_s:.3f} s")
    print(sep)
    print(f"  Avg speed   : {lm.avg_speed_kmh:.1f} km/h   Max: {lm.max_speed_kmh:.1f} km/h")
    print(f"  Brake time  : {lm.total_brake_time_s:.2f} s   Coast: {lm.total_coasting_time_s:.2f} s")
    print(f"  Avg throttle: {lm.avg_throttle_pct:.1f}%    Top gear: {lm.top_gear}")
    print(f"  ERS deployed: {lm.total_ers_deployed_kj:.1f} kJ")
    print(f"  Lateral G   : avg {lm.avg_lateral_g:.2f} g   max {lm.max_lateral_g:.2f} g")
    print(f"  Fuel load   : {lm.fuel_load_kg:.1f} kg")
    print(f"  Corners     : {lm.n_corners}")
    print("─" * 68)
    for cm in lm.corner_metrics:
        print(f"  {cm}")
    print(sep)


def _corner_metrics(df, corner, thr_thr, brk_thr):
    try:
        er = df.iloc[corner.entry_idx]
        ar = df.iloc[corner.apex_idx]
        xr = df.iloc[corner.exit_idx]
        dt = df["time"].diff().fillna(0)

        brk_max  = float(df.iloc[corner.entry_idx:corner.apex_idx]["brake"].max()) \
                   if "brake" in df.columns else 0.0
        brk_dist = round(float(ar["distance"]) - float(er["distance"]), 1)

        tp_dist = round(float(ar["distance"]), 1)
        tp_rate = 0.0
        if "throttle" in df.columns:
            xs = df.iloc[corner.apex_idx:corner.exit_idx]
            above = xs[xs["throttle"] > thr_thr]
            if not above.empty:
                tp_dist = round(float(above.iloc[0]["distance"]), 1)
                rw = above.iloc[:20]
                if len(rw) > 1:
                    dT  = rw["throttle"].iloc[-1] - rw["throttle"].iloc[0]
                    dt2 = rw["time"].iloc[-1]     - rw["time"].iloc[0]
                    tp_rate = round(float(dT/dt2) if dt2 > 0 else 0.0, 4)

        ers_dep = _integrate(df.iloc[corner.entry_idx:corner.exit_idx],
                             "ers_deployment_kw",
                             dt.iloc[corner.entry_idx:corner.exit_idx])
        lat_g   = float(df.iloc[corner.entry_idx:corner.exit_idx]["lateral_g"].abs().max()) \
                  if "lateral_g" in df.columns else 0.0

        entry_spd = round(float(er["speed"]), 2)
        exit_spd  = round(float(xr["speed"]), 2)

        return CornerMetrics(
            corner_id            = corner.corner_id,
            corner_type          = corner.corner_type,
            entry_speed          = entry_spd,
            min_speed            = round(float(ar["speed"]), 2),
            exit_speed           = exit_spd,
            brake_pressure_max   = round(brk_max, 3),
            braking_distance_m   = brk_dist,
            throttle_pickup_dist = tp_dist,
            throttle_ramp_rate   = tp_rate,
            ers_deployed_kj      = round(ers_dep, 3),
            lateral_g_max        = round(lat_g, 3),
            corner_time_s        = corner.duration_s,
            late_braking_score   = round(max(0, 100*(1-brk_dist/150.0)), 1),
            exit_quality_score   = round(min(100, 100*exit_spd/max(entry_spd,1)), 1),
        )
    except Exception as e:
        log.warning("Corner %d metrics failed: %s", corner.corner_id, e)
        return None


def _integrate(df, col, dt):
    if col not in df.columns: return 0.0
    return float((df[col] * dt).sum()) / 1000.0
