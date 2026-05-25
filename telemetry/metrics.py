"""
F1 Performance Intelligence System
telemetry/metrics.py

F1-specific performance metrics:
- Standard corner metrics (entry/apex/exit speed)
- ERS deployment efficiency per corner
- Downforce proxy (lateral G vs speed²)
- Tyre degradation model (lap-over-lap speed loss)
- Fuel-corrected lap time
- Brake balance estimation
- Straight-line top speed & ERS assist analysis
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

from segmentation import Corner

logger = logging.getLogger(__name__)


@dataclass
class CornerMetrics:
    corner_id:              int
    corner_type:            str
    # Speeds
    entry_speed:            float
    min_speed:              float
    exit_speed:             float
    v_loss:                 float = field(init=False)
    v_recovery:             float = field(init=False)
    # Braking
    brake_start_dist:       float = 0.0
    brake_pressure_max:     float = 0.0
    braking_distance_m:     float = 0.0
    # Throttle
    throttle_pickup_dist:   float = 0.0
    throttle_ramp_rate:     float = 0.0
    # ERS
    ers_deployed_kj:        float = 0.0   # energy deployed in corner (kJ)
    ers_harvested_kj:       float = 0.0
    ers_net_kj:             float = 0.0   # net corner ERS balance
    # Downforce proxy
    lateral_g_max:          float = 0.0   # peak lateral G through corner
    downforce_index:        float = 0.0   # lateral_g / (speed² / ref_speed²)
    # Timing
    corner_time_s:          float = 0.0
    # Scores (0-100)
    late_braking_score:     float = 0.0
    exit_quality_score:     float = 0.0
    ers_efficiency_score:   float = 0.0

    def __post_init__(self):
        self.v_loss     = round(self.entry_speed - self.min_speed, 2)
        self.v_recovery = round(self.exit_speed  - self.min_speed, 2)

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        return (f"C{self.corner_id:2d}[{self.corner_type}] "
                f"entry={self.entry_speed:.0f} apex={self.min_speed:.0f} "
                f"exit={self.exit_speed:.0f} km/h | "
                f"ERS={self.ers_deployed_kj:.1f}kJ | "
                f"LatG={self.lateral_g_max:.2f}g | "
                f"t={self.corner_time_s:.3f}s")


@dataclass
class StraightMetrics:
    straight_id:   int
    start_dist:    float
    end_dist:      float
    top_speed:     float
    avg_ers_kw:    float   # avg ERS deployment on straight
    drs_active:    bool
    time_s:        float

    def to_dict(self):
        return asdict(self)


@dataclass
class TyreDegradation:
    """Lap-over-lap tyre performance model."""
    lap_id:             int
    tyre_age_laps:      int
    compound:           str    # SOFT / MEDIUM / HARD / INTER / WET
    avg_lap_speed:      float
    speed_loss_vs_new:  float  # km/h vs lap 1 on compound
    predicted_deg_rate: float  # km/h per lap degradation

    def to_dict(self):
        return asdict(self)


@dataclass
class LapMetrics:
    lap_id:               int
    lap_time_s:           float
    fuel_corrected_time:  float   # +0.03 s per kg of fuel
    avg_speed_kmh:        float
    max_speed_kmh:        float
    total_brake_time_s:   float
    total_coasting_time_s:float
    avg_throttle_pct:     float
    top_gear:             int
    n_corners:            int
    # F1-specific
    total_ers_deployed_kj:float
    total_ers_harvested_kj:float
    ers_balance_kj:       float
    avg_lateral_g:        float
    max_lateral_g:        float
    fuel_load_kg:         float
    corner_metrics:       list[CornerMetrics] = field(default_factory=list)
    straight_metrics:     list[StraightMetrics] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["corner_metrics"]   = [c.to_dict() for c in self.corner_metrics]
        d["straight_metrics"] = [s.to_dict() for s in self.straight_metrics]
        return d


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_lap_metrics(
    df: pd.DataFrame,
    corners: list[Corner],
    lap_id: int = 0,
    tyre_age: int = 0,
    compound: str = "UNKNOWN",
    throttle_threshold: float = 0.10,
    brake_threshold:    float = 0.05,
) -> LapMetrics:

    dt = df["time"].diff().fillna(0)

    lap_time_s = float(df["time"].max() - df["time"].min())
    fuel_kg    = float(df["fuel_kg"].iloc[0]) if "fuel_kg" in df.columns else 0.0
    # FIA fuel effect: ~0.03 s/kg (car weight difference)
    fuel_corrected = round(lap_time_s - fuel_kg * 0.03, 3)

    total_brake_s   = float(dt[df["brake"]    > brake_threshold].sum()) if "brake"    in df.columns else 0.0
    total_coast_s   = float(dt[
        (df.get("throttle", pd.Series(0, index=df.index)) < throttle_threshold) &
        (df.get("brake",    pd.Series(0, index=df.index)) < brake_threshold)
    ].sum()) if "throttle" in df.columns else 0.0
    avg_thr_pct     = float(df["throttle"].mean() * 100) if "throttle" in df.columns else 0.0
    top_gear        = int(df["gear"].max()) if "gear" in df.columns else 0

    # ERS totals
    ers_dep = _integrate(df, "ers_deployment_kw", dt)
    ers_har = _integrate(df, "ers_harvesting_kw", dt)

    # G stats
    avg_g = float(df["g_lateral"].abs().mean())  if "g_lateral" in df.columns else 0.0
    max_g = float(df["g_lateral"].abs().max())   if "g_lateral" in df.columns else 0.0

    corner_metrics   = [cm for c in corners
                        if (cm := _corner_metrics(df, c, throttle_threshold, brake_threshold)) is not None]
    straight_metrics = _straight_metrics(df, corners)

    lap = LapMetrics(
        lap_id                = lap_id,
        lap_time_s            = round(lap_time_s, 3),
        fuel_corrected_time   = fuel_corrected,
        avg_speed_kmh         = round(float(df["speed"].mean()), 2),
        max_speed_kmh         = round(float(df["speed"].max()),  2),
        total_brake_time_s    = round(total_brake_s, 3),
        total_coasting_time_s = round(total_coast_s, 3),
        avg_throttle_pct      = round(avg_thr_pct, 2),
        top_gear              = top_gear,
        n_corners             = len(corner_metrics),
        total_ers_deployed_kj = round(ers_dep, 2),
        total_ers_harvested_kj= round(ers_har, 2),
        ers_balance_kj        = round(ers_dep - ers_har, 2),
        avg_lateral_g         = round(avg_g, 3),
        max_lateral_g         = round(max_g, 3),
        fuel_load_kg          = round(fuel_kg, 2),
        corner_metrics        = corner_metrics,
        straight_metrics      = straight_metrics,
    )
    logger.info(
        f"Lap {lap_id}: {lap.lap_time_s:.3f}s (fuel-corr {lap.fuel_corrected_time:.3f}s) | "
        f"ERS dep={lap.total_ers_deployed_kj:.1f}kJ har={lap.total_ers_harvested_kj:.1f}kJ | "
        f"maxG={lap.max_lateral_g:.2f}g"
    )
    return lap


# ── Corner metrics ────────────────────────────────────────────────────────────

def _corner_metrics(df, corner, thr_thr, brk_thr) -> Optional[CornerMetrics]:
    try:
        er = df.iloc[corner.entry_idx]
        ar = df.iloc[corner.apex_idx]
        xr = df.iloc[corner.exit_idx]
        dt = df["time"].diff().fillna(0)

        entry_spd = round(float(er["speed"]), 2)
        min_spd   = round(float(ar["speed"]), 2)
        exit_spd  = round(float(xr["speed"]), 2)

        brk_max  = 0.0
        brk_dist = 0.0
        if "brake" in df.columns:
            bs = df.iloc[corner.entry_idx:corner.apex_idx]
            brk_max  = round(float(bs["brake"].max()), 3)
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
                    dT = rw["throttle"].iloc[-1] - rw["throttle"].iloc[0]
                    dt2= rw["time"].iloc[-1]     - rw["time"].iloc[0]
                    tp_rate = round(float(dT / dt2) if dt2 > 0 else 0.0, 4)

        # ERS in corner
        ers_dep = _integrate(df.iloc[corner.entry_idx:corner.exit_idx],
                             "ers_deployment_kw", dt.iloc[corner.entry_idx:corner.exit_idx])
        ers_har = _integrate(df.iloc[corner.entry_idx:corner.exit_idx],
                             "ers_harvesting_kw", dt.iloc[corner.entry_idx:corner.exit_idx])

        # Lateral G
        lat_g = 0.0
        df_idx = 0.0
        if "g_lateral" in df.columns:
            seg   = df.iloc[corner.entry_idx:corner.exit_idx]["g_lateral"].abs()
            lat_g = round(float(seg.max()), 3)
            # Downforce index: normalise lateral G by (v²/v_ref²)
            v_ref = 200.0   # km/h reference
            v_sq  = (float(ar["speed"]) / v_ref) ** 2
            df_idx = round(lat_g / v_sq, 3) if v_sq > 0 else 0.0

        # Scores
        late_brk = round(max(0, 100 * (1 - brk_dist / 150.0)), 1)
        exit_q   = round(min(100, 100 * exit_spd / max(entry_spd, 1)), 1)
        ers_eff  = round(min(100, ers_dep / 50 * 100), 1) if ers_dep > 0 else 0.0

        return CornerMetrics(
            corner_id            = corner.corner_id,
            corner_type          = corner.corner_type,
            entry_speed          = entry_spd,
            min_speed            = min_spd,
            exit_speed           = exit_spd,
            brake_start_dist     = round(float(er["distance"]), 1),
            brake_pressure_max   = brk_max,
            braking_distance_m   = brk_dist,
            throttle_pickup_dist = tp_dist,
            throttle_ramp_rate   = tp_rate,
            ers_deployed_kj      = round(ers_dep, 3),
            ers_harvested_kj     = round(ers_har, 3),
            ers_net_kj           = round(ers_dep - ers_har, 3),
            lateral_g_max        = lat_g,
            downforce_index      = df_idx,
            corner_time_s        = corner.corner_time_s,
            late_braking_score   = late_brk,
            exit_quality_score   = exit_q,
            ers_efficiency_score = ers_eff,
        )
    except Exception as e:
        logger.warning(f"Corner {corner.corner_id} metrics failed: {e}")
        return None


# ── Straight metrics ──────────────────────────────────────────────────────────

def _straight_metrics(df, corners) -> list[StraightMetrics]:
    straights = []
    if not corners:
        return straights

    sid = 1
    prev_exit = df["distance"].min()

    for c in corners:
        seg_mask = (df["distance"] >= prev_exit) & (df["distance"] <= c.entry_distance)
        seg = df[seg_mask]
        if len(seg) < 5:
            prev_exit = c.exit_distance
            continue

        top_spd  = float(seg["speed"].max())
        avg_ers  = float(seg["ers_deployment_kw"].mean()) if "ers_deployment_kw" in seg else 0.0
        drs_on   = bool(seg["drs_state"].any()) if "drs_state" in seg.columns else False
        dt_seg   = seg["time"].diff().fillna(0)
        time_s   = float(dt_seg.sum())

        straights.append(StraightMetrics(
            straight_id  = sid,
            start_dist   = round(prev_exit, 1),
            end_dist     = round(c.entry_distance, 1),
            top_speed    = round(top_spd, 1),
            avg_ers_kw   = round(avg_ers, 1),
            drs_active   = drs_on,
            time_s       = round(time_s, 3),
        ))
        sid += 1
        prev_exit = c.exit_distance

    return straights


def compute_tyre_degradation(lap_metrics_list: list[LapMetrics],
                              compound: str = "MEDIUM") -> list[TyreDegradation]:
    """
    Compare speed across laps on same compound → degradation rate (km/h/lap).
    """
    if len(lap_metrics_list) < 2:
        return []
    base_speed = lap_metrics_list[0].avg_speed_kmh
    results    = []
    speeds     = [l.avg_speed_kmh for l in lap_metrics_list]

    if len(speeds) > 1:
        deg_rate = (speeds[0] - speeds[-1]) / max(len(speeds) - 1, 1)
    else:
        deg_rate = 0.0

    for i, lm in enumerate(lap_metrics_list):
        results.append(TyreDegradation(
            lap_id             = lm.lap_id,
            tyre_age_laps      = i,
            compound           = compound,
            avg_lap_speed      = lm.avg_speed_kmh,
            speed_loss_vs_new  = round(base_speed - lm.avg_speed_kmh, 3),
            predicted_deg_rate = round(deg_rate, 4),
        ))
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _integrate(df, col, dt):
    """Integrate col × dt → energy in kJ."""
    if col not in df.columns:
        return 0.0
    return float((df[col] * dt).sum()) / 1000.0


def metrics_to_dataframe(lap_metrics: LapMetrics) -> pd.DataFrame:
    return pd.DataFrame([c.to_dict() for c in lap_metrics.corner_metrics])


def print_lap_summary(lm: LapMetrics):
    sep = "═" * 68
    print(f"\n{sep}")
    print(f"  LAP {lm.lap_id}  ·  {lm.lap_time_s:.3f} s  "
          f"(fuel-corrected: {lm.fuel_corrected_time:.3f} s)")
    print(sep)
    print(f"  Avg speed     : {lm.avg_speed_kmh:.1f} km/h   Max: {lm.max_speed_kmh:.1f} km/h")
    print(f"  Brake time    : {lm.total_brake_time_s:.2f} s   Coast: {lm.total_coasting_time_s:.2f} s")
    print(f"  Avg throttle  : {lm.avg_throttle_pct:.1f} %    Top gear: {lm.top_gear}")
    print(f"  ERS deployed  : {lm.total_ers_deployed_kj:.1f} kJ   Harvested: {lm.total_ers_harvested_kj:.1f} kJ")
    print(f"  Lateral G     : avg {lm.avg_lateral_g:.2f} g   max {lm.max_lateral_g:.2f} g")
    print(f"  Fuel load     : {lm.fuel_load_kg:.1f} kg")
    print(f"  Corners       : {lm.n_corners}")
    print("─" * 68)
    for cm in lm.corner_metrics:
        print(f"  {cm}")
    print(sep)
