"""
F1 Performance Intelligence System
aero/aero_analysis.py

Aerodynamic performance extraction from telemetry:
- Drag coefficient proxy (straight-line deceleration without braking)
- Downforce index per corner (lateral G / v²)
- Aero balance estimation (front vs rear grip balance)
- Top speed vs downforce trade-off analysis
- DRS drag reduction quantification
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Reference constants (F1 2024 car nominal values)
CAR_MASS_KG   = 800.0   # min car + driver weight (kg)
FRONTAL_AREA  = 1.5     # approximate m²
AIR_DENSITY   = 1.225   # kg/m³ (sea level, 20°C)
G             = 9.81    # m/s²


@dataclass
class AeroMetrics:
    """Lap-level aerodynamic performance summary."""
    lap_id:              int
    # Drag
    cd_proxy:            float   # drag coefficient estimate (dimensionless)
    drag_force_n_100:    float   # drag force at 100 km/h (N)
    drag_force_n_300:    float   # drag force at 300 km/h (N)
    # Downforce
    avg_downforce_index: float   # lateral_G / (v/200)² averaged over corners
    peak_downforce_index:float
    # DRS
    drs_speed_gain_kmh:  float   # avg speed gain when DRS active
    drs_cd_reduction:    float   # estimated Cd reduction from DRS
    # Balance
    aero_balance_pct:    float   # front downforce % (proxy from g behaviour)
    # Corner classification
    slow_corner_grip:    float   # avg min speed in slow corners
    fast_corner_grip:    float   # avg min speed in fast corners

    def to_dict(self):
        return asdict(self)

    def __repr__(self):
        return (
            f"Aero | Cd≈{self.cd_proxy:.3f} | "
            f"DF_index={self.avg_downforce_index:.2f} | "
            f"DRS gain={self.drs_speed_gain_kmh:.1f} km/h | "
            f"Balance={self.aero_balance_pct:.1f}% front"
        )


def compute_aero_metrics(
    df: pd.DataFrame,
    corner_metrics: list,   # list[CornerMetrics]
    lap_id: int = 0,
) -> AeroMetrics:
    """
    Compute aerodynamic performance indices from telemetry.
    """
    cd      = _estimate_drag(df)
    df_idx  = _downforce_index(df, corner_metrics)
    drs     = _drs_analysis(df)
    balance = _aero_balance(df, corner_metrics)
    grip    = _corner_grip_by_type(corner_metrics)

    drag_100 = _drag_force(cd, 100 / 3.6)
    drag_300 = _drag_force(cd, 300 / 3.6)

    return AeroMetrics(
        lap_id               = lap_id,
        cd_proxy             = round(cd, 4),
        drag_force_n_100     = round(drag_100, 1),
        drag_force_n_300     = round(drag_300, 1),
        avg_downforce_index  = round(df_idx["avg"], 3),
        peak_downforce_index = round(df_idx["peak"], 3),
        drs_speed_gain_kmh   = round(drs["speed_gain"], 2),
        drs_cd_reduction     = round(drs["cd_reduction"], 4),
        aero_balance_pct     = round(balance, 1),
        slow_corner_grip     = round(grip["slow"], 1),
        fast_corner_grip     = round(grip["fast"], 1),
    )


# ── Drag estimation ───────────────────────────────────────────────────────────

def _estimate_drag(df: pd.DataFrame) -> float:
    """
    Estimate drag from straight-line coast-down sections
    (throttle ≈ 0, brake ≈ 0, speed > 100 km/h).
    F = ma = 0.5 × ρ × Cd × A × v²
    Cd = 2m × deceleration / (ρ × A × v²)
    """
    if "throttle" not in df.columns or "brake" not in df.columns:
        return 1.0  # F1 typical Cd ~1.0

    coast = df[
        (df["throttle"] < 0.05) &
        (df["brake"]    < 0.05) &
        (df["speed"]    > 100.0)
    ].copy()

    if len(coast) < 10:
        return 1.0

    v_ms  = coast["speed"] / 3.6
    decel = coast.get("d_speed", pd.Series(dtype=float)) / 3.6  # m/s²

    if "d_speed" not in coast.columns:
        dt    = coast["time"].diff().replace(0, np.nan)
        decel = coast["speed"].diff() / dt / 3.6

    decel = decel.dropna()
    decel = decel[decel < -0.5]   # only real deceleration

    if len(decel) < 5:
        return 1.0

    v_match = v_ms.loc[decel.index]
    cd_vals = (-2 * CAR_MASS_KG * decel) / (AIR_DENSITY * FRONTAL_AREA * v_match**2)
    cd_vals = cd_vals[(cd_vals > 0.5) & (cd_vals < 2.0)]

    return float(cd_vals.median()) if len(cd_vals) > 0 else 1.0


def _drag_force(cd: float, v_ms: float) -> float:
    """F_drag = 0.5 × ρ × Cd × A × v²"""
    return 0.5 * AIR_DENSITY * cd * FRONTAL_AREA * v_ms ** 2


# ── Downforce analysis ────────────────────────────────────────────────────────

def _downforce_index(df: pd.DataFrame, corner_metrics: list) -> dict:
    """
    Downforce index = lateral_G / (v/v_ref)²
    Higher index → more effective downforce per unit speed².
    """
    if not corner_metrics or "g_lateral" not in df.columns:
        return {"avg": 0.0, "peak": 0.0}

    indices = []
    for cm in corner_metrics:
        if cm.lateral_g_max > 0 and cm.min_speed > 0:
            v_norm = cm.min_speed / 200.0
            idx    = cm.lateral_g_max / (v_norm ** 2)
            if 0 < idx < 10:
                indices.append(idx)

    if not indices:
        return {"avg": 0.0, "peak": 0.0}
    return {"avg": float(np.mean(indices)), "peak": float(np.max(indices))}


# ── DRS analysis ─────────────────────────────────────────────────────────────

def _drs_analysis(df: pd.DataFrame) -> dict:
    """Quantify DRS speed benefit and estimate Cd reduction."""
    if "drs_state" not in df.columns:
        return {"speed_gain": 0.0, "cd_reduction": 0.0}

    drs_on  = df[df["drs_state"] == 1]["speed"]
    drs_off = df[df["drs_state"] == 0]["speed"]

    if drs_on.empty or drs_off.empty:
        return {"speed_gain": 0.0, "cd_reduction": 0.0}

    # Compare only at high speed (>220 km/h) to isolate straight sections
    drs_on_high  = df[(df["drs_state"] == 1) & (df["speed"] > 220)]["speed"]
    drs_off_high = df[(df["drs_state"] == 0) & (df["speed"] > 220)]["speed"]

    speed_gain = (float(drs_on_high.mean()) - float(drs_off_high.mean())) \
                  if (not drs_on_high.empty and not drs_off_high.empty) else 0.0

    # Cd reduction proxy: DRS reduces Cd by ~10–15% on most cars
    cd_reduction = 0.12 if speed_gain > 5 else (speed_gain / 50 * 0.12)

    return {"speed_gain": max(0.0, speed_gain), "cd_reduction": cd_reduction}


# ── Aero balance ──────────────────────────────────────────────────────────────

def _aero_balance(df: pd.DataFrame, corner_metrics: list) -> float:
    """
    Estimate front aero balance from lateral G behaviour:
    - High front balance → understeer → lower peak lateral G in slow corners
    - Low front balance  → oversteer  → lower peak lateral G in fast corners
    Returns approximate front aero % (typical F1: 45-50%).
    """
    if not corner_metrics:
        return 47.5  # nominal

    slow = [cm.lateral_g_max for cm in corner_metrics
            if cm.corner_type == "slow" and cm.lateral_g_max > 0]
    fast = [cm.lateral_g_max for cm in corner_metrics
            if cm.corner_type in ("fast", "kink") and cm.lateral_g_max > 0]

    if not slow or not fast:
        return 47.5

    slow_avg = np.mean(slow)
    fast_avg = np.mean(fast)

    # Simple normalised proxy: if slow corners have relatively high G,
    # front end is better → lower front wing → lower front balance %
    ratio = slow_avg / max(fast_avg, 0.1)
    balance = 50.0 - (ratio - 1.0) * 5.0
    return float(np.clip(balance, 42.0, 55.0))


def _corner_grip_by_type(corner_metrics: list) -> dict:
    slow = [cm.min_speed for cm in corner_metrics if cm.corner_type == "slow"]
    fast = [cm.min_speed for cm in corner_metrics if cm.corner_type in ("fast","kink")]
    return {
        "slow": float(np.mean(slow)) if slow else 0.0,
        "fast": float(np.mean(fast)) if fast else 0.0,
    }


# ── Setup recommendation ──────────────────────────────────────────────────────

def aero_setup_recommendation(aero: AeroMetrics) -> str:
    """
    Generate a plain-language aero setup recommendation.
    Mirrors what a real F1 race engineer would consider.
    """
    lines = ["── Aero Setup Recommendation ──────────────────────"]

    # Front wing
    if aero.aero_balance_pct > 50:
        lines.append("  FRONT WING: Reduce by 1-2 clicks — car over-rotating, too much front downforce.")
    elif aero.aero_balance_pct < 45:
        lines.append("  FRONT WING: Increase by 1-2 clicks — car understeering into slow corners.")
    else:
        lines.append("  FRONT WING: Balance within nominal range (45-50%). No change needed.")

    # Rear wing / DRS
    if aero.drs_speed_gain_kmh < 8:
        lines.append("  REAR WING:  Low DRS gain — check DRS actuator or consider higher downforce rear wing.")
    elif aero.drs_speed_gain_kmh > 18:
        lines.append("  REAR WING:  Very high DRS gain — rear wing may be too aggressive, costing top speed.")

    # Drag vs downforce trade-off
    if aero.cd_proxy > 1.15:
        lines.append("  DRAG:       High Cd detected. Consider lower-drag rear wing for low-DRS circuits.")
    elif aero.cd_proxy < 0.85:
        lines.append("  DRAG:       Low drag — check downforce targets, may lack mechanical grip backup.")

    # Slow vs fast corner balance
    if aero.slow_corner_grip < aero.fast_corner_grip * 0.5:
        lines.append("  SLOW CORNERS: Poor grip — check mechanical balance (springs, ARB) vs aero balance.")

    lines.append("─" * 52)
    return "\n".join(lines)
