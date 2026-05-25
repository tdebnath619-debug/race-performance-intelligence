"""
F1 Performance Intelligence System
setup/setup_optimizer.py

Car setup recommendations derived from telemetry:
- Suspension stiffness from ride height / travel signals
- Brake bias from brake temp balance
- Differential recommendation from traction zone analysis
- Wing level recommendation from aero analysis
- Tyre pressure proxy from tyre temp spread
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class SetupRecommendation:
    parameter:      str
    current_proxy:  str
    recommendation: str
    confidence:     str   # HIGH / MEDIUM / LOW
    delta_estimate: str   # estimated lap time gain

    def to_dict(self):
        return asdict(self)


def analyse_setup(df: pd.DataFrame, lap_metrics, aero_metrics=None) -> list[SetupRecommendation]:
    """
    Generate setup recommendations from telemetry signals.
    Returns list of SetupRecommendation objects.
    """
    recs = []

    recs += _brake_bias(df)
    recs += _tyre_pressures(df)
    recs += _suspension(df)
    recs += _differential(df, lap_metrics)
    if aero_metrics:
        recs += _wing_levels(aero_metrics)

    logger.info(f"Setup analysis: {len(recs)} recommendations generated.")
    return recs


def _brake_bias(df) -> list:
    recs = []
    cols = ["brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr"]
    if not all(c in df.columns for c in cols):
        return recs

    avg_front = (df["brake_temp_fl"].mean() + df["brake_temp_fr"].mean()) / 2
    avg_rear  = (df["brake_temp_rl"].mean() + df["brake_temp_rr"].mean()) / 2
    ratio = avg_front / max(avg_rear, 1)

    if ratio > 1.4:
        rec = "Reduce front brake bias by 1-2% — fronts significantly hotter than rears."
        est = "~0.05–0.10 s (reduced locking risk)"
        conf = "HIGH"
    elif ratio < 0.9:
        rec = "Increase front brake bias by 1-2% — rears overheating, risk of rear lock."
        est = "~0.05–0.08 s"
        conf = "HIGH"
    else:
        rec = "Brake bias within acceptable range. No change required."
        est = "Neutral"
        conf = "MEDIUM"

    recs.append(SetupRecommendation(
        parameter      = "Brake Bias",
        current_proxy  = f"Front avg {avg_front:.0f}°C / Rear avg {avg_rear:.0f}°C (ratio {ratio:.2f})",
        recommendation = rec,
        confidence     = conf,
        delta_estimate = est,
    ))

    # Left-right balance
    fl_avg = df["brake_temp_fl"].mean()
    fr_avg = df["brake_temp_fr"].mean()
    lr_diff = abs(fl_avg - fr_avg)
    if lr_diff > 80:
        side = "FL" if fl_avg > fr_avg else "FR"
        recs.append(SetupRecommendation(
            parameter      = "Brake Duct Balance",
            current_proxy  = f"FL={fl_avg:.0f}°C FR={fr_avg:.0f}°C (diff={lr_diff:.0f}°C)",
            recommendation = f"{side} running hot — increase brake duct opening on {side} side.",
            confidence     = "MEDIUM",
            delta_estimate = "~0.02–0.05 s (tyre protection)",
        ))
    return recs


def _tyre_pressures(df) -> list:
    recs = []
    cols = ["tyre_surf_fl", "tyre_surf_fr", "tyre_surf_rl", "tyre_surf_rr"]
    if not all(c in df.columns for c in cols):
        return recs

    avgs = {c: df[c].mean() for c in cols}
    peak_temps = {c: df[c].max() for c in cols}

    front_avg = (avgs["tyre_surf_fl"] + avgs["tyre_surf_fr"]) / 2
    rear_avg  = (avgs["tyre_surf_rl"] + avgs["tyre_surf_rr"]) / 2

    OPTIMAL_FRONT = 95.0
    OPTIMAL_REAR  = 100.0

    if front_avg > OPTIMAL_FRONT + 12:
        recs.append(SetupRecommendation(
            parameter      = "Front Tyre Pressure",
            current_proxy  = f"Front avg surface temp: {front_avg:.1f}°C (target ~{OPTIMAL_FRONT}°C)",
            recommendation = "Increase front tyre pressure by 0.5–1.0 psi to reduce overheating.",
            confidence     = "MEDIUM",
            delta_estimate = "~0.03–0.08 s over stint",
        ))
    elif front_avg < OPTIMAL_FRONT - 12:
        recs.append(SetupRecommendation(
            parameter      = "Front Tyre Pressure",
            current_proxy  = f"Front avg surface temp: {front_avg:.1f}°C (target ~{OPTIMAL_FRONT}°C)",
            recommendation = "Reduce front tyre pressure by 0.5 psi — tyres undertemperature, losing grip.",
            confidence     = "MEDIUM",
            delta_estimate = "~0.05–0.10 s",
        ))

    if rear_avg > OPTIMAL_REAR + 12:
        recs.append(SetupRecommendation(
            parameter      = "Rear Tyre Pressure",
            current_proxy  = f"Rear avg surface temp: {rear_avg:.1f}°C (target ~{OPTIMAL_REAR}°C)",
            recommendation = "Increase rear tyre pressure by 0.5–1.0 psi.",
            confidence     = "MEDIUM",
            delta_estimate = "~0.03–0.07 s",
        ))
    return recs


def _suspension(df) -> list:
    recs = []
    cols = ["susp_travel_fl","susp_travel_fr","susp_travel_rl","susp_travel_rr"]
    if not all(c in df.columns for c in cols):
        return recs

    travel_range_fl = df["susp_travel_fl"].max() - df["susp_travel_fl"].min()
    travel_range_rl = df["susp_travel_rl"].max() - df["susp_travel_rl"].min()

    if travel_range_fl > 35:
        recs.append(SetupRecommendation(
            parameter      = "Front Suspension Stiffness",
            current_proxy  = f"FL travel range: {travel_range_fl:.1f} mm",
            recommendation = "Increase front spring rate by 5–10 N/mm — excessive body roll / dive.",
            confidence     = "LOW",
            delta_estimate = "~0.05–0.15 s (circuit dependent)",
        ))
    if travel_range_rl > 40:
        recs.append(SetupRecommendation(
            parameter      = "Rear Suspension Stiffness",
            current_proxy  = f"RL travel range: {travel_range_rl:.1f} mm",
            recommendation = "Increase rear spring rate — excessive squat reducing aero efficiency.",
            confidence     = "LOW",
            delta_estimate = "~0.03–0.10 s",
        ))
    return recs


def _differential(df, lap_metrics) -> list:
    recs = []
    if not hasattr(lap_metrics, "corner_metrics") or not lap_metrics.corner_metrics:
        return recs

    slow_exits = [cm.throttle_ramp_rate for cm in lap_metrics.corner_metrics
                  if cm.corner_type == "slow" and cm.throttle_ramp_rate > 0]
    if not slow_exits:
        return recs

    avg_ramp = np.mean(slow_exits)
    if avg_ramp < 0.5:
        recs.append(SetupRecommendation(
            parameter      = "Rear Differential (On-Power)",
            current_proxy  = f"Avg throttle ramp rate in slow corners: {avg_ramp:.2f} /s",
            recommendation = "Reduce on-power diff lock — driver struggling to open throttle early. Car pushing wide on exit.",
            confidence     = "MEDIUM",
            delta_estimate = "~0.05–0.15 s across slow corners",
        ))
    elif avg_ramp > 2.5:
        recs.append(SetupRecommendation(
            parameter      = "Rear Differential (On-Power)",
            current_proxy  = f"Avg throttle ramp rate: {avg_ramp:.2f} /s",
            recommendation = "Increase on-power diff lock slightly — aggressive ramp may indicate oversteer on exit.",
            confidence     = "LOW",
            delta_estimate = "~0.02–0.08 s",
        ))
    return recs


def _wing_levels(aero) -> list:
    recs = []
    if aero.aero_balance_pct > 50.5:
        recs.append(SetupRecommendation(
            parameter      = "Front Wing Angle",
            current_proxy  = f"Aero balance proxy: {aero.aero_balance_pct:.1f}% front",
            recommendation = "Reduce front wing by 1–2 clicks — front-heavy aero balance causing understeer.",
            confidence     = "MEDIUM",
            delta_estimate = "~0.03–0.08 s in medium-speed corners",
        ))
    elif aero.aero_balance_pct < 45.5:
        recs.append(SetupRecommendation(
            parameter      = "Front Wing Angle",
            current_proxy  = f"Aero balance proxy: {aero.aero_balance_pct:.1f}% front",
            recommendation = "Increase front wing by 1–2 clicks — rear-biased balance causing rotation issues.",
            confidence     = "MEDIUM",
            delta_estimate = "~0.03–0.10 s",
        ))
    if aero.drs_speed_gain_kmh < 8 and aero.drs_speed_gain_kmh > 0:
        recs.append(SetupRecommendation(
            parameter      = "Rear Wing Level",
            current_proxy  = f"DRS speed gain: {aero.drs_speed_gain_kmh:.1f} km/h",
            recommendation = "Low DRS gain — consider reducing rear wing level for better straight-line speed.",
            confidence     = "LOW",
            delta_estimate = "~0.05–0.20 s on DRS straights",
        ))
    return recs


def print_setup_report(recs: list):
    print("\n" + "═" * 68)
    print("  SETUP OPTIMISATION REPORT")
    print("═" * 68)
    for r in recs:
        print(f"\n  [{r.confidence}] {r.parameter}")
        print(f"  Data    : {r.current_proxy}")
        print(f"  Action  : {r.recommendation}")
        print(f"  Est. Δt : {r.delta_estimate}")
    print("═" * 68)
