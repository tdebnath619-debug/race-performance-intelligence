"""
F1 Performance Intelligence System
strategy/race_strategy.py

Real F1 strategy concepts:
- Tyre compound performance windows
- Pit stop window optimisation
- Undercut / overcut simulation
- Safety car scenario modelling
- Fuel delta strategy (fuel saving vs performance)
- Track position vs pace trade-off
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ── Tyre model ────────────────────────────────────────────────────────────────

TYRE_DATA = {
    # compound: {peak_window_laps, deg_s_per_lap, warm_up_laps, delta_vs_medium_s}
    "SOFT":   {"peak_window": (1, 15),  "deg": 0.08,  "warm_up": 2,  "delta":  -0.4},
    "MEDIUM": {"peak_window": (3, 30),  "deg": 0.04,  "warm_up": 3,  "delta":   0.0},
    "HARD":   {"peak_window": (5, 45),  "deg": 0.02,  "warm_up": 5,  "delta":  +0.3},
    "INTER":  {"peak_window": (1, 20),  "deg": 0.15,  "warm_up": 2,  "delta": -1.0},  # vs dry
    "WET":    {"peak_window": (1, 15),  "deg": 0.20,  "warm_up": 2,  "delta": -2.0},
}

PIT_STOP_LOSS_S = 22.0    # typical F1 pit stop time loss (stationary + in/out lap delta)
SAFETY_CAR_SPEED_REDUCTION = 0.6   # 60% of normal speed → lap time multiplier ~1.67


@dataclass
class Stint:
    stint_id:      int
    compound:      str
    start_lap:     int
    end_lap:       int
    tyre_age_laps: int = 0   # if starting on used tyres

    @property
    def stint_length(self) -> int:
        return self.end_lap - self.start_lap + 1

    def lap_time_for_lap(self, base_lap_time: float, lap_in_stint: int) -> float:
        """
        Estimate lap time at a given lap within the stint.
        Accounts for: compound delta, warm-up phase, degradation.
        """
        data = TYRE_DATA.get(self.compound, TYRE_DATA["MEDIUM"])
        # Compound delta
        t = base_lap_time + data["delta"]
        # Warm-up: first N laps slower
        warm_up = data["warm_up"]
        if lap_in_stint <= warm_up:
            t += 0.5 * (warm_up - lap_in_stint + 1) / warm_up
        # Degradation: linear increase per lap
        effective_age = self.tyre_age_laps + lap_in_stint
        t += data["deg"] * effective_age
        return round(t, 3)

    def to_dict(self):
        return asdict(self)


@dataclass
class Strategy:
    strategy_id:     str
    stints:          list[Stint]
    total_laps:      int
    base_lap_time:   float

    def simulate(self) -> dict:
        """Simulate the full race time for this strategy."""
        total_time = 0.0
        lap_times  = []

        for stint in self.stints:
            for lap_in_stint in range(1, stint.stint_length + 1):
                lt = stint.lap_time_for_lap(self.base_lap_time, lap_in_stint)
                lap_times.append({
                    "lap":          stint.start_lap + lap_in_stint - 1,
                    "compound":     stint.compound,
                    "lap_time":     lt,
                    "stint_id":     stint.stint_id,
                    "tyre_age":     stint.tyre_age_laps + lap_in_stint,
                })
                total_time += lt

        # Add pit stop losses (n_stints - 1 stops)
        pit_losses = (len(self.stints) - 1) * PIT_STOP_LOSS_S
        total_time += pit_losses

        return {
            "strategy_id":    self.strategy_id,
            "total_time_s":   round(total_time, 3),
            "pit_losses_s":   pit_losses,
            "n_stops":        len(self.stints) - 1,
            "stints":         [s.to_dict() for s in self.stints],
            "lap_times":      lap_times,
        }

    def to_dict(self):
        return {"strategy_id": self.strategy_id, "stints": [s.to_dict() for s in self.stints]}


# ── Strategy optimiser ────────────────────────────────────────────────────────

def optimise_strategy(
    total_laps:     int,
    base_lap_time:  float,
    max_stops:      int = 3,
    available_compounds: list[str] = None,
    mandatory_compounds: list[str] = None,
) -> list[dict]:
    """
    Enumerate feasible strategies and rank by total race time.

    F1 rules (simplified):
    - Must use at least 2 dry compounds (if no SC / rain)
    - Max stops constrained by race length

    Returns sorted list of strategy simulation results.
    """
    if available_compounds is None:
        available_compounds = ["SOFT", "MEDIUM", "HARD"]
    if mandatory_compounds is None:
        mandatory_compounds = ["MEDIUM"]   # must use at least 1 non-soft dry tyre

    results = []

    for n_stops in range(1, max_stops + 1):
        n_stints = n_stops + 1
        # Generate stint length splits
        for split in _even_splits(total_laps, n_stints):
            for compounds in _compound_combos(available_compounds, n_stints, mandatory_compounds):
                stints = []
                lap = 1
                for i, (length, compound) in enumerate(zip(split, compounds)):
                    stints.append(Stint(
                        stint_id  = i + 1,
                        compound  = compound,
                        start_lap = lap,
                        end_lap   = lap + length - 1,
                    ))
                    lap += length
                strat = Strategy(
                    strategy_id   = f"{n_stops}S_{'_'.join(compounds)}",
                    stints        = stints,
                    total_laps    = total_laps,
                    base_lap_time = base_lap_time,
                )
                results.append(strat.simulate())

    results.sort(key=lambda r: r["total_time_s"])
    logger.info(f"Strategy optimiser: {len(results)} strategies evaluated.")
    return results[:10]  # top 10


def simulate_undercut(
    attacker_lap_time:  float,
    defender_lap_time:  float,
    gap_s:              float,
    attacker_compound:  str = "SOFT",
    defender_compound:  str = "MEDIUM",
    n_laps_to_evaluate: int = 5,
    base_lap_time:      float = 90.0,
) -> dict:
    """
    Simulate an undercut attempt.

    The attacker pits first, installs faster compound, runs faster out-laps.
    Returns whether the undercut succeeds and the gap after n_laps.
    """
    attacker_delta = TYRE_DATA[attacker_compound]["delta"]
    defender_delta = TYRE_DATA[defender_compound]["delta"]
    attacker_deg   = TYRE_DATA[attacker_compound]["deg"]
    defender_deg   = TYRE_DATA[defender_compound]["deg"]

    # Attacker loses PIT_STOP_LOSS_S but gains pace
    current_gap = gap_s + PIT_STOP_LOSS_S   # attacker exits pit behind

    lap_log = []
    for lap in range(1, n_laps_to_evaluate + 1):
        att_t = attacker_lap_time + attacker_delta + attacker_deg * lap
        def_t = defender_lap_time + defender_delta + defender_deg * (lap + 3)  # defender on older tyres
        delta  = def_t - att_t
        current_gap -= delta
        lap_log.append({
            "lap":            lap,
            "attacker_t":     round(att_t, 3),
            "defender_t":     round(def_t, 3),
            "delta_per_lap":  round(delta, 3),
            "gap_after":      round(current_gap, 3),
        })

    success = current_gap < 0
    return {
        "undercut_success": success,
        "final_gap_s":      round(current_gap, 3),
        "laps_evaluated":   n_laps_to_evaluate,
        "lap_log":          lap_log,
        "verdict": (
            f"Undercut SUCCESS — attacker ahead by {abs(current_gap):.2f} s after {n_laps_to_evaluate} laps."
            if success else
            f"Undercut FAILS — defender still ahead by {current_gap:.2f} s after {n_laps_to_evaluate} laps."
        ),
    }


def simulate_safety_car(
    current_lap:       int,
    total_laps:        int,
    current_gap_s:     float,
    sc_duration_laps:  int = 3,
    pit_during_sc:     bool = True,
) -> dict:
    """
    Model the impact of a Safety Car on strategy.
    Under SC all gaps compress to ~2 s (bunch-up).
    If pitting during SC: save pit loss time vs normal pitstop.
    """
    SC_BUNCH_GAP = 2.0   # gaps compress to ~2 s under safety car
    SC_PIT_LOSS  = 15.0  # pitting under SC costs ~15 s (vs 22 s under green)

    gap_after_sc = SC_BUNCH_GAP if current_gap_s > SC_BUNCH_GAP else current_gap_s

    if pit_during_sc:
        pit_cost = SC_PIT_LOSS
        time_saved = PIT_STOP_LOSS_S - SC_PIT_LOSS
    else:
        pit_cost   = PIT_STOP_LOSS_S
        time_saved = 0.0

    laps_remaining = total_laps - current_lap - sc_duration_laps

    return {
        "sc_laps":          sc_duration_laps,
        "gap_before_sc":    round(current_gap_s, 2),
        "gap_after_sc":     round(gap_after_sc, 2),
        "pit_during_sc":    pit_during_sc,
        "pit_cost_s":       pit_cost,
        "time_saved_vs_green": round(time_saved, 1),
        "laps_remaining":   laps_remaining,
        "recommendation": (
            f"PIT UNDER SC — save {time_saved:.1f} s vs green flag stop."
            if pit_during_sc and time_saved > 0
            else "STAY OUT — insufficient time to recover pit loss in remaining laps."
        ),
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _even_splits(total, n, min_len=5):
    """Generate reasonable stint length combinations."""
    results = []
    base    = total // n
    rem     = total % n
    # Generate a few variations around even split
    for offset in range(-5, 6):
        split = []
        remaining = total
        for i in range(n):
            length = max(min_len, base + (1 if i < rem else 0) + offset)
            if i == n - 1:
                length = remaining
            remaining -= length
            split.append(length)
        if all(l >= min_len for l in split) and sum(split) == total:
            if split not in results:
                results.append(split)
    return results[:5]   # limit combinations


def _compound_combos(available, n_stints, mandatory):
    """Generate valid compound combinations (must use mandatory + can't repeat same consecutively)."""
    from itertools import product
    combos = []
    for combo in product(available, repeat=n_stints):
        # No two consecutive identical compounds
        if any(combo[i] == combo[i+1] for i in range(len(combo)-1)):
            continue
        # Must include at least one mandatory compound
        if mandatory and not any(m in combo for m in mandatory):
            continue
        combos.append(list(combo))
    return combos[:20]  # cap for performance
