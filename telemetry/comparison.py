"""
F1 Performance Intelligence System
telemetry/comparison.py

F1 lap comparison:
- Distance-aligned time delta
- Per-corner analysis with ERS delta
- DRS zone speed comparison
- Mini-sector comparison table
- Driver coaching narrative
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    PLOTTING = True
except ImportError:
    PLOTTING = False


@dataclass
class CornerDelta:
    corner_id:              int
    corner_type:            str
    apex_dist:              float
    time_delta_s:           float   # positive → lap A faster
    entry_speed_delta:      float
    min_speed_delta:        float
    exit_speed_delta:       float
    brake_dist_delta_m:     float
    throttle_dist_delta_m:  float
    ers_delta_kj:           float   # lap A ERS deployed − lap B
    lateral_g_delta:        float
    coaching_note:          str = ""

    def __post_init__(self):
        self.coaching_note = self._coaching()

    def _coaching(self) -> str:
        notes = []
        if abs(self.time_delta_s) >= 0.005:
            w = "A" if self.time_delta_s > 0 else "B"
            notes.append(f"Lap {w} +{abs(self.time_delta_s):.3f}s")
        if abs(self.brake_dist_delta_m) >= 5:
            later = "A" if self.brake_dist_delta_m < 0 else "B"
            notes.append(f"Lap {later} brakes {abs(self.brake_dist_delta_m):.0f}m later")
        if abs(self.throttle_dist_delta_m) >= 5:
            earlier = "A" if self.throttle_dist_delta_m > 0 else "B"
            notes.append(f"Lap {earlier} throttle {abs(self.throttle_dist_delta_m):.0f}m earlier")
        if abs(self.exit_speed_delta) >= 2:
            faster = "A" if self.exit_speed_delta > 0 else "B"
            notes.append(f"Lap {faster} exits {abs(self.exit_speed_delta):.1f}km/h faster")
        if abs(self.ers_delta_kj) >= 1:
            more = "A" if self.ers_delta_kj > 0 else "B"
            notes.append(f"Lap {more} deploys {abs(self.ers_delta_kj):.1f}kJ more ERS here")
        return " | ".join(notes) if notes else "No significant delta."

    def to_dict(self):
        return asdict(self)


@dataclass
class ComparisonReport:
    lap_a_id:        int
    lap_b_id:        int
    total_delta_s:   float
    corner_deltas:   list[CornerDelta] = field(default_factory=list)
    summary:         str = ""
    key_findings:    list[str] = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        d["corner_deltas"] = [c.to_dict() for c in self.corner_deltas]
        return d

    def to_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path


def compare_laps(
    df_a, df_b,
    corners_a, corners_b,
    metrics_a=None, metrics_b=None,
    lap_a_id=0, lap_b_id=1,
) -> tuple:

    common_dist, sig_a, sig_b = _align(df_a, df_b)
    cum_delta = _time_delta(df_a, df_b, common_dist)
    total_delta = round(float(cum_delta[-1]), 3)

    matched = _match_corners(corners_a, corners_b)
    corner_deltas = []

    for ca, cb in matched:
        delta_at_exit = float(np.interp(ca.exit_distance, common_dist, cum_delta))

        # ERS delta
        ers_a = _corner_ers(df_a, ca)
        ers_b = _corner_ers(df_b, cb)

        # Lateral G delta
        g_a = _corner_latg(df_a, ca)
        g_b = _corner_latg(df_b, cb)

        cd = CornerDelta(
            corner_id             = ca.corner_id,
            corner_type           = ca.corner_type,
            apex_dist             = round((ca.apex_distance + cb.apex_distance) / 2, 1),
            time_delta_s          = round(delta_at_exit, 3),
            entry_speed_delta     = round(_spd(df_a, ca.entry_idx) - _spd(df_b, cb.entry_idx), 2),
            min_speed_delta       = round(_spd(df_a, ca.apex_idx)  - _spd(df_b, cb.apex_idx),  2),
            exit_speed_delta      = round(_spd(df_a, ca.exit_idx)  - _spd(df_b, cb.exit_idx),  2),
            brake_dist_delta_m    = round(ca.entry_distance - cb.entry_distance, 1),
            throttle_dist_delta_m = round(ca.apex_distance  - cb.apex_distance,  1),
            ers_delta_kj          = round(ers_a - ers_b, 3),
            lateral_g_delta       = round(g_a - g_b, 3),
        )
        corner_deltas.append(cd)
        logger.info(f"  C{ca.corner_id}: {cd.coaching_note}")

    # Key findings
    findings = _key_findings(corner_deltas, total_delta, lap_a_id, lap_b_id)
    faster = f"Lap {lap_a_id}" if total_delta > 0 else f"Lap {lap_b_id}"
    summary = f"{faster} faster by {abs(total_delta):.3f} s | {len(corner_deltas)} corners compared."

    report = ComparisonReport(
        lap_a_id      = lap_a_id,
        lap_b_id      = lap_b_id,
        total_delta_s = total_delta,
        corner_deltas = corner_deltas,
        summary       = summary,
        key_findings  = findings,
    )
    logger.info(summary)
    return report, cum_delta, common_dist


def plot_comparison(df_a, df_b, cum_delta, common_dist,
                    lap_a_label="Lap A", lap_b_label="Lap B",
                    out_path=None, show=False):
    if not PLOTTING:
        return None

    fig = plt.figure(figsize=(18, 12), facecolor="#0D0D0D")
    gs  = gridspec.GridSpec(5, 1, hspace=0.06, figure=fig,
                             top=0.94, bottom=0.06, left=0.07, right=0.97)

    RED   = "#E8002D"    # F1 red
    BLUE  = "#00A3E0"    # team blue
    WHITE = "#FFFFFF"
    GRAY  = "#333333"

    panels = [
        ("speed",             "Speed (km/h)"),
        ("throttle",          "Throttle"),
        ("brake",             "Brake"),
        ("ers_deployment_kw", "ERS Deploy (kW)"),
    ]

    axes = []
    for idx, (col, ylabel) in enumerate(panels):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("#0D0D0D")
        ax.tick_params(colors=WHITE, labelbottom=False, labelsize=7)
        ax.set_ylabel(ylabel, color=WHITE, fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRAY)
        ax.grid(True, color=GRAY, alpha=0.4, linewidth=0.5)

        for df, colour, label in [(df_a, RED, lap_a_label), (df_b, BLUE, lap_b_label)]:
            if col in df.columns:
                v = np.interp(common_dist, df["distance"], df[col])
                ax.plot(common_dist, v, color=colour, linewidth=0.8, label=label)
        axes.append(ax)

    axes[0].legend(loc="upper right", fontsize=8, facecolor="#1A1A1A",
                   labelcolor=WHITE, edgecolor=GRAY)
    axes[0].set_title(f"F1 Lap Comparison: {lap_a_label} vs {lap_b_label}",
                       color=WHITE, fontsize=11, pad=8, fontweight="bold")

    # Delta panel
    ax_d = fig.add_subplot(gs[4])
    ax_d.set_facecolor("#0D0D0D")
    ax_d.tick_params(colors=WHITE, labelsize=7)
    ax_d.set_ylabel("Δ Time (s)", color=WHITE, fontsize=7)
    ax_d.set_xlabel("Distance (m)", color=WHITE, fontsize=8)
    for spine in ax_d.spines.values():
        spine.set_edgecolor(GRAY)
    ax_d.axhline(0, color=WHITE, linewidth=0.6, linestyle="--", alpha=0.5)
    ax_d.fill_between(common_dist, cum_delta, 0,
                       where=cum_delta > 0, alpha=0.4, color=RED,   label=f"{lap_a_label} faster")
    ax_d.fill_between(common_dist, cum_delta, 0,
                       where=cum_delta < 0, alpha=0.4, color=BLUE,  label=f"{lap_b_label} faster")
    ax_d.plot(common_dist, cum_delta, color=WHITE, linewidth=0.7)
    ax_d.legend(loc="upper right", fontsize=7, facecolor="#1A1A1A",
                labelcolor=WHITE, edgecolor=GRAY)
    ax_d.grid(True, color=GRAY, alpha=0.4, linewidth=0.5)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        logger.info(f"Chart saved → {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return out_path


# ── Helpers ───────────────────────────────────────────────────────────────────

def _align(df_a, df_b, n=2000):
    d_min = max(df_a["distance"].min(), df_b["distance"].min())
    d_max = min(df_a["distance"].max(), df_b["distance"].max())
    cd    = np.linspace(d_min, d_max, n)
    sigs  = {}
    for k, df in [("a", df_a), ("b", df_b)]:
        sigs[k] = {col: np.interp(cd, df["distance"], df[col])
                   for col in ("speed","throttle","brake","time","ers_deployment_kw")
                   if col in df.columns}
    return cd, sigs["a"], sigs["b"]


def _time_delta(df_a, df_b, common_dist):
    va = np.interp(common_dist, df_a["distance"], df_a["speed"]) / 3.6
    vb = np.interp(common_dist, df_b["distance"], df_b["speed"]) / 3.6
    dd = np.diff(common_dist, prepend=common_dist[0])
    return np.cumsum(np.where(vb > 0, dd/vb, 0) - np.where(va > 0, dd/va, 0))


def _match_corners(ca_list, cb_list):
    matched, used = [], set()
    for ca in ca_list:
        best, bd = None, np.inf
        for cb in cb_list:
            d = abs(ca.apex_distance - cb.apex_distance)
            if id(cb) not in used and d < bd and d < 300:
                bd, best = d, cb
        if best:
            matched.append((ca, best))
            used.add(id(best))
    return matched


def _spd(df, idx):
    return round(float(df.iloc[max(0, min(idx, len(df)-1))]["speed"]), 2)


def _corner_ers(df, corner):
    if "ers_deployment_kw" not in df.columns:
        return 0.0
    seg = df.iloc[corner.entry_idx:corner.exit_idx]
    dt  = seg["time"].diff().fillna(0)
    return float((seg["ers_deployment_kw"] * dt).sum()) / 1000.0


def _corner_latg(df, corner):
    if "g_lateral" not in df.columns:
        return 0.0
    return float(df.iloc[corner.entry_idx:corner.exit_idx]["g_lateral"].abs().max())


def _key_findings(corner_deltas, total_delta, a_id, b_id):
    findings = []
    if not corner_deltas:
        return findings

    # Biggest time loss corner
    worst = max(corner_deltas, key=lambda c: abs(c.time_delta_s))
    findings.append(
        f"Biggest delta: Corner {worst.corner_id} ({worst.corner_type}) — "
        f"{abs(worst.time_delta_s):.3f} s"
    )
    # Avg ERS difference
    avg_ers = np.mean([c.ers_delta_kj for c in corner_deltas])
    if abs(avg_ers) > 0.5:
        more = f"Lap {a_id}" if avg_ers > 0 else f"Lap {b_id}"
        findings.append(f"{more} deploys {abs(avg_ers):.1f} kJ more ERS per corner on average.")

    # Brake point tendency
    later_brakers = [c for c in corner_deltas if c.brake_dist_delta_m < -10]
    if later_brakers:
        findings.append(
            f"Lap {a_id} brakes later at {len(later_brakers)} corner(s): "
            f"{[c.corner_id for c in later_brakers]}"
        )
    return findings
