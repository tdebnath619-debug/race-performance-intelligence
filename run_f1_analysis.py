"""
F1 Performance Intelligence System
run_f1_analysis.py  —  End-to-end pipeline

Usage
-----
    python run_f1_analysis.py                        # synthetic Bahrain data
    python run_f1_analysis.py path/to/telemetry.csv  # real CSV
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)

ROOT = Path(__file__).parent
sys.path.extend([
    str(ROOT / "telemetry"),
    str(ROOT / "strategy"),
    str(ROOT / "aero"),
    str(ROOT / "setup"),
    str(ROOT / "data" / "raw"),
])

# ── Imports ───────────────────────────────────────────────────────────────────
from loader        import load_telemetry, list_laps
from cleaner       import clean_telemetry, compute_derivatives, flag_coasting
from segmentation  import segment_corners, generate_mini_sectors, detect_drs_zones, corners_to_dataframe
from metrics       import compute_lap_metrics, print_lap_summary, compute_tyre_degradation
from comparison    import compare_laps, plot_comparison
from report        import save_report
from aero_analysis import compute_aero_metrics, aero_setup_recommendation
from race_strategy import optimise_strategy, simulate_undercut, simulate_safety_car
from setup_optimizer import analyse_setup, print_setup_report


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║         F1 PERFORMANCE INTELLIGENCE SYSTEM  v1.0                ║
║         Telemetry · Aero · Strategy · Setup                     ║
╚══════════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── 0. Generate or load telemetry ────────────────────────────────────
    if len(sys.argv) > 1:
        INPUT_CSV = Path(sys.argv[1])
    else:
        from generate_f1_telemetry import generate
        INPUT_CSV = generate()

    # ── 1. Load ───────────────────────────────────────────────────────────
    print("\n[1/7] LOADING TELEMETRY")
    print("─" * 60)
    df_all = load_telemetry(INPUT_CSV)
    print(list_laps(df_all).to_string(index=False))

    laps = sorted(df_all["lap_number"].unique())
    results = {}

    # ── 2. Per-lap processing ─────────────────────────────────────────────
    print("\n[2/7] SIGNAL CLEANING + CORNER SEGMENTATION")
    print("─" * 60)
    for lap_num in laps[:2]:
        df_lap = df_all[df_all["lap_number"] == lap_num].copy().reset_index(drop=True)
        df_clean = clean_telemetry(df_lap)
        df_clean = compute_derivatives(df_clean)
        df_clean = flag_coasting(df_clean)
        corners, df_tagged = segment_corners(df_clean)
        mini_sectors = generate_mini_sectors(df_tagged)
        drs_zones    = detect_drs_zones(df_tagged)
        lap_metrics  = compute_lap_metrics(df_tagged, corners, lap_id=lap_num)
        results[lap_num] = dict(df=df_clean, tagged=df_tagged,
                                corners=corners, metrics=lap_metrics,
                                mini_sectors=mini_sectors, drs_zones=drs_zones)

    # ── 3. Performance metrics ────────────────────────────────────────────
    print("\n[3/7] PERFORMANCE METRICS")
    print("─" * 60)
    for lap_num, r in results.items():
        print_lap_summary(r["metrics"])

    # ── 4. Aero analysis ──────────────────────────────────────────────────
    print("\n[4/7] AERODYNAMIC ANALYSIS")
    print("─" * 60)
    aero_results = {}
    for lap_num, r in results.items():
        aero = compute_aero_metrics(r["df"], r["metrics"].corner_metrics, lap_id=lap_num)
        aero_results[lap_num] = aero
        print(f"  Lap {lap_num}: {aero}")
        print(aero_setup_recommendation(aero))

    # ── 5. Setup optimisation ─────────────────────────────────────────────
    print("\n[5/7] SETUP OPTIMISATION")
    print("─" * 60)
    base_lap = list(results.keys())[0]
    setup_recs = analyse_setup(
        results[base_lap]["df"],
        results[base_lap]["metrics"],
        aero_results.get(base_lap),
    )
    print_setup_report(setup_recs)

    # ── 6. Race strategy ──────────────────────────────────────────────────
    print("\n[6/7] RACE STRATEGY ENGINE")
    print("─" * 60)
    base_lt = results[base_lap]["metrics"].lap_time_s
    top_strategies = optimise_strategy(
        total_laps    = 57,          # Bahrain GP laps
        base_lap_time = base_lt,
        max_stops     = 3,
    )
    print(f"\n  Top 5 strategies (57-lap race, base lap {base_lt:.3f} s):\n")
    print(f"  {'Rank':<5} {'Strategy':<30} {'Total time':>12} {'Stops':>6}")
    print("  " + "─" * 58)
    for i, s in enumerate(top_strategies[:5], 1):
        h, m = divmod(int(s["total_time_s"]), 3600)
        m2, sec = divmod(m, 60)
        print(f"  {i:<5} {s['strategy_id']:<30} {h}h {m2:02d}m {sec:02d}s {s['n_stops']:>6}")

    # Undercut simulation
    print(f"\n  Undercut simulation (SOFT undercut vs MEDIUM, gap +2.0 s):")
    uc = simulate_undercut(base_lt, base_lt + 0.3, gap_s=2.0,
                           attacker_compound="SOFT", defender_compound="MEDIUM",
                           n_laps_to_evaluate=6, base_lap_time=base_lt)
    print(f"  → {uc['verdict']}")

    # Safety car scenario
    print(f"\n  Safety Car scenario (lap 25 of 57, gap 4.5 s, pit under SC):")
    sc = simulate_safety_car(current_lap=25, total_laps=57,
                              current_gap_s=4.5, sc_duration_laps=4, pit_during_sc=True)
    print(f"  → {sc['recommendation']}")
    print(f"     Time saved vs green flag stop: {sc['time_saved_vs_green']:.1f} s")

    # ── 7. Lap comparison + reports ───────────────────────────────────────
    if len(results) >= 2:
        print("\n[7/7] LAP COMPARISON + REPORT OUTPUT")
        print("─" * 60)
        lap_ids = list(results.keys())
        a, b    = lap_ids[0], lap_ids[1]
        ra, rb  = results[a], results[b]

        report, cum_delta, common_dist = compare_laps(
            df_a=ra["df"], df_b=rb["df"],
            corners_a=ra["corners"], corners_b=rb["corners"],
            metrics_a=ra["metrics"], metrics_b=rb["metrics"],
            lap_a_id=a, lap_b_id=b,
        )

        print(f"\n  {report.summary}")
        print(f"\n  Key findings:")
        for f in report.key_findings:
            print(f"    • {f}")

        print(f"\n  Corner-by-corner:")
        print(f"  {'Turn':<6} {'Type':<8} {'Δt':>8} {'ΔEntry':>8} {'ΔApex':>8} "
              f"{'ΔExit':>8} {'ΔERS kJ':>9}")
        print("  " + "─" * 62)
        for cd in report.corner_deltas:
            print(f"  T{cd.corner_id:<5} {cd.corner_type:<8} "
                  f"{cd.time_delta_s:>+8.3f} {cd.entry_speed_delta:>+8.1f} "
                  f"{cd.min_speed_delta:>+8.1f} {cd.exit_speed_delta:>+8.1f} "
                  f"{cd.ers_delta_kj:>+9.2f}")

        # Tyre degradation
        tyre_deg = compute_tyre_degradation(
            [ra["metrics"], rb["metrics"]], compound="MEDIUM"
        )

        # Chart
        Path("reports").mkdir(exist_ok=True)
        plot_comparison(
            ra["df"], rb["df"], cum_delta, common_dist,
            lap_a_label=f"Lap {a}", lap_b_label=f"Lap {b}",
            out_path="reports/f1_comparison_chart.png",
        )

        # Full JSON report
        paths = save_report(
            lap_metrics       = ra["metrics"],
            comparison_report = report,
            aero_metrics      = aero_results.get(a),
            strategy_results  = top_strategies,
            tyre_deg          = tyre_deg,
            out_dir           = "reports",
            stem              = "f1_analysis",
        )

        print(f"\n  Output files:")
        for fmt, p in paths.items():
            print(f"    [{fmt.upper()}] {p}")
        print(f"    [PNG] reports/f1_comparison_chart.png")

    print("""
╔══════════════════════════════════════════════════════════════════╗
║  Analysis complete.                                             ║
║  Reports saved to /reports/                                     ║
╚══════════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
