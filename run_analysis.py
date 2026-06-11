"""
run_analysis.py
===============
Entry point for the motorsport telemetry analysis pipeline.

Usage
-----
    python run_analysis.py                       # auto-generates synthetic data
    python run_analysis.py data/raw/my_lap.csv   # your own telemetry file

Output
------
    reports/delta_report.json
    reports/analysis_corners.csv
    reports/comparison_chart.png  (if matplotlib available)
"""

import sys, logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

ROOT = Path(__file__).parent
for p in [str(ROOT/"telemetry"), str(ROOT/"data"/"raw"),
          str(ROOT/"strategy"), str(ROOT/"aero"), str(ROOT/"setup")]:
    if p not in sys.path: sys.path.insert(0, p)

from loader       import load_csv, lap_summary
from cleaner      import clean
from segmentation import segment, corners_to_dataframe
from metrics      import compute_lap_metrics, print_lap_summary
from delta        import compare
from report       import save_report


def run(csv_path: Path, driver_a="A", driver_b="B", session="UNKNOWN"):
    print("\n" + "═"*68)
    print("  MOTORSPORT TELEMETRY ANALYSIS PIPELINE")
    print("═"*68 + "\n")

    # ── 1. Load ────────────────────────────────────────────────────────────
    log.info("Stage 1/5 — Load")
    laps = load_csv(csv_path, driver=driver_a, session=session)
    print(lap_summary(laps).to_string(index=False))

    if len(laps) < 2:
        log.warning("Only 1 lap found — running single-lap analysis.")
        lap     = laps[0]
        lap_c,_ = clean(lap)
        seg     = segment(lap_c)
        lm      = compute_lap_metrics(lap_c, seg.corners)
        print_lap_summary(lm)
        return

    # Assign drivers
    lap_a_raw, lap_b_raw = laps[0], laps[1]
    lap_a_raw.driver = driver_a
    lap_b_raw.driver = driver_b

    # ── 2. Clean ───────────────────────────────────────────────────────────
    log.info("Stage 2/5 — Clean")
    lap_a, _ = clean(lap_a_raw)
    lap_b, _ = clean(lap_b_raw)

    # ── 3. Segment ─────────────────────────────────────────────────────────
    log.info("Stage 3/5 — Segment")
    seg_a = segment(lap_a)
    seg_b = segment(lap_b)

    print(f"\nCorners — {driver_a}: {len(seg_a.corners)}")
    for c in seg_a.corners: print(f"  {c}")
    print(f"\nCorners — {driver_b}: {len(seg_b.corners)}")
    for c in seg_b.corners: print(f"  {c}")

    # ── 4. Metrics ─────────────────────────────────────────────────────────
    log.info("Stage 4/5 — Metrics")
    lm_a = compute_lap_metrics(lap_a, seg_a.corners)
    lm_b = compute_lap_metrics(lap_b, seg_b.corners)
    print_lap_summary(lm_a)
    print_lap_summary(lm_b)

    # ── 5. Delta analysis ──────────────────────────────────────────────────
    log.info("Stage 5/5 — Delta")
    comparison = compare(lap_a, lap_b, seg_a, seg_b)
    comparison.print_report()

    # ── Save ───────────────────────────────────────────────────────────────
    Path("reports").mkdir(exist_ok=True)
    comparison.to_json("reports/delta_report.json")
    save_report(lm_a, out_dir="reports", stem="analysis")

    # Optional chart
    try:
        from plot import plot_comparison
        plot_comparison(
            lap_a.data, lap_b.data,
            comparison._cum_delta, comparison._common_dist,
            label_a=driver_a, label_b=driver_b,
            out_path="reports/comparison_chart.png",
        )
    except Exception:
        pass

    print("\n  Reports saved to reports/")
    print("═"*68 + "\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        from generate_telemetry import generate
        csv_path = generate()

    run(csv_path, driver_a="VER", driver_b="LEC", session="Bahrain_2024_Q")
