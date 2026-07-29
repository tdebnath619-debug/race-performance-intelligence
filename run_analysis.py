"""
run_analysis.py
===============
Motorsport telemetry analysis pipeline — entry point.

Usage
-----
    # Synthetic data (no internet required)
    python run_analysis.py

    # Real F1 data via FastF1
    python run_analysis.py --fastf1 --year 2024 --gp Bahrain --session Q --driver_a VER --driver_b LEC

    # Custom CSV
    python run_analysis.py data/raw/my_telemetry.csv
"""

import sys
import logging
import argparse
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
    if p not in sys.path:
        sys.path.insert(0, p)

from loader       import load_csv, lap_summary
from cleaner      import clean
from segmentation import segment, corners_to_dataframe
from metrics      import compute_lap_metrics, print_lap_summary
from delta        import compare
from report       import save_report


def run_csv(csv_path: Path, driver_a="VER", driver_b="LEC",
            session="Bahrain_2024_Q"):
    """Run pipeline on a CSV telemetry file."""
    print("\n" + "═"*68)
    print("  MOTORSPORT TELEMETRY ANALYSIS PIPELINE")
    print("  Source: " + str(csv_path))
    print("═"*68 + "\n")

    log.info("Stage 1/5 — Load")
    laps = load_csv(csv_path, driver=driver_a, session=session)
    print(lap_summary(laps).to_string(index=False))

    if len(laps) < 2:
        log.warning("Only 1 lap found — running single-lap analysis.")
        lap_c, _ = clean(laps[0])
        seg = segment(lap_c)
        lm  = compute_lap_metrics(lap_c, seg.corners)
        print_lap_summary(lm)
        _save_and_dashboard(lm, None)
        return

    lap_a_raw, lap_b_raw = laps[0], laps[1]
    lap_a_raw.driver = driver_a
    lap_b_raw.driver = driver_b

    _process_and_compare(lap_a_raw, lap_b_raw, session)


def run_fastf1(year: int, gp: str, session: str,
               driver_a: str, driver_b: str):
    """Run pipeline on real F1 data via FastF1."""
    print("\n" + "═"*68)
    print("  MOTORSPORT TELEMETRY ANALYSIS PIPELINE")
    print(f"  Source: FastF1 — {year} {gp} {session} | {driver_a} vs {driver_b}")
    print("═"*68 + "\n")

    try:
        from loader_fastf1 import load_fastf1_comparison, session_info

        log.info("Stage 0 — Session overview")
        overview = session_info(year, gp, session)
        if not overview.empty:
            print(overview.to_string(index=False))
            print()

        log.info("Stage 1/5 — Load FastF1 data")
        lap_a, lap_b = load_fastf1_comparison(year, gp, session, driver_a, driver_b)
        _process_and_compare(lap_a, lap_b, f"{year}_{gp}_{session}")

    except ImportError:
        log.error("FastF1 not installed. Run: pip install fastf1")
        sys.exit(1)
    except Exception as e:
        log.error("FastF1 load failed: %s", e)
        sys.exit(1)


def _process_and_compare(lap_a_raw, lap_b_raw, session: str):
    driver_a = lap_a_raw.driver
    driver_b = lap_b_raw.driver

    log.info("Stage 2/5 — Clean")
    lap_a, _ = clean(lap_a_raw)
    lap_b, _ = clean(lap_b_raw)

    log.info("Stage 3/5 — Segment")
    seg_a = segment(lap_a)
    seg_b = segment(lap_b)

    print(f"\nCorners — {driver_a}: {len(seg_a.corners)}")
    for c in seg_a.corners: print(f"  {c}")
    print(f"\nCorners — {driver_b}: {len(seg_b.corners)}")
    for c in seg_b.corners: print(f"  {c}")

    log.info("Stage 4/5 — Metrics")
    lm_a = compute_lap_metrics(lap_a, seg_a.corners)
    lm_b = compute_lap_metrics(lap_b, seg_b.corners)
    print_lap_summary(lm_a)
    print_lap_summary(lm_b)

    log.info("Stage 5/5 — Delta")
    comparison = compare(lap_a, lap_b, seg_a, seg_b)
    comparison.print_report()

    _save_and_dashboard(lm_a, comparison)

    # Optional comparison chart
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


def generate_dashboard():
    """Generate HTML dashboard from pipeline JSON output."""
    try:
        from dashboard import generate
        out = generate(
            analysis_path = "reports/analysis.json",
            delta_path    = "reports/delta_report.json",
            out_path      = "index.html",
        )
        print(f"  Dashboard -> {out}")
    except Exception as e:
        log.warning("Dashboard generation failed: %s", e)


def _save_and_dashboard(lm, comparison):
    Path("reports").mkdir(exist_ok=True)
    if comparison:
        comparison.to_json("reports/delta_report.json")
    save_report(lm, out_dir="reports", stem="analysis")
    generate_dashboard()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description="Motorsport telemetry pipeline")
    p.add_argument("csv", nargs="?", help="Path to telemetry CSV")
    p.add_argument("--fastf1",   action="store_true", help="Use FastF1 real data")
    p.add_argument("--year",     type=int, default=2024)
    p.add_argument("--gp",       default="Bahrain")
    p.add_argument("--session",  default="Q")
    p.add_argument("--driver_a", default="VER")
    p.add_argument("--driver_b", default="LEC")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.fastf1:
        run_fastf1(args.year, args.gp, args.session,
                   args.driver_a, args.driver_b)
    elif args.csv:
        run_csv(Path(args.csv), args.driver_a, args.driver_b,
                f"{args.gp}_{args.session}")
    else:
        # Default: synthetic data
        from generate_telemetry import generate
        csv_path = generate()
        run_csv(csv_path, driver_a="VER", driver_b="LEC",
                session="Bahrain_2024_Q")
