"""
F1 Performance Intelligence System
telemetry/report.py

Generate structured reports:
- JSON (machine-readable, feeds future modules)
- CSV (corner table)
- Console summary
"""

import json
import csv
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class _Encoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)):  return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray):     return obj.tolist()
        if isinstance(obj, (np.bool_,)):    return bool(obj)
        return super().default(obj)


def save_report(
    lap_metrics,
    comparison_report=None,
    aero_metrics=None,
    strategy_results=None,
    tyre_deg=None,
    out_dir="reports",
    stem="f1_analysis",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "system":   "F1 Performance Intelligence System",
        "version":  "1.0",
        "lap":      lap_metrics.to_dict(),
    }
    if comparison_report:
        report["comparison"] = comparison_report.to_dict()
    if aero_metrics:
        report["aero"] = aero_metrics.to_dict()
    if strategy_results:
        report["top_strategies"] = strategy_results[:5]
    if tyre_deg:
        report["tyre_degradation"] = [t.to_dict() for t in tyre_deg]

    # JSON
    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2, cls=_Encoder))
    logger.info(f"JSON → {json_path}")

    # CSV corners
    csv_path = out_dir / f"{stem}_corners.csv"
    corners  = report["lap"].get("corner_metrics", [])
    if corners:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(corners[0].keys()))
            w.writeheader()
            w.writerows(corners)
        logger.info(f"CSV  → {csv_path}")

    # Strategy CSV
    if strategy_results:
        strat_path = out_dir / f"{stem}_strategies.csv"
        rows = [{k: v for k, v in s.items() if k != "lap_times"}
                for s in strategy_results[:10]]
        if rows:
            with open(strat_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            logger.info(f"Strategies → {strat_path}")

    return {"json": json_path, "csv": csv_path}
