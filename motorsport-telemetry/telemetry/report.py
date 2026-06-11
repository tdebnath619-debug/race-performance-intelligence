"""
telemetry/report.py
===================
Report export — JSON and CSV.
"""
import json, csv
import numpy as np
from pathlib import Path
import logging
log = logging.getLogger(__name__)

class _Enc(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):  return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray):     return o.tolist()
        if isinstance(o, (np.bool_,)):    return bool(o)
        return super().default(o)

def save_report(lap_metrics, comparison_report=None, out_dir="reports", stem="analysis"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report  = {"lap": lap_metrics.to_dict()}
    if comparison_report:
        report["comparison"] = comparison_report.to_dict()

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(report, indent=2, cls=_Enc))
    log.info("JSON → %s", json_path)

    csv_path = out_dir / f"{stem}_corners.csv"
    corners  = report["lap"].get("corner_metrics", [])
    if corners:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(corners[0].keys()))
            w.writeheader(); w.writerows(corners)
        log.info("CSV  → %s", csv_path)
    return {"json": json_path, "csv": csv_path}
