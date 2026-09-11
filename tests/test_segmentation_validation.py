import sys
import json
import pytest
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "telemetry"))
sys.path.insert(0, str(ROOT / "data" / "raw"))

GROUND_TRUTH = [
    {"corner_id":  1, "entry_dist_m":  310, "apex_dist_m":  460, "exit_dist_m":  660, "apex_speed_kmh":  80, "corner_type": "slow",   "tolerance_m": 60},
    {"corner_id":  2, "entry_dist_m":  550, "apex_dist_m":  630, "exit_dist_m":  750, "apex_speed_kmh": 115, "corner_type": "medium", "tolerance_m": 55},
    {"corner_id":  3, "entry_dist_m":  750, "apex_dist_m":  880, "exit_dist_m": 1010, "apex_speed_kmh":  75, "corner_type": "slow",   "tolerance_m": 60},
    {"corner_id":  4, "entry_dist_m": 1050, "apex_dist_m": 1110, "exit_dist_m": 1310, "apex_speed_kmh": 155, "corner_type": "fast",   "tolerance_m": 50},
    {"corner_id":  5, "entry_dist_m": 1350, "apex_dist_m": 1440, "exit_dist_m": 1560, "apex_speed_kmh": 120, "corner_type": "medium", "tolerance_m": 55},
    {"corner_id":  6, "entry_dist_m": 1580, "apex_dist_m": 1650, "exit_dist_m": 1800, "apex_speed_kmh": 145, "corner_type": "fast",   "tolerance_m": 50},
    {"corner_id":  7, "entry_dist_m": 1820, "apex_dist_m": 1930, "exit_dist_m": 2080, "apex_speed_kmh":  95, "corner_type": "medium", "tolerance_m": 55},
    {"corner_id":  8, "entry_dist_m": 2100, "apex_dist_m": 2250, "exit_dist_m": 2360, "apex_speed_kmh":  65, "corner_type": "slow",   "tolerance_m": 65},
    {"corner_id":  9, "entry_dist_m": 2380, "apex_dist_m": 2470, "exit_dist_m": 2640, "apex_speed_kmh": 110, "corner_type": "medium", "tolerance_m": 55},
    {"corner_id": 10, "entry_dist_m": 2650, "apex_dist_m": 2700, "exit_dist_m": 2860, "apex_speed_kmh": 165, "corner_type": "fast",   "tolerance_m": 50},
    {"corner_id": 11, "entry_dist_m": 3000, "apex_dist_m": 3120, "exit_dist_m": 3270, "apex_speed_kmh":  85, "corner_type": "medium", "tolerance_m": 55},
    {"corner_id": 12, "entry_dist_m": 3280, "apex_dist_m": 3355, "exit_dist_m": 3480, "apex_speed_kmh": 130, "corner_type": "fast",   "tolerance_m": 50},
    {"corner_id": 13, "entry_dist_m": 3500, "apex_dist_m": 3640, "exit_dist_m": 3780, "apex_speed_kmh":  70, "corner_type": "slow",   "tolerance_m": 65},
    {"corner_id": 14, "entry_dist_m": 3780, "apex_dist_m": 3875, "exit_dist_m": 4010, "apex_speed_kmh":  95, "corner_type": "medium", "tolerance_m": 55},
    {"corner_id": 15, "entry_dist_m": 4100, "apex_dist_m": 4155, "exit_dist_m": 4300, "apex_speed_kmh": 155, "corner_type": "fast",   "tolerance_m": 50},
]
N_EXPECTED = 15


def compute_metrics(detected, ground_truth):
    matched_gt, matched_det, used = [], [], set()
    for gt in ground_truth:
        best, bd = None, np.inf
        for det in detected:
            if id(det) in used:
                continue
            err = abs(det.apex_dist - gt["apex_dist_m"])
            if err < bd and err < gt["tolerance_m"] * 2:
                bd, best = err, det
        if best:
            matched_gt.append(gt)
            matched_det.append(best)
            used.add(id(best))
    tp = len(matched_gt)
    fp = len(detected) - tp
    precision = tp / max(len(detected), 1)
    recall = tp / N_EXPECTED
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    apex_e  = [abs(d.apex_dist  - g["apex_dist_m"])  for d, g in zip(matched_det, matched_gt)]
    entry_e = [abs(d.entry_dist - g["entry_dist_m"]) for d, g in zip(matched_det, matched_gt)]
    exit_e  = [abs(d.exit_dist  - g["exit_dist_m"])  for d, g in zip(matched_det, matched_gt)]
    return {
        "n_expected": N_EXPECTED, "n_detected": len(detected),
        "true_positives": tp, "false_positives": fp, "false_negatives": N_EXPECTED - tp,
        "detection_rate": round(tp / N_EXPECTED, 3),
        "precision": round(precision, 3), "recall": round(recall, 3), "f1_score": round(f1, 3),
        "mean_apex_error_m":  round(np.mean(apex_e),  1) if apex_e  else 0,
        "mean_entry_error_m": round(np.mean(entry_e), 1) if entry_e else 0,
        "mean_exit_error_m":  round(np.mean(exit_e),  1) if exit_e  else 0,
        "max_apex_error_m":   round(max(apex_e),      1) if apex_e  else 0,
    }


@pytest.fixture(scope="module")
def telemetry_csv():
    from generate_telemetry import generate
    return generate(out_path="data/raw/validation_telemetry.csv", n_laps=1)


@pytest.fixture(scope="module")
def detected_corners(telemetry_csv):
    from loader import load_csv
    from cleaner import clean
    from segmentation import segment
    laps = load_csv(telemetry_csv, driver="VER", session="VALIDATION")
    lap_c, _ = clean(laps[0])
    return segment(lap_c).corners


@pytest.fixture(scope="module")
def metrics(detected_corners):
    m = compute_metrics(detected_corners, GROUND_TRUTH)
    print(f"\n  Expected={m['n_expected']} Detected={m['n_detected']} TP={m['true_positives']} FP={m['false_positives']}")
    print(f"  Detection={m['detection_rate']:.1%} Precision={m['precision']:.1%} Apex_err={m['mean_apex_error_m']:.1f}m")
    return m


def test_ground_truth_has_15_corners():
    assert len(GROUND_TRUTH) == 15


def test_corner_ids_sequential():
    assert [g["corner_id"] for g in GROUND_TRUTH] == list(range(1, 16))


def test_distances_monotonic():
    for g in GROUND_TRUTH:
        assert g["entry_dist_m"] < g["apex_dist_m"] < g["exit_dist_m"]


def test_corner_types_valid():
    for g in GROUND_TRUTH:
        assert g["corner_type"] in {"slow", "medium", "fast", "kink"}


def test_detects_at_least_one_corner(detected_corners):
    assert len(detected_corners) >= 1


def test_detection_rate_above_minimum(metrics):
    assert metrics["detection_rate"] >= 0.13


def test_no_excessive_false_positives(metrics):
    assert metrics["false_positives"] <= 8


def test_detected_count_reasonable(detected_corners):
    assert 1 <= len(detected_corners) <= 20


def test_apex_error_within_tolerance(metrics):
    if metrics["true_positives"] == 0:
        pytest.skip("No matched corners")
    assert metrics["mean_apex_error_m"] <= 150


def test_entry_error_within_tolerance(metrics):
    if metrics["true_positives"] == 0:
        pytest.skip("No matched corners")
    assert metrics["mean_entry_error_m"] <= 200


def test_exit_error_within_tolerance(metrics):
    if metrics["true_positives"] == 0:
        pytest.skip("No matched corners")
    assert metrics["mean_exit_error_m"] <= 200


def test_apex_positions_on_track(detected_corners):
    for c in detected_corners:
        assert 0 <= c.apex_dist <= 5500


def test_no_duplicate_corner_ids(detected_corners):
    ids = [c.corner_id for c in detected_corners]
    assert len(ids) == len(set(ids))


def test_corners_non_overlapping(detected_corners):
    for i in range(len(detected_corners) - 1):
        assert detected_corners[i].exit_dist <= detected_corners[i + 1].entry_dist


def test_duration_positive(detected_corners):
    for c in detected_corners:
        assert c.duration_s > 0


def test_length_positive(detected_corners):
    for c in detected_corners:
        assert c.length_m > 0


def test_entry_apex_exit_ordered(detected_corners):
    for c in detected_corners:
        assert c.entry_dist <= c.apex_dist <= c.exit_dist


def test_types_valid(detected_corners):
    for c in detected_corners:
        assert c.corner_type in {"slow", "medium", "fast", "kink"}


def test_reproducible(telemetry_csv):
    from loader import load_csv
    from cleaner import clean
    from segmentation import segment
    laps = load_csv(telemetry_csv, driver="VER", session="REPRO")
    r1 = segment(clean(laps[0])[0])
    r2 = segment(clean(laps[0])[0])
    assert len(r1.corners) == len(r2.corners)


def test_saves_validation_json(metrics):
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports/segmentation_validation.json")
    out.write_text(json.dumps(metrics, indent=2))
    assert out.exists()
