import sys
import pytest
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "telemetry"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "raw"))

@pytest.fixture(scope="module")
def synthetic_csv():
    from generate_telemetry import generate
    return generate(out_path="data/raw/test_telemetry.csv", n_laps=2)

@pytest.fixture(scope="module")
def loaded_laps(synthetic_csv):
    from loader import load_csv
    return load_csv(synthetic_csv, driver="VER", session="TEST")

def test_loads_two_laps(loaded_laps):
    assert len(loaded_laps) == 2

def test_speed_non_negative(loaded_laps):
    for lap in loaded_laps:
        assert (lap.data["speed"] >= 0).all()

def test_time_starts_at_zero(loaded_laps):
    for lap in loaded_laps:
        assert lap.data["time"].iloc[0] == pytest.approx(0.0, abs=0.01)

def test_throttle_normalised(loaded_laps):
    for lap in loaded_laps:
        if "throttle" in lap.data.columns:
            assert lap.data["throttle"].max() <= 1.0

def test_brake_normalised(loaded_laps):
    for lap in loaded_laps:
        if "brake" in lap.data.columns:
            assert lap.data["brake"].max() <= 1.0

def test_cleaning_works(loaded_laps):
    from cleaner import clean
    lap_c, report = clean(loaded_laps[0])
    assert len(report.steps) > 0

def test_corners_detected(loaded_laps):
    from cleaner import clean
    from segmentation import segment
    lap_c, _ = clean(loaded_laps[0])
    result = segment(lap_c)
    assert len(result.corners) >= 1

def test_dashboard_generates(tmp_path):
    from dashboard import generate
    out = generate(
        analysis_path="reports/analysis.json",
        delta_path="reports/delta_report.json",
        out_path=str(tmp_path / "index.html"),
    )
    content = out.read_text()
    assert "addEventListener" in content
    assert "data-tab" in content
