"""
tests/test_telemetry_pipeline.py
==================================
Unit and integration tests for the telemetry analysis pipeline.

Run with:
    pytest tests/ -v

Tests cover:
- Data loading and validation
- Signal cleaning correctness
- Corner detection completeness
- Delta analysis output structure
- Dashboard generation
"""

import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add telemetry to path
sys.path.insert(0, str(Path(__file__).parent.parent / "telemetry"))
sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "raw"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_telemetry():
    """Generate synthetic telemetry data for testing."""
    from generate_telemetry import generate
    path = generate(out_path="data/raw/test_telemetry.csv", n_laps=2)
    return path


@pytest.fixture(scope="module")
def loaded_laps(synthetic_telemetry):
    from loader import load_csv
    return load_csv(synthetic_telemetry, driver="VER", session="TEST")


@pytest.fixture(scope="module")
def cleaned_lap(loaded_laps):
    from cleaner import clean
    lap_c, report = clean(loaded_laps[0])
    return lap_c, report


@pytest.fixture(scope="module")
def segmented_lap(cleaned_lap):
    from segmentation import segment
    lap_c, _ = cleaned_lap
    return segment(lap_c)


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestLoader:

    def test_loads_two_laps(self, loaded_laps):
        assert len(loaded_laps) == 2, "Expected 2 laps from synthetic data"

    def test_required_columns_present(self, loaded_laps):
        lap = loaded_laps[0]
        for col in ["time", "distance", "speed"]:
            assert col in lap.data.columns, f"Required column '{col}' missing"

    def test_time_starts_at_zero(self, loaded_laps):
        for lap in loaded_laps:
            assert lap.data["time"].iloc[0] == pytest.approx(0.0, abs=0.01), \
                "Lap time must start at 0"

    def test_distance_starts_at_zero(self, loaded_laps):
        for lap in loaded_laps:
            assert lap.data["distance"].iloc[0] == pytest.approx(0.0, abs=1.0), \
                "Distance must start at 0"

    def test_speed_non_negative(self, loaded_laps):
        for lap in loaded_laps:
            assert (lap.data["speed"] >= 0).all(), "Speed must be non-negative"

    def test_throttle_normalised(self, loaded_laps):
        for lap in loaded_laps:
            if "throttle" in lap.data.columns:
                assert lap.data["throttle"].max() <= 1.0, "Throttle must be <= 1.0"
                assert lap.data["throttle"].min() >= 0.0, "Throttle must be >= 0.0"

    def test_brake_normalised(self, loaded_laps):
        for lap in loaded_laps:
            if "brake" in lap.data.columns:
                assert lap.data["brake"].max() <= 1.0, "Brake must be <= 1.0"
                assert lap.data["brake"].min() >= 0.0, "Brake must be >= 0.0"

    def test_lap_time_positive(self, loaded_laps):
        for lap in loaded_laps:
            assert lap.lap_time_s > 0, "Lap time must be positive"

    def test_sample_rate_detected(self, loaded_laps):
        for lap in loaded_laps:
            assert lap.sample_rate_hz > 0, "Sample rate must be detected"

    def test_signals_listed(self, loaded_laps):
        lap = loaded_laps[0]
        assert len(lap.signals_present) >= 5, \
            "At least 5 signals should be present"

    def test_driver_assigned(self, loaded_laps):
        assert loaded_laps[0].driver == "VER"


# ---------------------------------------------------------------------------
# Cleaner tests
# ---------------------------------------------------------------------------

class TestCleaner:

    def test_cleaning_report_produced(self, cleaned_lap):
        _, report = cleaned_lap
        assert len(report.steps) > 0, "CleaningReport must contain steps"

    def test_no_nans_after_cleaning(self, cleaned_lap):
        lap_c, _ = cleaned_lap
        num_cols = lap_c.data.select_dtypes(include=np.number).columns
        nan_count = lap_c.data[num_cols].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values after cleaning"

    def test_speed_within_bounds(self, cleaned_lap):
        lap_c, _ = cleaned_lap
        assert lap_c.data["speed"].max() <= 380.0
        assert lap_c.data["speed"].min() >= 0.0

    def test_derivatives_computed(self, cleaned_lap):
        lap_c, _ = cleaned_lap
        assert "d_speed" in lap_c.data.columns, "d_speed derivative missing"

    def test_clip_step_present(self, cleaned_lap):
        _, report = cleaned_lap
        step_names = [s.step for s in report.steps]
        assert "clip_bounds" in step_names

    def test_smooth_step_present(self, cleaned_lap):
        _, report = cleaned_lap
        step_names = [s.step for s in report.steps]
        assert "smooth" in step_names

    def test_throttle_within_bounds(self, cleaned_lap):
        lap_c, _ = cleaned_lap
        if "throttle" in lap_c.data.columns:
            assert lap_c.data["throttle"].max() <= 1.0
            assert lap_c.data["throttle"].min() >= 0.0


# ---------------------------------------------------------------------------
# Segmentation tests
# ---------------------------------------------------------------------------

class TestSegmentation:

    def test_corners_detected(self, segmented_lap):
        result = segmented_lap
        assert len(result.corners) >= 1, "At least 1 corner must be detected"

    def test_corner_distances_monotonic(self, segmented_lap):
        corners = segmented_lap.corners
        for c in corners:
            assert c.entry_dist <= c.apex_dist <= c.exit_dist, \
                f"Corner {c.corner_id}: entry/apex/exit distances not monotonic"

    def test_corner_duration_positive(self, segmented_lap):
        for c in segmented_lap.corners:
            assert c.duration_s > 0, \
                f"Corner {c.corner_id} has non-positive duration"

    def test_phase_column_added(self, segmented_lap):
        df = segmented_lap.tagged_data
        assert "phase" in df.columns
        assert "corner_id" in df.columns

    def test_phase_values_valid(self, segmented_lap):
        df    = segmented_lap.tagged_data
        valid = {"straight", "braking", "corner", "exit"}
        found = set(df["phase"].unique())
        assert found.issubset(valid), f"Invalid phase values: {found - valid}"

    def test_corner_types_valid(self, segmented_lap):
        valid_types = {"slow", "medium", "fast", "kink"}
        for c in segmented_lap.corners:
            assert c.corner_type in valid_types, \
                f"Corner {c.corner_id} has invalid type: {c.corner_type}"

    def test_corners_not_overlapping(self, segmented_lap):
        corners = segmented_lap.corners
        for i in range(len(corners) - 1):
            assert corners[i].exit_dist <= corners[i+1].entry_dist, \
                f"Corner {corners[i].corner_id} overlaps with {corners[i+1].corner_id}"

    def test_result_has_timing_split(self, segmented_lap):
        assert segmented_lap.total_corner_time_s   > 0
        assert segmented_lap.total_straight_time_s > 0


# ---------------------------------------------------------------------------
# Delta tests
# ---------------------------------------------------------------------------

class TestDelta:

    @pytest.fixture(scope="class")
    def comparison(self, loaded_laps):
        from cleaner      import clean
        from segmentation import segment
        from delta        import compare

        lap_a_raw, lap_b_raw = loaded_laps[0], loaded_laps[1]
        lap_a_raw.driver = "VER"; lap_b_raw.driver = "LEC"

        lap_a, _ = clean(lap_a_raw)
        lap_b, _ = clean(lap_b_raw)
        seg_a    = segment(lap_a)
        seg_b    = segment(lap_b)
        return compare(lap_a, lap_b, seg_a, seg_b)

    def test_report_has_drivers(self, comparison):
        assert comparison.driver_a == "VER"
        assert comparison.driver_b == "LEC"

    def test_total_delta_is_float(self, comparison):
        assert isinstance(comparison.total_delta_s, float)

    def test_cum_delta_array_present(self, comparison):
        assert comparison._cum_delta is not None
        assert len(comparison._cum_delta) > 0

    def test_common_dist_array_present(self, comparison):
        assert comparison._common_dist is not None
        assert len(comparison._common_dist) > 0

    def test_summary_string_nonempty(self, comparison):
        s = comparison.summary()
        assert len(s) > 10

    def test_json_serialisable(self, comparison, tmp_path):
        path = comparison.to_json(tmp_path / "test_delta.json")
        assert path.exists()
        import json
        data = json.loads(path.read_text())
        assert "driver_a" in data
        assert "total_delta_s" in data


# ---------------------------------------------------------------------------
# Dashboard test
# ---------------------------------------------------------------------------

class TestDashboard:

    def test_dashboard_generates(self, tmp_path):
        from dashboard import generate
        out = generate(
            analysis_path = "reports/analysis.json",
            delta_path    = "reports/delta_report.json",
            out_path      = str(tmp_path / "index.html"),
        )
        assert out.exists(), "index.html not generated"
        content = out.read_text()
        assert len(content) > 1000, "Dashboard is suspiciously small"

    def test_dashboard_has_tab_switching(self, tmp_path):
        from dashboard import generate
        out = generate(
            analysis_path = "reports/analysis.json",
            delta_path    = "reports/delta_report.json",
            out_path      = str(tmp_path / "index.html"),
        )
        content = out.read_text()
        assert "addEventListener" in content, "Tab switching JS missing"
        assert "data-tab"         in content, "data-tab attributes missing"

    def test_dashboard_has_four_tabs(self, tmp_path):
        from dashboard import generate
        out = generate(
            analysis_path = "reports/analysis.json",
            delta_path    = "reports/delta_report.json",
            out_path      = str(tmp_path / "index.html"),
        )
        content = out.read_text()
        for tab in ["lap", "corners", "delta", "chart"]:
            assert f"data-tab='{tab}'" in content or f'data-tab="{tab}"' in content, \
                f"Tab '{tab}' missing from dashboard"
