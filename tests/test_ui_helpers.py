"""Tests for pure presentation transformations used by the dashboard."""

from datetime import date, datetime, time, timezone
import ast
from pathlib import Path
import unittest

from core.lunar_coordinates import SITE06_NOBILE_RIM_1
from core.mission_analysis import (
    MissionAnalysisResult,
    MissionSample,
    MissionVisibilityWindow,
    TargetMissionSummary,
)
from ui_helpers import (
    build_geometry_rows,
    build_timeline_rows,
    build_timeline_svg,
    build_utc_interval,
    build_window_rows,
    format_duration,
)


class UIHelpersTest(unittest.TestCase):
    def test_streamlit_entrypoint_is_valid_python_without_importing_optional_ui_packages(self) -> None:
        source = Path(__file__).resolve().parents[1] / "app.py"
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    def test_map_uses_plotly_streamlit_renderer_without_altair_path(self) -> None:
        source = Path(__file__).resolve().parents[1] / "app.py"
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertTrue(any(isinstance(node.func, ast.Attribute) and node.func.attr == "plotly_chart" for node in calls))
        self.assertFalse(any(isinstance(node.func, ast.Attribute) and node.func.attr == "altair_chart" for node in calls))

    def _result(self) -> MissionAnalysisResult:
        first = MissionSample(
            "2026-10-05T00:00:00Z", "SUN", 359.5, 8.0, 2.0, 6.0,
            True, "limited-range", 10.0,
        )
        second = MissionSample(
            "2026-10-05T00:00:00Z", "EARTH", 181.0, -4.0, 1.5, -5.5,
            False, "limited-range", 10.0,
        )
        window = MissionVisibilityWindow(
            "SUN", "2026-10-05T00:00:00Z", "2026-10-05T00:00:00Z", 0, 1,
            "limited-range", 10.0,
        )
        return MissionAnalysisResult(
            SITE06_NOBILE_RIM_1,
            "2026-10-05T00:00:00Z",
            "2026-10-05T00:00:00Z",
            60,
            (first, second),
            (window,),
            (),
            (
                TargetMissionSummary("SUN", 0, 1.0, 0, 1, 1),
                TargetMissionSummary("EARTH", 0, 0.0, 0, 0, 0),
            ),
            "limited-range",
            10.0,
        )

    def test_utc_interval_same_day_and_overnight(self) -> None:
        self.assertEqual(
            build_utc_interval(date(2026, 10, 5), time(8, 30), time(10, 0)),
            ("2026-10-05T08:30:00Z", "2026-10-05T10:00:00Z"),
        )
        self.assertEqual(
            build_utc_interval(date(2026, 10, 5), time(23, 0), time(1, 0)),
            ("2026-10-05T23:00:00Z", "2026-10-06T01:00:00Z"),
        )

    def test_duration_formatting(self) -> None:
        self.assertEqual(format_duration(0), "0 min")
        self.assertEqual(format_duration(75), "1 h 15 min")
        self.assertEqual(format_duration(120), "2 h")
        with self.assertRaises(ValueError):
            format_duration(-1)

    def test_timeline_conversion_preserves_sampled_visible_and_blocked_tracks(self) -> None:
        rows = build_timeline_rows(self._result())
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["track"], "Sun · visible")
        self.assertEqual(rows[1]["track"], "Earth · blocked")
        self.assertEqual(rows[0]["visibility"], "Visible")
        self.assertEqual(rows[1]["visibility"], "Blocked")
        self.assertEqual(rows[0]["timestamp_utc"], datetime(2026, 10, 5, tzinfo=timezone.utc))

    def test_timeline_svg_distinguishes_visible_and_blocked_samples(self) -> None:
        svg = build_timeline_svg(self._result())
        self.assertIn('aria-label="Sampled Sun and Earth visibility timeline"', svg)
        self.assertIn('fill="#16834a"', svg)
        self.assertIn('fill="#bd3434"', svg)
        self.assertIn(">Visible</text>", svg)
        self.assertIn(">Blocked</text>", svg)
        self.assertIn("Sun: visible", svg)
        self.assertIn("Earth: blocked", svg)
        self.assertIn("2026-10-05 00:00 UTC", svg)

    def test_timeline_svg_is_empty_without_samples(self) -> None:
        result = self._result()
        empty = MissionAnalysisResult(
            result.site, result.start_utc, result.end_utc, result.sampling_interval_minutes,
            (), (), (), result.target_summaries, result.horizon_status, result.horizon_range_km,
        )
        self.assertEqual(build_timeline_svg(empty), "")

    def test_geometry_table_keeps_angles_clearance_and_status(self) -> None:
        rows = build_geometry_rows(self._result())
        self.assertEqual(rows[0]["Azimuth (°)"], 359.5)
        self.assertEqual(rows[0]["Geometric elevation (°)"], 8.0)
        self.assertEqual(rows[0]["Terrain horizon elevation (°)"], 2.0)
        self.assertEqual(rows[0]["Clearance (°)"], 6.0)
        self.assertEqual(rows[1]["Status"], "Blocked")
        self.assertIn("UTC timestamp", rows[0])
        self.assertEqual(rows[0]["Target"], "Sun")

    def test_dashboard_calls_out_limited_range_and_metric_card_contrast(self) -> None:
        source = Path(__file__).resolve().parents[1] / "app.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("10 km limited-range terrain analysis", text)
        self.assertIn("visibility beyond this radius is not established", text)
        self.assertIn('div[data-testid="stMetric"] [data-testid="stMetricLabel"]', text)
        self.assertIn("color: #10223d !important", text)

    def test_map_labels_20m_context_and_saved_composite_horizon_separately(self) -> None:
        source = Path(__file__).resolve().parents[1] / "app.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("20 m terrain context", text)
        self.assertIn("The dashed circle marks the 10 km limited-range analysis radius.", text)
        self.assertIn(
            "Sun/Earth visibility uses the saved composite 10 km horizon profile, not the map shading.",
            text,
        )

    def test_window_table_uses_explicit_sample_based_boundary_label(self) -> None:
        rows = build_window_rows(self._result().visibility_windows)
        self.assertEqual(rows[0]["Start (UTC)"], "2026-10-05T00:00:00Z")
        self.assertEqual(rows[0]["Elapsed duration"], "0 min")
        self.assertEqual(rows[0]["Samples"], 1)
        self.assertIn("not exact rise/set", rows[0]["Boundary semantics"])
        self.assertEqual(build_window_rows(()), [])


if __name__ == "__main__":
    unittest.main()
