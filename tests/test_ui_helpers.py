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
    build_daily_summary_rows,
    build_daily_visibility_figure,
    build_site_information_rows,
    build_timeline_rows,
    build_timeline_svg,
    build_utc_interval,
    build_utc_date_range,
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

    def test_daily_comparison_chart_has_fixed_percent_axis_and_dated_markers(self) -> None:
        rows = [
            {"UTC date": "2026-11-18", "Sun sampled visibility": 0.75, "Earth sampled visibility": 0.25},
            {"UTC date": "2026-11-15", "Sun sampled visibility": 0.25, "Earth sampled visibility": 1.0},
            {"UTC date": "2026-11-17", "Sun sampled visibility": None, "Earth sampled visibility": None},
            {"UTC date": "2026-11-16", "Sun sampled visibility": 0.5, "Earth sampled visibility": 0.0},
        ]
        figure = build_daily_visibility_figure(rows)
        sun, earth = figure.data
        self.assertEqual(list(sun.x), list(earth.x))
        self.assertEqual([value.strftime("%Y-%m-%d") for value in sun.x], [
            "2026-11-15", "2026-11-16", "2026-11-17", "2026-11-18",
        ])
        self.assertEqual(list(figure.layout.xaxis.ticktext), ["15 Nov", "16 Nov", "17 Nov", "18 Nov"])
        self.assertEqual(len(figure.layout.xaxis.tickvals), 4)
        self.assertEqual(list(sun.y), [25.0, 50.0, None, 75.0])
        self.assertEqual(list(earth.y), [100.0, 0.0, None, 25.0])
        self.assertEqual(figure.layout.xaxis.type, "date")
        self.assertEqual(list(figure.layout.yaxis.range), [0, 100])
        self.assertEqual(sun.mode, "lines+markers")
        self.assertEqual(earth.mode, "lines+markers")
        self.assertFalse(sun.connectgaps)
        self.assertFalse(earth.connectgaps)
        self.assertEqual(sun.marker.symbol, "circle")
        self.assertEqual(earth.marker.symbol, "diamond")

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

    def test_date_range_uses_utc_bounds_and_rejects_reversed_dates(self) -> None:
        self.assertEqual(
            build_utc_date_range(date(2026, 10, 5), date(2026, 10, 6), time(23), time(1)),
            ("2026-10-05T23:00:00Z", "2026-10-06T01:00:00Z"),
        )
        self.assertEqual(
            build_utc_date_range(date(2026, 10, 5), date(2026, 10, 5), time(23), time(1)),
            ("2026-10-05T23:00:00Z", "2026-10-06T01:00:00Z"),
        )
        with self.assertRaises(ValueError):
            build_utc_date_range(date(2026, 10, 6), date(2026, 10, 5), time(0), time(1))

    def test_daily_summary_groups_by_utc_date_and_preserves_empty_days(self) -> None:
        result = self._result()
        samples = (
            MissionSample("2026-10-05T23:00:00Z", "SUN", 0, 5, 0, 5, True, "limited-range", 10),
            MissionSample("2026-10-05T23:00:00Z", "EARTH", 0, 5, 0, 5, True, "limited-range", 10),
            MissionSample("2026-10-06T00:00:00Z", "SUN", 0, 5, 0, 5, True, "limited-range", 10),
            MissionSample("2026-10-06T00:00:00Z", "EARTH", 0, 5, 0, 5, True, "limited-range", 10),
            MissionSample("2026-10-06T01:00:00Z", "SUN", 0, 5, 0, 5, True, "limited-range", 10),
            MissionSample("2026-10-06T01:00:00Z", "EARTH", 0, -1, 0, -1, False, "limited-range", 10),
            MissionSample("2026-10-08T00:00:00Z", "SUN", 0, 5, 0, 5, True, "limited-range", 10),
            MissionSample("2026-10-08T00:00:00Z", "EARTH", 0, -1, 0, -1, False, "limited-range", 10),
        )
        result = MissionAnalysisResult(
            result.site, "2026-10-05T23:00:00Z", "2026-10-08T00:00:00Z", 60,
            samples, (), (), result.target_summaries, "limited-range", 10,
        )
        rows = build_daily_summary_rows(result, date(2026, 10, 5), date(2026, 10, 8))
        self.assertEqual([row["UTC date"] for row in rows], ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"])
        self.assertEqual(rows[0]["Sun sampled visibility"], 1.0)
        self.assertEqual(rows[1]["Sun sampled visibility"], 1.0)
        self.assertEqual(rows[1]["Earth sampled visibility"], 0.5)
        self.assertEqual(rows[1]["Sun-visible sampled-window duration"], 60)
        self.assertEqual(rows[1]["Earth-visible sampled-window duration"], 0)
        self.assertEqual(rows[1]["Overlapping Sun/Earth duration"], 0)
        self.assertIsNone(rows[2]["Sun sampled visibility"])
        self.assertIsNone(rows[2]["Earth sampled visibility"])
        self.assertEqual(rows[2]["Data availability"], "Unavailable — no samples")
        self.assertIsNone(rows[2]["Sun-visible sampled-window duration"])
        self.assertIsNone(rows[2]["Earth-visible sampled-window duration"])
        self.assertIsNone(rows[2]["Overlapping Sun/Earth duration"])
        self.assertEqual(rows[2]["Sampled timestamps"], 0)
        self.assertEqual(rows[3]["Sun-visible sampled-window duration"], 0)
        self.assertEqual(
            build_utc_interval(date(2026, 10, 5), time(23, 0), time(1, 0)),
            ("2026-10-05T23:00:00Z", "2026-10-06T01:00:00Z"),
        )

    def test_site_information_panel_reports_provenance_scope_and_data_availability(self) -> None:
        rows = build_site_information_rows(self._result())
        info = {row["Information"]: row for row in rows}
        self.assertIn("Site06", info["Analysis support"]["Details"])
        self.assertIn("not evaluated", info["Analysis support"]["Details"])
        self.assertIn("85.43810°S", info["Coordinates"]["Details"])
        self.assertIn("37.36667°E", info["Coordinates"]["Details"])
        self.assertIn("NASA LOLA South Pole site documentation", info["Coordinate provenance"]["Details"])
        self.assertIn("MOON_ME_DE421", info["Sun/Earth geometry"]["Details"])
        self.assertIn("10 km", info["Terrain horizon used for visibility"]["Availability"])
        self.assertIn("20 m DEM is map context only", info["LOLA map context"]["Details"])
        self.assertEqual(info["Sampled observations"]["Availability"], "Available (1 UTC timestamps)")

    def test_site_information_and_daily_durations_mark_empty_samples_unavailable(self) -> None:
        result = self._result()
        empty = MissionAnalysisResult(
            result.site, result.start_utc, result.end_utc, result.sampling_interval_minutes,
            (), (), (), result.target_summaries, result.horizon_status, result.horizon_range_km,
        )
        info = {row["Information"]: row for row in build_site_information_rows(empty)}
        self.assertEqual(info["Sampled observations"]["Availability"], "Unavailable — no samples")
        daily = build_daily_summary_rows(empty, date(2026, 10, 5), date(2026, 10, 5))[0]
        self.assertIsNone(daily["Sun sampled visibility"])
        self.assertIsNone(daily["Sun-visible sampled-window duration"])
        self.assertEqual(daily["Data availability"], "Unavailable — no samples")

    def test_dashboard_renders_site_scope_and_unavailable_empty_metrics(self) -> None:
        source = Path(__file__).resolve().parents[1] / "app.py"
        text = source.read_text(encoding="utf-8")
        self.assertIn("_render_site_assessment(result)", text)
        self.assertIn("sun_visibility = f\"{sun.visible_fraction:.1%}\" if has_samples else \"Unavailable\"", text)
        self.assertIn("build_site_information_rows(result)", text)

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
        self.assertIn("NASA SPACE APPS CHALLENGE 2026", text)
        self.assertIn("visibility at Site06", text)
        self.assertIn("do not establish", text)
        self.assertIn("Site06 only · 10 km limited-range terrain analysis", text)
        self.assertIn("scripts/setup_spice_kernels.py", text)
        self.assertIn("10 km limited-range terrain analysis", text)
        self.assertIn("visibility beyond this radius is not established", text)
        self.assertIn('div[data-testid="stMetric"] [data-testid="stMetricLabel"]', text)
        self.assertIn("color: #10223d !important", text)

    def test_netlify_demo_is_explicitly_static_and_sample_based(self) -> None:
        root = Path(__file__).resolve().parents[1]
        html = (root / "netlify-demo" / "index.html").read_text(encoding="utf-8")
        readme = (root / "netlify-demo" / "README.md").read_text(encoding="utf-8")
        self.assertIn("STATIC FRONTEND DEMO", html)
        self.assertIn("does not run the scientific SPICE analysis", html)
        self.assertIn("does not fetch live NASA data", html)
        self.assertIn("embedded, unverified interface fixture values", html)
        self.assertIn("Not a verified analysis result", html)
        self.assertIn("not the scientific Streamlit application", readme)

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
        self.assertEqual(rows[0]["Sampled-window elapsed duration"], "0 min")
        self.assertEqual(rows[0]["Samples"], 1)
        self.assertIn("not exact rise/set", rows[0]["Boundary semantics"])
        self.assertIn("between-sample visibility unverified", rows[0]["Boundary semantics"])
        self.assertEqual(build_window_rows(()), [])


if __name__ == "__main__":
    unittest.main()
