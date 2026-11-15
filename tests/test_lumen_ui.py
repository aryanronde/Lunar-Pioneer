from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from core.lumen.ui_lumen import (
    DESIGN_ASSUMPTIONS,
    SCENARIO_LABELS,
    STAGE12_REPORT_FILES,
    _feasibility_label,
    _load_stage12_report,
    _modeled_outcome_label,
    _render_stage12_reports,
    build_lumen_dashboard_model,
    _render_overview,
    render_lumen_dashboard,
)


def make_scenario(name, *, rescue_success=None, generation=20.0):
    step = SimpleNamespace(
        sample_utc="2026-01-01T00:05:00Z",
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T00:10:00Z",
        station_battery_energy_wh=10_000.0,
        rover_battery_energy_wh=600.0,
        direct_generation_wh=generation,
        reflected_generation_wh=5.0 if name != "direct_only_baseline" else 0.0,
        rover_energy_delivered_wh=3.0,
        station_load_consumed_wh=2.0,
        rover_energy_consumed_wh=1.5,
        conversion_losses_wh=0.5,
        transfer_losses_wh=0.2,
        rover_operating_state=SimpleNamespace(value="NORMAL"),
        mirror_rover_visibility=None,
        mirror_station_visibility=True,
        emergency_recovery_success=rescue_success,
        spacecraft_position_km=(1_800.0, 20.0, 30.0),
        geometry_frame="MOON_PA_DE440",
        mirror_sun_visibility=True,
        safe_return_feasible=True,
    )
    return SimpleNamespace(
        scenario=name,
        steps=(step,),
        initial_station_energy_wh=9_000.0,
        final_station_energy_wh=10_000.0,
        initial_rover_energy_wh=500.0,
        final_rover_energy_wh=600.0,
        direct_generation_wh=generation,
        reflected_generation_wh=5.0 if name != "direct_only_baseline" else 0.0,
        solar_energy_input_wh=25.0,
        station_load_consumed_wh=2.0,
        rover_energy_consumed_wh=1.5,
        rover_energy_delivered_wh=3.0,
        conversion_losses_wh=0.5,
        curtailed_energy_wh=0.0,
        transfer_losses_wh=0.2,
        energy_balance_residual_wh=0.0,
        unexplained_discrepancy=False,
        rescue_success=rescue_success,
        rescue_failure_reasons=() if rescue_success else ("Insufficient illuminated interval.",),
        assumptions=("circular lunar orbit is assumed kinematic",),
        provenance=("SPICE Sun geometry", "DEM terrain source"),
    )


class FakeSimulation:
    def __init__(self):
        self.results = {
            "normal": make_scenario("normal"),
            "rescue": make_scenario("rescue", rescue_success=False),
            "direct_only_baseline": make_scenario("direct_only_baseline"),
        }
        self.comparison = SimpleNamespace(
            additional_reflected_generation_wh=5.0,
            additional_rover_energy_delivered_wh=1.0,
            additional_station_stored_energy_wh=2.0,
        )
        self.assumptions = ("flat mirror redirects; it does not focus",)
        self.provenance = ("SPICE-derived Sun vector", "assumed circular orbit")

    def scenario(self, name):
        return self.results[name]


class LumenUiTests(unittest.TestCase):
    def test_lumen_overview_renders_empty_state_without_missing_renderer_nameerror(self):
        calls = []

        class StubStreamlit:
            def subheader(self, text):
                calls.append(("subheader", text))

            def info(self, text):
                calls.append(("info", text))

        _render_overview(StubStreamlit(), build_lumen_dashboard_model())
        self.assertEqual(calls[0], ("subheader", "Mission snapshot"))
        self.assertIn("UNKNOWN", calls[1][1])
        self.assertIn("no telemetry", calls[1][1])

    def test_lumen_dashboard_renders_overview_for_empty_session(self):
        class Tab:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class StubStreamlit:
            session_state = {}

            def __getattr__(self, name):
                if name in {"markdown", "subheader", "write", "caption", "dataframe", "warning", "info", "metric"}:
                    return lambda *args, **kwargs: None
                raise AttributeError(name)

            def expander(self, *args, **kwargs):
                return Tab()

            def columns(self, count):
                return tuple(self for _ in range(count))

            def download_button(self, *args, **kwargs):
                return None

            def selectbox(self, label, options, **kwargs):
                return options[0]

            def tabs(self, labels):
                return tuple(Tab() for _ in labels)

            def checkbox(self, *args, **kwargs):
                return False

        with patch.dict("sys.modules", {"streamlit": StubStreamlit()}):
            render_lumen_dashboard()

    def test_lumen_workspace_is_available_without_changing_clps_pipeline(self):
        root = Path(__file__).resolve().parents[1]
        app_text = (root / "app.py").read_text(encoding="utf-8")
        self.assertIn("CLPS Lunar Mission Browser", app_text)
        self.assertIn("LUMEN Station & Reflector", app_text)
        self.assertIn("render_lumen_dashboard()", app_text)
        self.assertIn('st.session_state["mission_analysis_result"]', app_text)
        self.assertIn("_run_analysis(site, start_utc, end_utc, interval)", app_text)

    def test_missing_result_stays_unknown_and_does_not_fabricate_telemetry(self):
        model = build_lumen_dashboard_model()
        self.assertFalse(model["available"])
        self.assertIsNone(model["scenario"])
        self.assertEqual(model["time_series"], [])
        self.assertEqual(model["trajectory"], [])
        self.assertEqual(model["site_rows"][0]["Overall evidence"], "UNKNOWN")
        self.assertFalse(model["site_selected"])

    def test_scenario_selection_and_existing_calculated_energy_values(self):
        simulation = FakeSimulation()
        selected = build_lumen_dashboard_model(simulation, scenario_name="normal")
        self.assertEqual(selected["scenario_label"], SCENARIO_LABELS["normal"])
        self.assertEqual(selected["selected_totals"].final_station_energy_wh, 10_000.0)
        self.assertEqual(selected["time_series"][0]["Direct generation (Wh)"], 20.0)
        self.assertEqual(selected["time_series"][0]["Direct generation (W)"], 120.0)
        self.assertEqual(selected["time_series"][0]["Reflected energy ledger (W)"], 30.0)
        baseline = build_lumen_dashboard_model(simulation, scenario_name="direct_only_baseline")
        self.assertEqual(baseline["time_series"][0]["Reflected energy ledger (Wh)"], 0.0)
        self.assertEqual(baseline["trajectory"][0]["Frame"], "MOON_PA_DE440")
        with self.assertRaises(ValueError):
            build_lumen_dashboard_model(simulation, scenario_name="invented")

    def test_provenance_and_scientific_unknowns_are_visible_in_model(self):
        model = build_lumen_dashboard_model(FakeSimulation(), scenario_name="rescue")
        self.assertIn("SPICE-derived Sun vector", model["provenance"])
        self.assertEqual(model["rescue_success"], False)
        self.assertIn("Insufficient illuminated interval.", model["rescue_failure_reasons"])
        self.assertEqual(model["time_series"][0]["Mirror to rover visibility"], "UNKNOWN")
        self.assertIn("Mirror normal and reflection residual are not included", model["unsupported_optics_fields"][0])
        self.assertIn("Beam footprint", model["unsupported_optics_fields"][1])

    def test_design_assumptions_have_explicit_units_and_provenance(self):
        values = {(label, unit): (value, provenance) for label, value, unit, provenance in DESIGN_ASSUMPTIONS}
        self.assertEqual(values[("Station nominal battery", "Wh")][0], 20_000)
        self.assertEqual(values[("Protected station reserve", "Wh")][0], 5_000)
        self.assertEqual(values[("Rover nominal battery", "Wh")][0], 1_000)
        self.assertEqual(values[("Charging efficiency", "fraction")][0], 0.90)
        self.assertEqual(values[("Station average load", "W")][0], 200)
        self.assertTrue(all(provenance for _, provenance in values.values()))

    def test_all_three_scenarios_are_exposed_for_comparison(self):
        model = build_lumen_dashboard_model(FakeSimulation())
        self.assertEqual(set(SCENARIO_LABELS), {"normal", "rescue", "direct_only_baseline"})
        self.assertEqual(model["normal"].scenario, "normal")
        self.assertEqual(model["rescue"].scenario, "rescue")
        self.assertEqual(model["baseline"].scenario, "direct_only_baseline")
        self.assertEqual(model["comparison"].additional_reflected_generation_wh, 5.0)

    def test_stage12_report_viewer_loads_valid_report_statuses(self):
        with tempfile.TemporaryDirectory() as temp:
            report_dir = Path(temp)
            payload = {
                "scenario_name": "normal",
                "scenario_audit_status": "PASS",
                "visibility_status": "CLEAR",
                "receiver_capture_status": "VALID",
                "operational_feasibility": "UNKNOWN",
                "calculated": {
                    "electrical_energy_Wh": {
                        "direct_generation": 12.5,
                        "receiver_reflected_generation": 4.25,
                    },
                    "emergency_recovery": {"scenario_recovery_success": None},
                    "energy_balance_residual_Wh": 0.0,
                },
            }
            (report_dir / STAGE12_REPORT_FILES["normal"]).write_text(json.dumps(payload), encoding="utf-8")
            report = _load_stage12_report("normal", report_dir)
            self.assertTrue(report["available"])
            self.assertTrue(report["valid_json"])
            self.assertEqual(report["payload"]["scenario_audit_status"], "PASS")
            self.assertEqual(report["payload"]["operational_feasibility"], "UNKNOWN")

    def test_stage12_report_missing_and_malformed_are_handled(self):
        with tempfile.TemporaryDirectory() as temp:
            report_dir = Path(temp)
            missing = _load_stage12_report("normal", report_dir)
            self.assertFalse(missing["available"])
            self.assertIsNone(missing["raw_bytes"])
            (report_dir / STAGE12_REPORT_FILES["normal"]).write_text("{broken", encoding="utf-8")
            malformed = _load_stage12_report("normal", report_dir)
            self.assertTrue(malformed["available"])
            self.assertFalse(malformed["valid_json"])
            self.assertIsNone(malformed["payload"])
            self.assertEqual(malformed["raw_bytes"], b"{broken")

    def test_stage12_viewer_keeps_missing_fields_unknown_and_downloads_original_bytes(self):
        class Expander:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class StubColumn:
            def __init__(self, parent):
                self.parent = parent

            def metric(self, label, value, **kwargs):
                self.parent.metrics.append((label, value))

        class StubStreamlit:
            def __init__(self):
                self.metrics = []
                self.downloads = []
                self.messages = []

            def expander(self, *args, **kwargs):
                return Expander()

            def caption(self, value):
                self.messages.append(("caption", value))

            def selectbox(self, label, options, **kwargs):
                return "normal"

            def columns(self, count):
                return tuple(StubColumn(self) for _ in range(count))

            def info(self, value):
                self.messages.append(("info", value))

            def warning(self, value):
                self.messages.append(("warning", value))

            def error(self, value):
                self.messages.append(("error", value))

            def markdown(self, value):
                self.messages.append(("markdown", value))

            def download_button(self, label, **kwargs):
                self.downloads.append((label, kwargs["data"], kwargs["file_name"]))

        with tempfile.TemporaryDirectory() as temp:
            report_dir = Path(temp)
            expected_bytes = {}
            for name, filename in STAGE12_REPORT_FILES.items():
                data = {"scenario_name": name, "scenario_audit_status": "PASS", "operational_feasibility": "UNKNOWN"} if name == "normal" else {"scenario_name": name}
                raw = json.dumps(data).encode("utf-8")
                (report_dir / filename).write_bytes(raw)
                expected_bytes[filename] = raw
            stub = StubStreamlit()
            _render_stage12_reports(stub, report_dir)
            shown = dict(stub.metrics)
            self.assertEqual(shown["Scenario audit"], "PASS (MODEL AUDIT)")
            self.assertEqual(shown["Operational feasibility"], "UNKNOWN")
            self.assertEqual(shown["Visibility"], "UNKNOWN")
            self.assertEqual(shown["Receiver capture"], "UNKNOWN")
            self.assertEqual(shown["Direct electrical energy"], "UNKNOWN")
            self.assertEqual(shown["Reflected receiver electrical energy"], "UNKNOWN")
            self.assertEqual(shown["Emergency recovery"], "UNKNOWN")
            self.assertEqual(shown["Energy-balance residual"], "UNKNOWN")
            self.assertEqual(len(stub.downloads), 3)
            for _label, raw, filename in stub.downloads:
                self.assertEqual(raw, expected_bytes[filename])

    def test_stage12_viewer_handles_missing_and_malformed_files_without_crashing(self):
        class Expander:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        class StubStreamlit:
            def __init__(self):
                self.messages = []
                self.downloads = []

            def expander(self, *args, **kwargs):
                return Expander()

            def caption(self, value):
                self.messages.append(("caption", value))

            def selectbox(self, label, options, **kwargs):
                return "normal"

            def columns(self, count):
                return tuple(SimpleNamespace(metric=lambda *args, **kwargs: None) for _ in range(count))

            def warning(self, value):
                self.messages.append(("warning", value))

            def error(self, value):
                self.messages.append(("error", value))

            def info(self, value):
                self.messages.append(("info", value))

            def markdown(self, value):
                self.messages.append(("markdown", value))

            def download_button(self, label, **kwargs):
                self.downloads.append(kwargs["file_name"])

        with tempfile.TemporaryDirectory() as temp:
            report_dir = Path(temp)
            (report_dir / STAGE12_REPORT_FILES["normal"]).write_text("not json", encoding="utf-8")
            stub = StubStreamlit()
            _render_stage12_reports(stub, report_dir)
            self.assertTrue(any(level == "error" for level, _ in stub.messages))
            self.assertTrue(any("unavailable for download" in message for level, message in stub.messages if level == "caption"))
            self.assertEqual(stub.downloads, [STAGE12_REPORT_FILES["normal"]])

    def test_recovery_and_safe_return_outcomes_use_non_visibility_language(self):
        self.assertEqual(_modeled_outcome_label(True), "MODELED SUCCESS")
        self.assertEqual(_modeled_outcome_label(False), "INFEASIBLE")
        self.assertEqual(_modeled_outcome_label(None), "UNKNOWN")
        self.assertEqual(_feasibility_label(True), "FEASIBLE (MODELED)")
        self.assertEqual(_feasibility_label(False), "INFEASIBLE")
        self.assertEqual(_feasibility_label(None), "UNKNOWN")
        simulation = FakeSimulation()
        model = build_lumen_dashboard_model(simulation, scenario_name="rescue")
        self.assertEqual(model["time_series"][0]["Emergency recovery outcome"], "INFEASIBLE")
        self.assertNotIn(model["time_series"][0]["Emergency recovery outcome"], {"VISIBLE", "BLOCKED"})


if __name__ == "__main__":
    unittest.main()
