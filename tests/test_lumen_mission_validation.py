"""End-to-end scientific/ledger audit tests for LUMEN."""

from dataclasses import replace
import unittest

from core.lumen.energy_storage import BatteryState
from core.lumen.lumen_optics import MirrorOpticsAssumptions
from core.lumen.mission_simulation import MissionSimulationConfig, MissionTimelineSegment, StationPowerAssumptions
from core.lumen.mission_validation import ValidationStatus, validate_lumen_mission
from tests.test_lumen_mission_simulation import config, simulate


class LumenMissionValidationTest(unittest.TestCase):
    def test_normal_end_to_end_reflector_station_and_rover_charging(self) -> None:
        settings = config()
        timeline = tuple(MissionTimelineSegment(
            segment.start_offset_s, segment.end_offset_s, segment.activity,
            segment.rover_consumption_power_w, segment.reflector_target, 20.0,
        ) for segment in settings.mission_timeline)
        settings = MissionSimulationConfig(**(settings.__dict__ | {"mission_timeline": timeline}))
        report = validate_lumen_mission(settings, simulate(settings))
        normal = report.scenario("normal")
        self.assertEqual(normal.visibility_status, "CLEAR")
        self.assertEqual(normal.receiver_capture_status, "VALID")
        self.assertGreater(normal.intercepted_optical_energy_wh, 0.0)
        self.assertGreater(normal.reflected_electrical_energy_wh, 0.0)
        self.assertGreater(normal.station_battery_change_wh, 0.0)
        self.assertGreater(normal.rover_battery_change_wh, 0.0)
        self.assertEqual(normal.ledger_status, ValidationStatus.PASS)

    def test_emergency_recovery_audits_deficit_delivery_and_safe_return(self) -> None:
        result = simulate(config())
        report = validate_lumen_mission(config(), result)
        rescue = report.scenario("rescue")
        self.assertTrue(result.scenario("rescue").rescue_success)
        self.assertGreater(rescue.reflected_electrical_energy_wh, 0.0)
        self.assertGreater(rescue.emergency_energy_deficit_wh, 0.0)
        self.assertGreater(rescue.emergency_energy_delivered_wh, 0.0)
        self.assertGreater(rescue.rover_battery_change_wh, 0.0)
        self.assertEqual(rescue.safe_return_status, ValidationStatus.PASS)
        self.assertEqual(rescue.ledger_status, ValidationStatus.PASS)

    def test_direct_only_baseline_credits_no_reflector_energy(self) -> None:
        result = simulate(config())
        baseline = validate_lumen_mission(config(), result).scenario("direct_only_baseline")
        self.assertEqual(baseline.reflected_electrical_energy_wh, 0.0)
        self.assertEqual(baseline.receiver_capture_status, "UNKNOWN")
        self.assertTrue(any(check.name == "direct-only-isolation" and check.status is ValidationStatus.PASS
                            for check in baseline.checks))

    def test_blocked_sun_mirror_path_receives_no_reflected_power(self) -> None:
        settings = config(blocked_orbit=True)
        rescue = validate_lumen_mission(settings, simulate(settings)).scenario("rescue")
        self.assertEqual(rescue.visibility_status, "BLOCKED")
        self.assertEqual(rescue.reflected_electrical_energy_wh, 0.0)
        self.assertTrue(any(check.name == "visibility-and-capture-gating" and check.status is ValidationStatus.PASS
                            for check in rescue.checks))

    def test_blocked_mirror_to_receiver_result_rejects_credited_power(self) -> None:
        settings = config()
        result = simulate(settings)
        normal = result.scenario("normal")
        step = normal.steps[0]
        self.assertGreater(step.reflected_generation_wh, 0.0)
        blocked_step = replace(step, mirror_station_visibility=False)
        altered_normal = replace(normal, steps=(blocked_step,) + normal.steps[1:])
        altered = replace(result, scenarios=(altered_normal,) + result.scenarios[1:])
        report = validate_lumen_mission(settings, altered).scenario("normal")
        self.assertTrue(any(check.name == "visibility-and-capture-gating" and check.status is ValidationStatus.FAIL
                            for check in report.checks))

    def test_zero_receiver_capture_is_not_credited(self) -> None:
        settings = config()
        station = StationPowerAssumptions(**(settings.station_power.__dict__ | {"reflected_receiver_area_m2": 0.0}))
        settings = MissionSimulationConfig(**(settings.__dict__ | {"station_power": station}))
        normal = validate_lumen_mission(settings, simulate(settings)).scenario("normal")
        self.assertEqual(normal.receiver_capture_status, "ZERO_CAPTURE")
        self.assertEqual(normal.reflected_electrical_energy_wh, 0.0)
        self.assertEqual(normal.ledger_status, ValidationStatus.PASS)

    def test_full_station_battery_curtailed_energy_preserves_balance(self) -> None:
        settings = config()
        station = StationPowerAssumptions(**(
            settings.station_power.__dict__
            | {"direct_array_area_m2": 1_000.0, "initial_battery_state": BatteryState(20_000.0)}
        ))
        settings = MissionSimulationConfig(**(settings.__dict__ | {"station_power": station}))
        result = simulate(settings)
        normal = result.scenario("normal")
        report = validate_lumen_mission(settings, result).scenario("normal")
        self.assertGreater(normal.curtailed_energy_wh, 0.0)
        self.assertEqual(normal.final_station_energy_wh, 20_000.0)
        self.assertAlmostEqual(report.energy_balance_residual_wh, 0.0, delta=1e-5)
        self.assertEqual(report.ledger_status, ValidationStatus.PASS)

    def test_missing_terrain_and_unmodeled_pointing_remain_unknown(self) -> None:
        settings = config(with_profiles=False)
        report = validate_lumen_mission(settings, simulate(settings))
        self.assertEqual(report.scenario("normal").visibility_status, "UNKNOWN")
        self.assertEqual(report.scenario("normal").receiver_capture_status, "UNKNOWN")
        self.assertNotEqual(report.overall_status, ValidationStatus.PASS)
        self.assertTrue(any(check.name == "evidence-completeness" and check.status is ValidationStatus.UNKNOWN
                            for check in report.scenario("normal").checks))

        settings = config()
        result = simulate(settings)
        normal = result.scenario("normal")
        altered_step = replace(normal.steps[0], geometry_frame="")
        altered_normal = replace(normal, steps=(altered_step,) + normal.steps[1:])
        altered_result = replace(result, scenarios=(altered_normal,) + result.scenarios[1:])
        geometry_report = validate_lumen_mission(settings, altered_result)
        self.assertTrue(any(check.name == "geometry-completeness" and check.status is ValidationStatus.UNKNOWN
                            for check in geometry_report.scenario("normal").checks))
        self.assertNotEqual(geometry_report.overall_status, ValidationStatus.PASS)

        settings = config()
        optics = MirrorOpticsAssumptions(**(settings.mirror_optics.__dict__ | {"pointing_error_deg": 1.0}))
        settings = MissionSimulationConfig(**(settings.__dict__ | {"mirror_optics": optics}))
        pointed = validate_lumen_mission(settings, simulate(settings)).scenario("normal")
        self.assertEqual(pointed.receiver_capture_status, "UNKNOWN")
        self.assertEqual(pointed.reflected_electrical_energy_wh, 0.0)
        pointed_report = validate_lumen_mission(settings, simulate(settings))
        self.assertNotEqual(pointed_report.overall_status, ValidationStatus.PASS)
        self.assertTrue(any(check.name == "evidence-completeness" and check.status is ValidationStatus.UNKNOWN
                            for check in pointed_report.scenario("normal").checks))

    def test_report_distinguishes_model_calculation_from_operational_unknown(self) -> None:
        settings = config()
        report = validate_lumen_mission(settings, simulate(settings))
        self.assertEqual(report.operational_feasibility, ValidationStatus.UNKNOWN)
        self.assertTrue(any("not evidence" in item for item in report.limitations))
        normal = report.scenario("normal")
        self.assertTrue(any("assumed kinematic" in item for item in normal.input_assumptions))
        self.assertTrue(normal.provenance)

    def test_independent_ledger_detects_tampered_step_loss(self) -> None:
        settings = config()
        result = simulate(settings)
        normal = result.scenario("normal")
        altered_step = replace(normal.steps[0], conversion_losses_wh=normal.steps[0].conversion_losses_wh + 1.0)
        altered_scenario = replace(normal, steps=(altered_step,) + normal.steps[1:])
        altered_result = replace(result, scenarios=(altered_scenario,) + result.scenarios[1:])
        report = validate_lumen_mission(settings, altered_result)
        self.assertEqual(report.scenario("normal").ledger_status, ValidationStatus.FAIL)
        self.assertEqual(report.overall_status, ValidationStatus.FAIL)


if __name__ == "__main__":
    unittest.main()
