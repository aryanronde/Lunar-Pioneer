"""Tests for explicit-window LUMEN orbital-reflector rover recovery estimates."""

from datetime import datetime, timedelta, timezone
from dataclasses import replace
import math
import unittest

from core.lumen.energy_storage import BatteryConfig
from core.lumen.emergency_recovery import (
    RecoveryIlluminationInterval,
    RecoveryRequest,
    assess_emergency_recovery,
)
from core.lumen.lumen_optics import (
    MirrorOpticsAssumptions,
    TaggedLunarPosition,
    calculate_mirror_optics,
)
from core.lumen.lumen_receiver import ReceiverAssumptions, calculate_receiver_power
from core.lumen.lumen_visibility import (
    SiteHorizonProfile,
    evaluate_lumen_visibility,
)
from core.lumen.rover_energy import RoverConfig


RADIUS_KM = 1_737.4
START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def stamp(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def position(vector) -> TaggedLunarPosition:
    return TaggedLunarPosition(tuple(vector), "J2000")


def make_case(*, blocked=False, receiver_area_m2=100.0, panel_efficiency=0.3, pointing_error_deg=0.0):
    sun = position((150_000_000.0, 0.0, 0.0))
    mirror = position((-3_000.0, 0.0, 0.0) if blocked else (3_000.0, 0.0, 0.0))
    station = position((RADIUS_KM, 0.0, 0.0))
    rover = position((RADIUS_KM, 100.0, 0.0))
    visibility = evaluate_lumen_visibility(
        sun,
        mirror,
        station,
        rover,
        lunar_reference_radius_km=RADIUS_KM,
        rover_horizon=SiteHorizonProfile(
            profile_deg=tuple([0.0] * 360),
            site_position_km=rover.position_km,
            frame="J2000",
            east_axis=(0.0, 0.0, 1.0),
            north_axis=(0.0, 1.0, 0.0),
            up_axis=(1.0, 0.0, 0.0),
            range_km=20_000,
            status="complete",
            site_name="synthetic test horizon",
        ),
    )
    optics = calculate_mirror_optics(
        sun,
        mirror,
        rover,
        MirrorOpticsAssumptions(
            mirror_area_m2=1_000_000.0,
            reflectivity=0.9,
            pointing_error_deg=pointing_error_deg,
            solar_irradiance_w_m2=1361.0,
        ),
        geometry_origin="assumed",
    )
    return visibility, optics, receiver_area_m2, panel_efficiency


def make_interval(
    *,
    start=START,
    end=None,
    illumination=True,
    blocked=False,
    receiver_area_m2=100.0,
    panel_efficiency=0.3,
    receiver_angle_deg=0.0,
    receiver_offset_u_m=0.0,
    pointing_error_deg=0.0,
):
    if end is None:
        end = start + timedelta(hours=1)
    visibility, optics, area, efficiency = make_case(
        blocked=blocked,
        receiver_area_m2=receiver_area_m2,
        panel_efficiency=panel_efficiency,
        pointing_error_deg=pointing_error_deg,
    )
    toward_mirror = tuple(-value for value in optics.mirror_to_target_unit)
    axis = (1.0, 0.0, 0.0) if abs(toward_mirror[0]) < 0.9 else (0.0, 1.0, 0.0)
    projection = sum(axis[i] * toward_mirror[i] for i in range(3))
    tangent = tuple(axis[i] - projection * toward_mirror[i] for i in range(3))
    tangent_norm = sum(value * value for value in tangent) ** 0.5
    tangent = tuple(value / tangent_norm for value in tangent)
    angle = math.radians(receiver_angle_deg)
    normal = tuple(math.cos(angle) * toward_mirror[i] + math.sin(angle) * tangent[i] for i in range(3))
    receiver_power = calculate_receiver_power(optics, ReceiverAssumptions(
        physical_area_m2=area,
        surface_normal_unit=normal,
        frame=optics.frame,
        photovoltaic_efficiency=efficiency,
        center_offset_u_m=receiver_offset_u_m,
    ))
    return RecoveryIlluminationInterval(
        start_utc=stamp(start),
        end_utc=stamp(end),
        illumination_available=illumination,
        visibility=visibility,
        optics=optics,
        receiver_area_m2=area,
        receiver_incidence_angle_deg=receiver_angle_deg,
        receiver_conversion_efficiency=efficiency,
        receiver_power=receiver_power,
    )


def request(intervals, **overrides):
    values = dict(
        current_battery_energy_wh=200.0,
        safe_return_energy_wh=10.0,
        survival_start_utc=stamp(START),
        survival_deadline_utc=stamp(START + timedelta(hours=1)),
        rover_config=RoverConfig(),
        intervals=tuple(intervals),
    )
    values.update(overrides)
    return RecoveryRequest(**values)


class EmergencyRecoveryTest(unittest.TestCase):
    def test_successful_recovery_meets_safe_return_energy_before_deadline(self) -> None:
        result = assess_emergency_recovery(request([make_interval()]))
        self.assertTrue(result.feasible, result.failure_reasons)
        self.assertEqual(result.energy_deficit_wh, 10.0)
        self.assertGreaterEqual(result.final_usable_energy_wh, result.safe_return_energy_wh)
        self.assertGreaterEqual(result.delivered_battery_energy_wh, result.energy_deficit_wh)
        self.assertEqual(result.earliest_feasible_start_utc, stamp(START))
        self.assertEqual(result.latest_feasible_end_utc, stamp(START + timedelta(hours=1)))
        self.assertGreater(result.receiver_optical_energy_wh, result.receiver_electrical_energy_wh)
        self.assertGreater(result.receiver_electrical_energy_wh, result.delivered_battery_energy_wh)
        self.assertIn("flat mirror redirects but does not focus sunlight", result.assumptions)

    def test_explicit_energy_deficit_and_safe_return_threshold(self) -> None:
        result = assess_emergency_recovery(request([]))
        self.assertEqual(result.current_battery_energy_wh, 200)
        self.assertEqual(result.current_usable_energy_wh, 0)
        self.assertEqual(result.safe_return_energy_wh, 10)
        self.assertEqual(result.energy_deficit_wh, 10)
        self.assertFalse(result.feasible)
        self.assertEqual(result.delivered_battery_energy_wh, 0)
        self.assertLess(result.final_safe_return_margin_wh, 0)

    def test_insufficient_illumination_window_rejected(self) -> None:
        interval = make_interval(end=START + timedelta(minutes=1))
        result = assess_emergency_recovery(request(
            [interval], survival_deadline_utc=stamp(START + timedelta(hours=1))
        ))
        self.assertFalse(result.feasible)
        self.assertLess(result.delivered_battery_energy_wh, result.energy_deficit_wh)
        self.assertTrue(any("shorter" in reason or "insufficient" in reason or "below" in reason
                            for reason in result.failure_reasons))

    def test_blocked_line_of_sight_rejected(self) -> None:
        result = assess_emergency_recovery(request([make_interval(blocked=True)]))
        self.assertFalse(result.feasible)
        self.assertEqual(result.delivered_battery_energy_wh, 0)
        self.assertTrue(any("blocked" in reason for reason in result.failure_reasons))

    def test_clear_geometry_alone_does_not_claim_recovery(self) -> None:
        interval = make_interval(illumination=False)
        self.assertTrue(interval.visibility.sun_to_mirror.visible)
        self.assertTrue(interval.visibility.mirror_to_rover.visible)
        result = assess_emergency_recovery(request([interval]))
        self.assertFalse(result.feasible)
        self.assertEqual(result.delivered_battery_energy_wh, 0)
        self.assertTrue(any("unavailable" in reason for reason in result.failure_reasons))

    def test_survival_deadline_clips_intervals_and_limits_energy(self) -> None:
        interval = make_interval()
        deadline = START + timedelta(seconds=30)
        result = assess_emergency_recovery(request([interval], survival_deadline_utc=stamp(deadline)))
        self.assertFalse(result.feasible)
        self.assertEqual(result.feasible_intervals[0].duration_s, 30)
        self.assertEqual(result.feasible_intervals[0].end_utc, stamp(deadline))
        self.assertLess(result.delivered_battery_energy_wh, result.energy_deficit_wh)

    def test_charging_power_limit_applies_to_each_available_interval(self) -> None:
        config = RoverConfig(
            battery_config=BatteryConfig(
                nominal_capacity_wh=1_000,
                maximum_depth_of_discharge=0.8,
                protected_reserve_wh=200,
                maximum_charge_power_w=1.0,
            )
        )
        result = assess_emergency_recovery(request([make_interval()], rover_config=config))
        self.assertFalse(result.feasible)
        self.assertAlmostEqual(result.delivered_battery_energy_wh, 0.9)
        self.assertGreater(result.charge_power_or_capacity_curtailed_wh, 0)
        self.assertTrue(any("insufficient" in reason for reason in result.failure_reasons))

    def test_battery_capacity_limit_reports_curtailed_receiver_energy(self) -> None:
        result = assess_emergency_recovery(request(
            [make_interval()],
            current_battery_energy_wh=990,
            safe_return_energy_wh=900,
        ))
        self.assertFalse(result.feasible)
        self.assertAlmostEqual(result.delivered_battery_energy_wh, 10.0)
        self.assertGreater(result.charge_power_or_capacity_curtailed_wh, 0.0)
        self.assertEqual(result.final_battery_energy_wh, 1_000.0)

    def test_recovery_uses_same_receiver_result_as_stage32_api(self):
        interval = make_interval(receiver_area_m2=100.0)
        expected = interval.receiver_power
        result = assess_emergency_recovery(request([interval]))
        self.assertAlmostEqual(result.receiver_intercepted_optical_energy_wh,
            expected.intercepted_optical_power_w, places=8)
        self.assertAlmostEqual(result.receiver_optical_energy_wh,
            expected.received_optical_power_w, places=8)
        self.assertAlmostEqual(result.receiver_electrical_energy_wh,
            expected.electrical_power_w, places=8)

    def test_recovery_recomputes_with_same_receiver_api_when_result_is_absent(self):
        supplied = make_interval(receiver_area_m2=100.0)
        omitted = replace(supplied, receiver_power=None)
        with_result = assess_emergency_recovery(request([supplied]))
        recomputed = assess_emergency_recovery(request([omitted]))
        self.assertAlmostEqual(recomputed.receiver_intercepted_optical_energy_wh,
            with_result.receiver_intercepted_optical_energy_wh, places=10)
        self.assertAlmostEqual(recomputed.receiver_electrical_energy_wh,
            with_result.receiver_electrical_energy_wh, places=10)
        self.assertAlmostEqual(recomputed.delivered_battery_energy_wh,
            with_result.delivered_battery_energy_wh, places=10)

    def test_small_partial_and_zero_overlap_use_shared_capture_model(self):
        partial = make_interval(receiver_area_m2=100.0)
        self.assertGreater(partial.receiver_power.capture_fraction, 0.0)
        self.assertLess(partial.receiver_power.capture_fraction, 1.0)
        zero = make_interval(receiver_area_m2=100.0, receiver_offset_u_m=20_000.0)
        self.assertEqual(zero.receiver_power.capture_fraction, 0.0)
        result = assess_emergency_recovery(request([zero]))
        self.assertEqual(result.receiver_electrical_energy_wh, 0.0)
        self.assertEqual(result.delivered_battery_energy_wh, 0.0)

    def test_pointing_error_is_applied_to_shared_capture_model(self):
        interval = make_interval(receiver_area_m2=100.0, pointing_error_deg=0.5)
        self.assertEqual(interval.receiver_power.capture_fraction, 0.0)
        result = assess_emergency_recovery(request([interval]))
        self.assertEqual(result.receiver_electrical_energy_wh, 0.0)
        self.assertFalse(result.feasible)

    def test_safe_return_already_feasible_requires_no_recovery(self) -> None:
        result = assess_emergency_recovery(request(
            [], current_battery_energy_wh=400, safe_return_energy_wh=100
        ))
        self.assertTrue(result.feasible)
        self.assertEqual(result.energy_deficit_wh, 0)
        self.assertEqual(result.delivered_battery_energy_wh, 0)

    def test_energy_losses_and_storage_balance(self) -> None:
        result = assess_emergency_recovery(request([make_interval()]))
        self.assertAlmostEqual(
            result.receiver_electrical_energy_wh,
            result.delivered_battery_energy_wh
            + result.battery_charging_loss_wh
            + result.charge_power_or_capacity_curtailed_wh,
            places=8,
        )
        self.assertAlmostEqual(
            result.receiver_optical_energy_wh,
            result.receiver_electrical_energy_wh + result.receiver_conversion_loss_wh,
            places=8,
        )

    def test_invalid_request_and_interval_inputs(self) -> None:
        interval = make_interval()
        with self.assertRaises(ValueError):
            make_interval(receiver_area_m2=-1)
        with self.assertRaises(ValueError):
            make_interval(receiver_angle_deg=91)
        with self.assertRaises(ValueError):
            make_interval(panel_efficiency=0)
        with self.assertRaises(ValueError):
            RecoveryRequest(1001, 10, stamp(START), stamp(START + timedelta(hours=1)), RoverConfig(), (interval,))
        with self.assertRaises(ValueError):
            RecoveryRequest(200, -1, stamp(START), stamp(START + timedelta(hours=1)), RoverConfig(), ())
        with self.assertRaises(ValueError):
            RecoveryRequest(200, 1, stamp(START), stamp(START), RoverConfig(), ())
        with self.assertRaises(ValueError):
            RecoveryRequest(
                200, 10, stamp(START), stamp(START + timedelta(hours=1)), RoverConfig(),
                (interval, interval),
            )


if __name__ == "__main__":
    unittest.main()
