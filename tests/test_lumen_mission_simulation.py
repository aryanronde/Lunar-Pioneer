"""Integrated LUMEN mission simulation tests using deterministic synthetic SPICE."""

from datetime import datetime, timezone
import unittest

from core.lumen.energy_storage import BatteryConfig, BatteryState
from core.lumen.lumen_optics import MirrorOpticsAssumptions, TaggedLunarPosition
from core.lumen.lumen_orbit import CircularLunarOrbit
from core.lumen.lumen_visibility import SiteHorizonProfile
from core.lumen.mission_simulation import (
    MissionSimulationConfig,
    MissionTimelineSegment,
    RoverSolarReceiverAssumptions,
    StationPowerAssumptions,
    simulate_lumen_mission,
)
from core.lumen.rover_energy import MissionEnergyRequirements, RoverConfig


START = "2026-01-01T00:00:00Z"
R = 1_737.4


class FakeGeometry:
    """Minimal deterministic SPICE-compatible geometry adapter for unit tests."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def utc_to_et(self, timestamp):
        value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        return value.replace(tzinfo=timezone.utc).timestamp()

    def sun_position_km(self, et):
        return (150_000_000.0, 0.0, 0.0)

    def j2000_to_moon_fixed(self, vector, et):
        return tuple(vector)


class FakeBridge:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return None

    def vector_dem_to_geometry(self, vector, et):
        return tuple(vector)


def site_profile(site, *, range_km=1000.0, status="complete"):
    return SiteHorizonProfile(
        profile_deg=tuple([0.0] * 360),
        site_position_km=site.position_km,
        frame=site.frame,
        east_axis=(0.0, 1.0, 0.0),
        north_axis=(0.0, 0.0, 1.0),
        up_axis=(1.0, 0.0, 0.0),
        range_km=range_km,
        status=status,
        site_name="synthetic complete test terrain",
    )


def config(*, blocked_orbit=False, with_profiles=True, step_s=60.0):
    station = TaggedLunarPosition((R, 0.0, 0.0), "MOON_PA_DE440")
    rover = TaggedLunarPosition((R, 10.0, 0.0), "MOON_PA_DE440")
    station_profile = site_profile(station) if with_profiles else None
    rover_profile = site_profile(rover) if with_profiles else None
    station_assumptions = StationPowerAssumptions(
        solar_irradiance_1au_w_m2=1361.0,
        direct_array_area_m2=10.0,
        direct_conversion_efficiency=0.2,
        direct_incidence_angle_deg=0.0,
        reflected_receiver_area_m2=100.0,
        reflected_conversion_efficiency=0.2,
        reflected_incidence_angle_deg=0.0,
        average_load_w=200.0,
        battery_config=BatteryConfig(
            20_000.0, 0.8, 5_000.0,
            maximum_charge_power_w=5_000.0,
            maximum_discharge_power_w=5_000.0,
        ),
        initial_battery_state=BatteryState(10_000.0),
        battery_charge_efficiency=0.9,
        rover_transfer_power_w=100.0,
        rover_transfer_efficiency=0.95,
    )
    rover_config = RoverConfig(
        battery_config=BatteryConfig(
            1_000.0, 0.8, 200.0,
            maximum_charge_power_w=50_000.0,
            maximum_discharge_power_w=1_000.0,
        ),
        charging_efficiency=0.9,
        return_required_margin_wh=20.0,
    )
    energy_fields = (
        "outbound_travel_energy_wh",
        "outbound_communication_navigation_energy_wh",
        "outbound_thermal_energy_wh",
        "return_travel_energy_wh",
        "return_communication_navigation_energy_wh",
        "return_thermal_survival_energy_wh",
        "uncertainty_reserve_energy_wh",
    )
    requirements = MissionEnergyRequirements(
        outbound_travel_energy_wh=0,
        outbound_communication_navigation_energy_wh=0,
        outbound_thermal_energy_wh=0,
        return_travel_energy_wh=50,
        return_communication_navigation_energy_wh=10,
        return_thermal_survival_energy_wh=20,
        uncertainty_reserve_energy_wh=20,
        energy_origins=tuple((name, "assumed") for name in energy_fields),
    )
    phase = 180.0 if blocked_orbit else 0.0
    return MissionSimulationConfig(
        start_utc=START,
        duration_s=120.0,
        time_step_s=step_s,
        orbit=CircularLunarOrbit(
            altitude_km=100.0,
            inclination_deg=0.0,
            ascending_node_longitude_deg=0.0,
            initial_orbital_phase_deg=phase,
            epoch_utc=START,
        ),
        station_position=station,
        rover_position=rover,
        station_horizon=station_profile,
        rover_horizon=rover_profile,
        station_power=station_assumptions,
        rover_config=rover_config,
        initial_rover_battery_state=BatteryState(210.0 if blocked_orbit else 300.0),
        rover_energy_requirements=requirements,
        rover_receiver=RoverSolarReceiverAssumptions(
            receiver_area_m2=100_000.0,
            incidence_angle_deg=0.0,
            conversion_efficiency=0.3,
        ),
        mirror_optics=MirrorOpticsAssumptions(
            mirror_area_m2=1_000_000.0,
            reflectivity=0.9,
            pointing_error_deg=0.0,
            solar_irradiance_w_m2=1361.0,
        ),
        mission_timeline=(
            MissionTimelineSegment(0.0, 60.0, "explore", 10.0, "station", 0.0),
            MissionTimelineSegment(60.0, 120.0, "return", 10.0, "station", 0.0),
        ),
        stranded_at_offset_s=60.0,
        stranded_survival_power_w=0.1,
        rescue_window_s=60.0,
        lunar_reference_radius_km=R,
        flag_energy_residual_above_wh=1e-5,
    )


def simulate(settings):
    return simulate_lumen_mission(
        settings,
        geometry_engine_factory=FakeGeometry,
        frame_bridge_factory=FakeBridge,
    )


class LumenMissionSimulationTest(unittest.TestCase):
    def test_deterministic_repeatability_and_time_step_behaviour(self) -> None:
        first = simulate(config())
        second = simulate(config())
        self.assertEqual(first, second)
        self.assertEqual(len(first.scenario("normal").steps), 2)
        finer = simulate(config(step_s=50.0))
        self.assertEqual(len(finer.scenario("normal").steps), 3)
        self.assertTrue(finer.scenario("normal").steps[-1].end_utc.endswith("00:02:00Z"))

    def test_all_three_reproducible_scenarios_and_matched_baseline(self) -> None:
        result = simulate(config())
        self.assertEqual([run.scenario for run in result.scenarios], [
            "normal", "rescue", "direct_only_baseline"
        ])
        normal = result.scenario("normal")
        baseline = result.scenario("direct_only_baseline")
        self.assertEqual(normal.initial_station_energy_wh, baseline.initial_station_energy_wh)
        self.assertEqual(normal.initial_rover_energy_wh, baseline.initial_rover_energy_wh)
        self.assertAlmostEqual(normal.direct_generation_wh, baseline.direct_generation_wh)
        self.assertEqual(baseline.reflected_generation_wh, 0)
        self.assertGreater(normal.reflected_generation_wh, 0)
        self.assertGreater(result.comparison.additional_reflected_generation_wh, 0)

    def test_energy_balance_capacity_and_station_reserve(self) -> None:
        result = simulate(config())
        for scenario in result.scenarios:
            self.assertLessEqual(scenario.final_station_energy_wh, 20_000)
            self.assertGreaterEqual(scenario.final_station_energy_wh, 5_000)
            self.assertLessEqual(scenario.final_rover_energy_wh, 1_000)
            self.assertGreaterEqual(scenario.final_rover_energy_wh, 200)
            self.assertAlmostEqual(scenario.energy_balance_residual_wh, 0.0, delta=1e-5)
            self.assertFalse(scenario.unexplained_discrepancy)

    def test_direct_and_reflected_generation_are_separate(self) -> None:
        normal = simulate(config()).scenario("normal")
        baseline = simulate(config()).scenario("direct_only_baseline")
        for step in normal.steps:
            self.assertGreater(step.direct_generation_wh, 0)
            self.assertGreaterEqual(step.reflected_generation_wh, 0)
        self.assertGreater(normal.reflected_generation_wh, 0)
        self.assertEqual(baseline.reflected_generation_wh, 0)

    def test_receiver_interception_electrical_output_and_storage_are_distinct(self) -> None:
        result = simulate(config())
        normal = result.scenario("normal")
        baseline = result.scenario("direct_only_baseline")
        self.assertGreater(normal.receiver_intercepted_optical_energy_wh, 0.0)
        self.assertGreater(normal.receiver_electrical_generation_wh, 0.0)
        self.assertLess(normal.receiver_electrical_generation_wh, normal.receiver_intercepted_optical_energy_wh)
        self.assertAlmostEqual(normal.receiver_electrical_generation_wh, normal.reflected_generation_wh, places=10)
        self.assertEqual(baseline.receiver_intercepted_optical_energy_wh, 0.0)
        self.assertEqual(baseline.receiver_electrical_generation_wh, 0.0)
        # Generation and stored energy are separate ledger terms; no storage
        # quantity is added to reflected_generation_wh a second time.
        self.assertEqual(normal.receiver_battery_stored_energy_wh, 0.0)
        for scenario in result.scenarios:
            self.assertAlmostEqual(scenario.energy_balance_residual_wh, 0.0, delta=1e-5)

    def test_zero_station_receiver_capture_produces_no_reflected_generation(self) -> None:
        settings = config()
        station_power = StationPowerAssumptions(
            **(settings.station_power.__dict__ | {"reflected_receiver_area_m2": 0.0})
        )
        settings = MissionSimulationConfig(**(settings.__dict__ | {"station_power": station_power}))
        normal = simulate(settings).scenario("normal")
        self.assertEqual(normal.reflected_generation_wh, 0.0)
        self.assertEqual(normal.receiver_electrical_generation_wh, 0.0)
        self.assertTrue(all(step.receiver_capture_status == "ZERO_CAPTURE" for step in normal.steps))
        self.assertAlmostEqual(normal.energy_balance_residual_wh, 0.0, delta=1e-5)

    def test_blocked_receiver_path_is_not_credited(self) -> None:
        rescue = simulate(config(blocked_orbit=True)).scenario("rescue")
        self.assertEqual(rescue.receiver_electrical_generation_wh, 0.0)
        self.assertTrue(all(step.receiver_capture_status in {"BLOCKED", "UNKNOWN"} for step in rescue.steps))

    def test_rover_receiver_charging_respects_battery_capacity(self) -> None:
        settings = config()
        settings = MissionSimulationConfig(
            **(settings.__dict__ | {"initial_rover_battery_state": BatteryState(999.9)})
        )
        rescue = simulate(settings).scenario("rescue")
        self.assertLessEqual(rescue.final_rover_energy_wh, 1_000.0)
        self.assertGreaterEqual(rescue.receiver_electrical_generation_wh, rescue.receiver_battery_stored_energy_wh)
        self.assertGreaterEqual(rescue.receiver_battery_stored_energy_wh, 0.0)
        self.assertAlmostEqual(rescue.energy_balance_residual_wh, 0.0, delta=1e-5)

    def test_scenarios_keep_receiver_generation_separate(self) -> None:
        result = simulate(config())
        normal = result.scenario("normal")
        rescue = result.scenario("rescue")
        baseline = result.scenario("direct_only_baseline")
        self.assertGreater(normal.receiver_electrical_generation_wh, 0.0)
        self.assertGreater(rescue.receiver_electrical_generation_wh, 0.0)
        self.assertEqual(baseline.receiver_electrical_generation_wh, 0.0)
        self.assertEqual(baseline.reflected_generation_wh, 0.0)

    def test_station_to_rover_transfer_is_accounted_in_matched_baseline(self) -> None:
        settings = config()
        timeline = tuple(
            MissionTimelineSegment(
                part.start_offset_s,
                part.end_offset_s,
                part.activity,
                part.rover_consumption_power_w,
                part.reflector_target,
                20.0,
            )
            for part in settings.mission_timeline
        )
        settings = MissionSimulationConfig(**(settings.__dict__ | {"mission_timeline": timeline}))
        result = simulate(settings)
        baseline = result.scenario("direct_only_baseline")
        self.assertGreater(baseline.rover_energy_delivered_wh, 0.0)
        self.assertGreater(baseline.transfer_losses_wh, 0.0)
        self.assertAlmostEqual(baseline.energy_balance_residual_wh, 0.0, delta=1e-5)

    def test_blocked_sun_to_mirror_makes_rescue_infeasible(self) -> None:
        result = simulate(config(blocked_orbit=True))
        rescue = result.scenario("rescue")
        self.assertFalse(rescue.rescue_success)
        self.assertTrue(any("blocked" in reason.lower() for reason in rescue.rescue_failure_reasons))
        self.assertEqual(rescue.rover_energy_delivered_wh, 0)

    def test_successful_rescue_in_synthetic_case(self) -> None:
        settings = config()
        settings = MissionSimulationConfig(
            **{
                **settings.__dict__,
                "initial_rover_battery_state": BatteryState(210.0),
                "rover_receiver": RoverSolarReceiverAssumptions(100_000.0, 0.0, 0.3),
            }
        )
        result = simulate(settings).scenario("rescue")
        self.assertTrue(result.rescue_success, result.rescue_failure_reasons)
        self.assertGreater(result.rover_energy_delivered_wh, 0)
        self.assertTrue(any(step.emergency_recovery_success for step in result.steps))
        self.assertTrue(any(step.safe_return_feasible for step in result.steps))

    def test_missing_terrain_is_unknown_and_does_not_create_generation_or_rescue(self) -> None:
        result = simulate(config(with_profiles=False))
        normal = result.scenario("normal")
        self.assertTrue(all(step.direct_station_illumination is None for step in normal.steps))
        self.assertEqual(normal.direct_generation_wh, 0)
        rescue = result.scenario("rescue")
        self.assertFalse(rescue.rescue_success)

    def test_safe_return_logic_and_step_provenance(self) -> None:
        result = simulate(config())
        normal = result.scenario("normal")
        self.assertTrue(any(not step.safe_return_feasible for step in normal.steps))
        self.assertIn("SPICE-derived Sun and UTC/ET geometry; assumed orbit kinematics", normal.provenance)
        self.assertTrue(all("assumed kinematic" in item for item in result.provenance if item.startswith("orbit source")))
        self.assertTrue(all(step.geometry_frame == "MOON_PA_DE440" for step in normal.steps))

    def test_invalid_configuration_rejected(self) -> None:
        settings = config()
        with self.assertRaises(ValueError):
            MissionSimulationConfig(**(settings.__dict__ | {"duration_s": -1}))
        with self.assertRaises(ValueError):
            MissionSimulationConfig(**(settings.__dict__ | {"time_step_s": 0}))
        with self.assertRaises(ValueError):
            MissionSimulationConfig(**(settings.__dict__ | {"mission_timeline": ()}))
        with self.assertRaises(ValueError):
            MissionTimelineSegment(0, 10, "x", 1, "invalid", 0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            StationPowerAssumptions(
                **(settings.station_power.__dict__ | {
                    "battery_config": BatteryConfig(20_000, 0.8, 4_999),
                    "initial_battery_state": BatteryState(10_000),
                })
            )


if __name__ == "__main__":
    unittest.main()
