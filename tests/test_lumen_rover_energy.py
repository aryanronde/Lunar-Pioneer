"""Unit tests for LUMEN rover energy policy and battery operations."""

from dataclasses import FrozenInstanceError
import math
import unittest

from core.lumen.energy_storage import BatteryConfig, BatteryState
from core.lumen.rover_energy import (
    MissionEnergyRequirements,
    RoverConfig,
    RoverOperatingState,
    RoverState,
    charge_rover,
    consume_energy,
    evaluate_departure,
    safe_return_energy_wh,
    usable_rover_energy_wh,
)


ENERGY_FIELDS = (
    "outbound_travel_energy_wh",
    "outbound_communication_navigation_energy_wh",
    "outbound_thermal_energy_wh",
    "return_travel_energy_wh",
    "return_communication_navigation_energy_wh",
    "return_thermal_survival_energy_wh",
    "uncertainty_reserve_energy_wh",
)


def requirements(**changes) -> MissionEnergyRequirements:
    values = dict(
        outbound_travel_energy_wh=100,
        outbound_communication_navigation_energy_wh=20,
        outbound_thermal_energy_wh=30,
        return_travel_energy_wh=200,
        return_communication_navigation_energy_wh=30,
        return_thermal_survival_energy_wh=40,
        uncertainty_reserve_energy_wh=50,
        energy_origins=tuple((name, "assumed") for name in ENERGY_FIELDS),
    )
    values.update(changes)
    return MissionEnergyRequirements(**values)


class RoverEnergyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RoverConfig()
        self.full = RoverState(BatteryState(1_000))
        self.requirements = requirements()

    def test_full_rover_battery_and_planned_usable_capacity(self) -> None:
        self.assertEqual(self.full.battery_state.energy_wh, 1_000)
        self.assertEqual(usable_rover_energy_wh(self.config), 800)

    def test_safe_return_energy_includes_all_explicit_terms_and_uncertainty(self) -> None:
        self.assertEqual(safe_return_energy_wh(self.requirements), 320)
        self.assertEqual(self.requirements.outbound_operations_energy_wh, 150)
        self.assertEqual(self.requirements.safe_return_energy_wh, 200 + 30 + 40 + 50)

    def test_successful_departure(self) -> None:
        decision = evaluate_departure(self.full, self.config, self.requirements)
        self.assertTrue(decision.departure_allowed)
        self.assertEqual(decision.battery_energy_wh, 1_000)
        self.assertEqual(decision.usable_energy_wh, 800)
        self.assertEqual(decision.predicted_remaining_energy_wh, 850)
        self.assertEqual(decision.safe_return_energy_wh, 320)
        self.assertEqual(decision.energy_margin_wh, 330)
        self.assertIn("allowed", decision.reason.lower())

    def test_departure_rejected_when_predicted_energy_is_insufficient(self) -> None:
        req = requirements(outbound_travel_energy_wh=500)
        decision = evaluate_departure(self.full, self.config, req)
        self.assertFalse(decision.departure_allowed)
        self.assertEqual(decision.predicted_remaining_energy_wh, 450)
        self.assertEqual(decision.energy_margin_wh, -70)
        self.assertIn("rejected", decision.reason.lower())

    def test_travel_consumption_reports_requested_and_accepted_energy(self) -> None:
        result = consume_energy(self.full, self.config, 100, self.requirements)
        self.assertEqual(result.requested_energy_wh, 100)
        self.assertEqual(result.accepted_energy_wh, 100)
        self.assertEqual(result.unmet_energy_wh, 0)
        self.assertEqual(result.battery_energy_wh, 900)
        self.assertEqual(result.usable_energy_wh, 700)
        self.assertEqual(result.operating_state, RoverOperatingState.NORMAL)

    def test_communication_thermal_and_travel_energy_are_accounted_without_floor_double_count(self) -> None:
        req = requirements(
            outbound_travel_energy_wh=60,
            outbound_communication_navigation_energy_wh=25,
            outbound_thermal_energy_wh=15,
        )
        total = req.outbound_operations_energy_wh
        result = consume_energy(self.full, self.config, total, req)
        self.assertEqual(total, 100)
        self.assertEqual(result.battery_energy_wh, 900)
        self.assertEqual(result.accepted_energy_wh, 100)
        # Safe-return energy is reported as a threshold and is not subtracted.
        self.assertEqual(result.battery_energy_wh, self.full.battery_state.energy_wh - total)
        self.assertEqual(result.safe_return_energy_wh, 320)

    def test_uncertainty_reserve_is_included_in_safe_return(self) -> None:
        no_uncertainty = safe_return_energy_wh(requirements(uncertainty_reserve_energy_wh=0))
        with_uncertainty = safe_return_energy_wh(requirements(uncertainty_reserve_energy_wh=75))
        self.assertEqual(no_uncertainty, 270)
        self.assertEqual(with_uncertainty, 345)

    def test_return_required_transition_at_configured_margin(self) -> None:
        result = consume_energy(self.full, self.config, 450, self.requirements)
        self.assertEqual(result.battery_energy_wh, 550)
        self.assertEqual(result.usable_energy_wh, 350)
        self.assertEqual(result.energy_margin_wh, 30)
        self.assertEqual(result.operating_state, RoverOperatingState.RETURN_REQUIRED)

    def test_emergency_transition_when_safe_return_energy_is_unavailable(self) -> None:
        result = consume_energy(self.full, self.config, 550, self.requirements)
        self.assertEqual(result.battery_energy_wh, 450)
        self.assertEqual(result.usable_energy_wh, 250)
        self.assertEqual(result.energy_margin_wh, -70)
        self.assertEqual(result.operating_state, RoverOperatingState.EMERGENCY)

    def test_battery_limit_reports_unmet_consumption(self) -> None:
        result = consume_energy(self.full, self.config, 900, self.requirements)
        self.assertEqual(result.accepted_energy_wh, 800)
        self.assertEqual(result.unmet_energy_wh, 100)
        self.assertEqual(result.battery_energy_wh, 200)
        self.assertEqual(result.operating_state, RoverOperatingState.EMERGENCY)

    def test_rover_charging_efficiency_and_capacity_limit(self) -> None:
        state = RoverState(BatteryState(900))
        result = charge_rover(state, self.config, 200, self.requirements)
        self.assertAlmostEqual(result.accepted_input_energy_wh, 100 / 0.9)
        self.assertAlmostEqual(result.stored_energy_wh, 100)
        self.assertAlmostEqual(result.loss_energy_wh, 100 / 0.9 - 100)
        self.assertAlmostEqual(result.curtailed_input_energy_wh, 200 - 100 / 0.9)
        self.assertEqual(result.battery_energy_wh, 1_000)

    def test_rover_charge_and_discharge_power_limits(self) -> None:
        config = RoverConfig(
            battery_config=BatteryConfig(
                1_000,
                maximum_depth_of_discharge=0.8,
                protected_reserve_wh=200,
                maximum_charge_power_w=90,
                maximum_discharge_power_w=360,
            )
        )
        charged = charge_rover(
            RoverState(BatteryState(200)), config, 100, self.requirements, duration_s=3600
        )
        self.assertEqual(charged.accepted_input_energy_wh, 90)
        self.assertEqual(charged.stored_energy_wh, 81)
        consumed = consume_energy(
            self.full, config, 500, self.requirements, duration_s=3600
        )
        self.assertEqual(consumed.accepted_energy_wh, 360)
        self.assertEqual(consumed.unmet_energy_wh, 140)

    def test_input_and_result_dataclasses_are_immutable(self) -> None:
        result = evaluate_departure(self.full, self.config, self.requirements)
        with self.assertRaises(FrozenInstanceError):
            self.full.operating_state = RoverOperatingState.EMERGENCY  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            self.requirements.return_travel_energy_wh = 0  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.energy_margin_wh = 0  # type: ignore[misc]

    def test_operating_states_are_validated_and_recovered_after_charging(self) -> None:
        low = consume_energy(self.full, self.config, 550, self.requirements)
        self.assertEqual(low.operating_state, RoverOperatingState.EMERGENCY)
        charged = charge_rover(
            RoverState(BatteryState(low.battery_energy_wh), low.operating_state),
            self.config,
            188.8,
            self.requirements,
        )
        self.assertEqual(charged.operating_state, RoverOperatingState.RETURN_REQUIRED)
        with self.assertRaises(ValueError):
            RoverState(BatteryState(500), "UNKNOWN")  # type: ignore[arg-type]

    def test_invalid_inputs_rejected(self) -> None:
        base = dict(
            outbound_travel_energy_wh=1,
            outbound_communication_navigation_energy_wh=1,
            outbound_thermal_energy_wh=1,
            return_travel_energy_wh=1,
            return_communication_navigation_energy_wh=1,
            return_thermal_survival_energy_wh=1,
            uncertainty_reserve_energy_wh=1,
            energy_origins=tuple((name, "assumed") for name in ENERGY_FIELDS),
        )
        for field_name, value in (
            ("outbound_travel_energy_wh", -1),
            ("return_thermal_survival_energy_wh", math.inf),
            ("uncertainty_reserve_energy_wh", math.nan),
        ):
            with self.subTest(field=field_name), self.assertRaises(ValueError):
                MissionEnergyRequirements(**(base | {field_name: value}))
        with self.assertRaises(ValueError):
            MissionEnergyRequirements(**(base | {"energy_origins": ()}))
        bad_origins = tuple((name, "spice") for name in ENERGY_FIELDS)
        with self.assertRaises(ValueError):
            MissionEnergyRequirements(**(base | {"energy_origins": bad_origins}))
        with self.assertRaises(ValueError):
            RoverConfig(charging_efficiency=0)
        with self.assertRaises(ValueError):
            RoverConfig(return_required_margin_wh=-1)
        with self.assertRaises(ValueError):
            evaluate_departure(RoverState(BatteryState(1_001)), self.config, self.requirements)
        with self.assertRaises(ValueError):
            consume_energy(self.full, self.config, -1, self.requirements)
        with self.assertRaises(ValueError):
            charge_rover(self.full, self.config, -1, self.requirements)


if __name__ == "__main__":
    unittest.main()
