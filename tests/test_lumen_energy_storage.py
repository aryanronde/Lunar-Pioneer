"""Unit tests for LUMEN's immutable battery energy accounting."""

from dataclasses import FrozenInstanceError
import unittest

from core.lumen.energy_storage import (
    BatteryConfig,
    BatteryState,
    charge,
    discharge,
    operational_floor_wh,
    usable_capacity_wh,
)


class EnergyStorageTest(unittest.TestCase):
    def setUp(self) -> None:
        self.station = BatteryConfig(
            nominal_capacity_wh=20_000,
            maximum_depth_of_discharge=0.80,
            protected_reserve_wh=5_000,
        )

    def test_initial_full_battery_and_immutable_dataclasses(self) -> None:
        state = BatteryState(20_000)
        self.assertEqual(state.energy_wh, self.station.nominal_capacity_wh)
        with self.assertRaises(FrozenInstanceError):
            state.energy_wh = 1  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            self.station.nominal_capacity_wh = 10  # type: ignore[misc]

    def test_maximum_usable_capacity_and_operational_floor(self) -> None:
        self.assertEqual(usable_capacity_wh(self.station), 16_000)
        self.assertEqual(operational_floor_wh(self.station), 5_000)
        self.assertEqual(usable_capacity_wh(BatteryConfig(1_000)), 1_000)

    def test_charging_efficiency_and_energy_conservation(self) -> None:
        result = charge(BatteryState(0), BatteryConfig(1_000), 1_000, 0.90)
        self.assertEqual(result.accepted_input_energy_wh, 1_000)
        self.assertEqual(result.stored_energy_wh, 900)
        self.assertAlmostEqual(result.loss_energy_wh, 100)
        self.assertEqual(result.resulting_state.energy_wh, 900)
        self.assertAlmostEqual(
            result.requested_input_energy_wh,
            result.stored_energy_wh + result.loss_energy_wh + result.curtailed_input_energy_wh,
        )

    def test_charge_is_safely_limited_at_capacity_and_reports_curtailed_energy(self) -> None:
        result = charge(BatteryState(950), BatteryConfig(1_000), 100, 0.90)
        self.assertAlmostEqual(result.accepted_input_energy_wh, 50 / 0.90)
        self.assertAlmostEqual(result.stored_energy_wh, 50)
        self.assertAlmostEqual(result.curtailed_input_energy_wh, 100 - 50 / 0.90)
        self.assertEqual(result.resulting_state.energy_wh, 1_000)

    def test_normal_discharge_stops_at_protected_reserve(self) -> None:
        result = discharge(BatteryState(20_000), self.station, 18_000, "normal")
        self.assertEqual(result.accepted_energy_wh, 15_000)
        self.assertEqual(result.unmet_energy_wh, 3_000)
        self.assertEqual(result.resulting_state.energy_wh, 5_000)
        self.assertEqual(result.loss_energy_wh, 0)
        self.assertEqual(result.purpose, "normal")
        self.assertFalse(result.emergency_authorized)

    def test_emergency_discharge_requires_authorization_and_reports_separately(self) -> None:
        with self.assertRaises(PermissionError):
            discharge(BatteryState(6_000), self.station, 1_500, "emergency")
        result = discharge(
            BatteryState(6_000), self.station, 1_500, "emergency", emergency_authorized=True
        )
        self.assertEqual(result.accepted_energy_wh, 1_500)
        self.assertEqual(result.resulting_state.energy_wh, 4_500)
        self.assertEqual(result.purpose, "emergency")
        self.assertTrue(result.emergency_authorized)

    def test_emergency_may_not_cross_maximum_dod_floor(self) -> None:
        result = discharge(
            BatteryState(4_500), self.station, 2_000, "emergency", emergency_authorized=True
        )
        self.assertAlmostEqual(result.accepted_energy_wh, 500)
        self.assertAlmostEqual(result.unmet_energy_wh, 1_500)
        self.assertAlmostEqual(result.resulting_state.energy_wh, 4_000)

    def test_discharge_energy_balance(self) -> None:
        result = discharge(BatteryState(12_000), self.station, 2_500, "normal")
        self.assertEqual(result.accepted_energy_wh, 2_500)
        self.assertEqual(result.resulting_state.energy_wh, 9_500)
        self.assertEqual(12_000, result.resulting_state.energy_wh + result.accepted_energy_wh)
        self.assertEqual(
            result.requested_energy_wh,
            result.accepted_energy_wh + result.unmet_energy_wh,
        )

    def test_charge_and_discharge_power_limits(self) -> None:
        config = BatteryConfig(
            2_000,
            maximum_depth_of_discharge=1.0,
            maximum_charge_power_w=900,
            maximum_discharge_power_w=360,
        )
        charged = charge(BatteryState(0), config, 1_000, 1.0, duration_s=3600)
        self.assertEqual(charged.accepted_input_energy_wh, 900)
        self.assertEqual(charged.curtailed_input_energy_wh, 100)
        discharged = discharge(
            BatteryState(1_000), config, 500, "normal", duration_s=3600
        )
        self.assertEqual(discharged.accepted_energy_wh, 360)
        self.assertEqual(discharged.unmet_energy_wh, 140)

    def test_power_limited_operation_requires_duration(self) -> None:
        config = BatteryConfig(1_000, maximum_charge_power_w=100, maximum_discharge_power_w=100)
        with self.assertRaises(ValueError):
            charge(BatteryState(0), config, 10, 1.0)
        with self.assertRaises(ValueError):
            discharge(BatteryState(100), config, 10, "normal")

    def test_invalid_configurations_states_energy_and_efficiencies(self) -> None:
        invalid_configs = (
            dict(nominal_capacity_wh=0),
            dict(nominal_capacity_wh=-1),
            dict(nominal_capacity_wh=10, maximum_depth_of_discharge=1.1),
            dict(nominal_capacity_wh=10, maximum_depth_of_discharge=-0.1),
            dict(nominal_capacity_wh=10, protected_reserve_wh=11),
            dict(nominal_capacity_wh=10, maximum_depth_of_discharge=0.8, protected_reserve_wh=1),
            dict(nominal_capacity_wh=10, maximum_charge_power_w=-1),
            dict(nominal_capacity_wh=10, maximum_discharge_power_w=float("inf")),
        )
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BatteryConfig(**kwargs)
        for value in (-1, float("nan"), float("inf")):
            with self.subTest(state=value), self.assertRaises(ValueError):
                BatteryState(value)
        with self.assertRaises(ValueError):
            charge(BatteryState(0), BatteryConfig(1_000), 1, 0)
        with self.assertRaises(ValueError):
            charge(BatteryState(0), BatteryConfig(1_000), 1, 1.01)
        with self.assertRaises(ValueError):
            charge(BatteryState(0), BatteryConfig(1_000), -1, 0.9)
        with self.assertRaises(ValueError):
            discharge(BatteryState(1), BatteryConfig(1_000), -1, "normal")
        with self.assertRaises(ValueError):
            discharge(BatteryState(20_001), self.station, 0, "normal")
        with self.assertRaises(ValueError):
            BatteryConfig(1_000, maximum_depth_of_discharge=0.8).nominal_capacity_wh

    def test_invalid_purpose_and_misapplied_authorization(self) -> None:
        with self.assertRaises(ValueError):
            discharge(BatteryState(100), BatteryConfig(1_000), 1, "routine")
        with self.assertRaises(ValueError):
            discharge(
                BatteryState(100), BatteryConfig(1_000), 1, "normal", emergency_authorized=True
            )


if __name__ == "__main__":
    unittest.main()
