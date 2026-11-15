"""Unit tests for LUMEN station generation and interval power accounting."""

from dataclasses import FrozenInstanceError
import math
import unittest

from core.lumen.energy_storage import BatteryConfig, BatteryState
from core.lumen.station_power import StationPowerInput, calculate_station_power


class StationPowerTest(unittest.TestCase):
    def base(self, **overrides) -> StationPowerInput:
        values = dict(
            duration_s=3600,
            direct_irradiance_w_m2=1000,
            direct_array_area_m2=1,
            direct_incidence_angle_deg=0,
            direct_conversion_efficiency=0.2,
            station_load_w=200,
            battery_config=None,
            battery_state=None,
        )
        values.update(overrides)
        return StationPowerInput(**values)

    def test_direct_generation(self) -> None:
        result = calculate_station_power(self.base())
        self.assertEqual(result.direct_generation_w, 200)
        self.assertEqual(result.direct_generation_wh, 200)
        self.assertEqual(result.direct_conversion_loss_wh, 800)
        self.assertEqual(result.reflected_generation_wh, 0)
        self.assertEqual(result.gross_generation_wh, 200)

    def test_reflected_generation_is_separate_and_only_when_available(self) -> None:
        result = calculate_station_power(self.base(
            direct_irradiance_w_m2=0,
            reflected_irradiance_w_m2=800,
            reflected_receiver_area_m2=2,
            reflected_incidence_angle_deg=0,
            reflected_conversion_efficiency=0.25,
            reflected_illumination_available=True,
        ))
        self.assertEqual(result.direct_generation_wh, 0)
        self.assertEqual(result.reflected_generation_w, 400)
        self.assertEqual(result.reflected_generation_wh, 400)
        self.assertEqual(result.reflected_conversion_loss_wh, 1200)
        self.assertEqual(result.gross_generation_wh, 400)

    def test_reflected_power_is_not_assumed_available(self) -> None:
        result = calculate_station_power(self.base(
            direct_irradiance_w_m2=0,
            reflected_irradiance_w_m2=1000,
            reflected_receiver_area_m2=1,
            reflected_conversion_efficiency=0.5,
            reflected_illumination_available=False,
        ))
        self.assertEqual(result.reflected_generation_wh, 0)
        self.assertFalse(result.reflected_illumination_available)
        self.assertEqual(result.reflected_conversion_loss_wh, 0)

    def test_combined_generation_has_no_double_counting(self) -> None:
        result = calculate_station_power(self.base(
            direct_irradiance_w_m2=1000,
            direct_array_area_m2=2,
            direct_conversion_efficiency=0.2,
            reflected_irradiance_w_m2=500,
            reflected_receiver_area_m2=2,
            reflected_conversion_efficiency=0.2,
            reflected_illumination_available=True,
            station_load_w=0,
        ))
        self.assertEqual(result.direct_generation_wh, 400)
        self.assertEqual(result.reflected_generation_wh, 200)
        self.assertEqual(result.gross_generation_wh, 600)
        self.assertEqual(
            result.gross_generation_wh,
            result.direct_generation_wh + result.reflected_generation_wh,
        )

    def test_zero_illumination_and_station_load(self) -> None:
        result = calculate_station_power(self.base(
            direct_irradiance_w_m2=0,
            station_load_w=200,
            battery_config=None,
            battery_state=None,
        ))
        self.assertEqual(result.gross_generation_wh, 0)
        self.assertEqual(result.station_load_energy_wh, 200)
        self.assertEqual(result.net_energy_wh, -200)
        self.assertEqual(result.load_unserved_energy_wh, 200)

    def test_incidence_angle_cosine_effect(self) -> None:
        normal = calculate_station_power(self.base(direct_incidence_angle_deg=0))
        oblique = calculate_station_power(self.base(direct_incidence_angle_deg=60))
        grazing = calculate_station_power(self.base(direct_incidence_angle_deg=90))
        self.assertAlmostEqual(oblique.direct_generation_wh, normal.direct_generation_wh * 0.5)
        self.assertAlmostEqual(grazing.direct_generation_wh, 0.0, places=12)

    def test_conversion_efficiency_effect_and_provenance_labels(self) -> None:
        result = calculate_station_power(self.base(
            direct_conversion_efficiency=0.15,
            direct_irradiance_origin="calculated",
            direct_incidence_origin="calculated",
        ))
        self.assertAlmostEqual(result.direct_generation_wh, 150)
        self.assertAlmostEqual(result.direct_conversion_loss_wh, 850)
        self.assertIn("direct_irradiance_w_m2", result.calculated_inputs)
        self.assertIn("direct_incidence_angle_deg", result.calculated_inputs)
        self.assertIn("direct_array_area_m2", result.assumed_inputs)
        self.assertIn("station_load_w", result.assumed_inputs)
        self.assertIn("battery_charge_efficiency", result.assumed_inputs)

    def test_default_station_load_assumption_is_200_w(self) -> None:
        result = calculate_station_power(StationPowerInput(
            duration_s=1800,
            direct_irradiance_w_m2=0,
            direct_array_area_m2=0,
            direct_incidence_angle_deg=0,
            direct_conversion_efficiency=0.2,
            battery_config=None,
            battery_state=None,
        ))
        self.assertEqual(result.station_load_w, 200)
        self.assertEqual(result.station_load_energy_wh, 100)

    def test_surplus_charging_losses_and_energy_conservation(self) -> None:
        config = BatteryConfig(20_000, 0.8, 5_000)
        result = calculate_station_power(self.base(
            direct_irradiance_w_m2=1500,
            direct_array_area_m2=1,
            direct_conversion_efficiency=0.5,
            station_load_w=200,
            battery_config=config,
            battery_state=BatteryState(5_000),
            battery_charge_efficiency=0.9,
        ))
        self.assertEqual(result.gross_generation_wh, 750)
        self.assertEqual(result.station_load_energy_wh, 200)
        self.assertEqual(result.net_energy_wh, 550)
        self.assertAlmostEqual(result.battery_stored_energy_wh, 495)
        self.assertAlmostEqual(result.battery_charge_loss_wh, 55)
        self.assertEqual(result.curtailed_energy_wh, 0)
        self.assertAlmostEqual(result.energy_balance_residual_wh, 0, places=10)
        self.assertAlmostEqual(
            result.gross_generation_wh,
            result.station_load_energy_wh
            + result.battery_stored_energy_wh
            + result.battery_charge_loss_wh
            + result.curtailed_energy_wh,
        )
        self.assertAlmostEqual(result.total_conversion_losses_wh, 805)  # PV + battery conversion

    def test_full_battery_curtailed_surplus_is_reported(self) -> None:
        result = calculate_station_power(self.base(
            direct_irradiance_w_m2=1000,
            direct_array_area_m2=1,
            direct_conversion_efficiency=1,
            station_load_w=0,
            battery_config=BatteryConfig(20_000, 0.8, 5_000),
            battery_state=BatteryState(20_000),
        ))
        self.assertEqual(result.battery_stored_energy_wh, 0)
        self.assertEqual(result.curtailed_energy_wh, 1000)
        self.assertAlmostEqual(result.energy_balance_residual_wh, 0)

    def test_input_and_result_dataclasses_are_immutable(self) -> None:
        inputs = self.base()
        result = calculate_station_power(inputs)
        with self.assertRaises(FrozenInstanceError):
            inputs.station_load_w = 1  # type: ignore[misc]
        with self.assertRaises(FrozenInstanceError):
            result.gross_generation_wh = 1  # type: ignore[misc]

    def test_invalid_inputs_are_rejected(self) -> None:
        common = dict(
            duration_s=3600,
            direct_irradiance_w_m2=100,
            direct_array_area_m2=1,
            direct_incidence_angle_deg=0,
            direct_conversion_efficiency=0.2,
            battery_config=None,
            battery_state=None,
        )
        invalid = (
            {"duration_s": 0},
            {"duration_s": -1},
            {"duration_s": math.inf},
            {"direct_irradiance_w_m2": -1},
            {"direct_array_area_m2": -1},
            {"direct_incidence_angle_deg": -1},
            {"direct_incidence_angle_deg": 91},
            {"direct_conversion_efficiency": -0.1},
            {"direct_conversion_efficiency": 1.1},
            {"reflected_irradiance_w_m2": -1},
            {"reflected_receiver_area_m2": -1},
            {"reflected_incidence_angle_deg": 91},
            {"reflected_conversion_efficiency": 1.1},
            {"station_load_w": -1},
            {"battery_charge_efficiency": 0},
            {"direct_irradiance_origin": "spice"},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                StationPowerInput(**(common | changes))
        with self.assertRaises(ValueError):
            StationPowerInput(**(common | {"battery_config": BatteryConfig(100), "battery_state": BatteryState(101)}))
        with self.assertRaises(ValueError):
            StationPowerInput(**(common | {"battery_state": BatteryState(1)}))


if __name__ == "__main__":
    unittest.main()
