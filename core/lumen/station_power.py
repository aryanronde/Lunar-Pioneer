"""Station solar generation and interval energy balance, using W, Wh, m² and s.

Direct and orbital-reflected illumination are independent inputs. Reflected
generation is zero unless its availability flag is explicitly true. The
module models electrical conversion and optional battery charging; it does
not infer reflector visibility or orbital availability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Literal

from .energy_storage import BatteryConfig, BatteryState, ChargeResult, charge


InputOrigin = Literal["assumed", "calculated"]


def _finite(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _nonnegative(value: float, label: str) -> float:
    number = _finite(value, label)
    if number < 0.0:
        raise ValueError(f"{label} must be nonnegative.")
    return number


def _efficiency(value: float, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 < result <= 1.0:
        raise ValueError(f"{label} must be in (0, 1].")
    return result


def _angle(value: float, label: str) -> float:
    result = _finite(value, label)
    if not 0.0 <= result <= 90.0:
        raise ValueError(f"{label} must be in [0, 90] degrees.")
    return result


@dataclass(frozen=True)
class StationPowerInput:
    """Inputs for one station power interval.

    Irradiance is W/m², array/receiver area is m², incidence is degrees,
    conversion and battery charge efficiency are fractions, load is W, and
    duration is seconds. Areas, efficiencies, load, battery limits and the
    default 200 W load are assumed unless the irradiance/incidence origin is
    explicitly marked ``"calculated"``. Reflected illumination defaults to
    unavailable, regardless of any orbital geometry assumptions.
    """

    duration_s: float
    direct_irradiance_w_m2: float
    direct_array_area_m2: float
    direct_incidence_angle_deg: float
    direct_conversion_efficiency: float
    reflected_irradiance_w_m2: float = 0.0
    reflected_receiver_area_m2: float = 0.0
    reflected_incidence_angle_deg: float = 0.0
    reflected_conversion_efficiency: float = 1.0
    reflected_illumination_available: bool = False
    station_load_w: float = 200.0
    battery_charge_efficiency: float = 0.90
    battery_config: BatteryConfig | None = field(
        default_factory=lambda: BatteryConfig(
            nominal_capacity_wh=20_000.0,
            maximum_depth_of_discharge=0.80,
            protected_reserve_wh=5_000.0,
        )
    )
    battery_state: BatteryState | None = field(default_factory=lambda: BatteryState(5_000.0))
    direct_irradiance_origin: InputOrigin = "assumed"
    direct_incidence_origin: InputOrigin = "assumed"
    reflected_irradiance_origin: InputOrigin = "assumed"
    reflected_incidence_origin: InputOrigin = "assumed"

    def __post_init__(self) -> None:
        for name in (
            "duration_s", "direct_irradiance_w_m2", "direct_array_area_m2",
            "reflected_irradiance_w_m2", "reflected_receiver_area_m2", "station_load_w",
        ):
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name.replace("_", " ")))
        if self.duration_s <= 0.0:
            raise ValueError("duration_s must be greater than zero.")
        for name in ("direct_incidence_angle_deg", "reflected_incidence_angle_deg"):
            object.__setattr__(self, name, _angle(getattr(self, name), name.replace("_", " ")))
        for name in (
            "direct_conversion_efficiency", "reflected_conversion_efficiency", "battery_charge_efficiency"
        ):
            object.__setattr__(self, name, _efficiency(getattr(self, name), name.replace("_", " ")))
        if not isinstance(self.reflected_illumination_available, bool):
            raise ValueError("reflected_illumination_available must be a bool.")
        if self.battery_config is None:
            if self.battery_state is not None:
                raise ValueError("battery_state requires battery_config.")
        elif not isinstance(self.battery_config, BatteryConfig) or not isinstance(self.battery_state, BatteryState):
            raise ValueError("battery_config and battery_state must both be valid energy_storage dataclasses.")
        elif self.battery_state.energy_wh > self.battery_config.nominal_capacity_wh:
            raise ValueError("battery_state exceeds the configured nominal capacity.")
        for name in (
            "direct_irradiance_origin", "direct_incidence_origin",
            "reflected_irradiance_origin", "reflected_incidence_origin",
        ):
            if getattr(self, name) not in ("assumed", "calculated"):
                raise ValueError(f"{name} must be 'assumed' or 'calculated'.")


@dataclass(frozen=True)
class StationPowerResult:
    """Calculated power and energy balance for one interval; energy is Wh."""

    duration_s: float
    direct_generation_w: float
    direct_generation_wh: float
    direct_conversion_loss_wh: float
    reflected_generation_w: float
    reflected_generation_wh: float
    reflected_conversion_loss_wh: float
    reflected_illumination_available: bool
    gross_generation_w: float
    gross_generation_wh: float
    station_load_w: float
    station_load_energy_wh: float
    net_energy_wh: float
    load_unserved_energy_wh: float
    battery_charge: ChargeResult | None
    battery_stored_energy_wh: float
    battery_charge_loss_wh: float
    curtailed_energy_wh: float
    total_conversion_losses_wh: float
    energy_balance_residual_wh: float
    assumed_inputs: tuple[str, ...]
    calculated_inputs: tuple[str, ...]


def calculate_station_power(inputs: StationPowerInput) -> StationPowerResult:
    """Calculate direct/reflected electrical generation and station balance.

    Power uses ``irradiance * area * cos(incidence) * efficiency``. Net energy
    is gross electrical generation minus requested station load. Any deficit
    is reported as unserved load; this function does not silently discharge a
    battery. Positive surplus is passed once to ``energy_storage.charge``.
    Without a configured battery, the surplus is explicitly curtailed.
    """
    if not isinstance(inputs, StationPowerInput):
        raise TypeError("inputs must be StationPowerInput.")
    hours = inputs.duration_s / 3600.0

    direct_cos = math.cos(math.radians(inputs.direct_incidence_angle_deg))
    direct_incident_w = inputs.direct_irradiance_w_m2 * inputs.direct_array_area_m2 * direct_cos
    direct_w = direct_incident_w * inputs.direct_conversion_efficiency

    reflected_is_active = inputs.reflected_illumination_available
    reflected_cos = math.cos(math.radians(inputs.reflected_incidence_angle_deg))
    reflected_incident_w = (
        inputs.reflected_irradiance_w_m2 * inputs.reflected_receiver_area_m2 * reflected_cos
        if reflected_is_active else 0.0
    )
    reflected_w = reflected_incident_w * inputs.reflected_conversion_efficiency

    direct_wh = direct_w * hours
    reflected_wh = reflected_w * hours
    gross_w = direct_w + reflected_w
    gross_wh = direct_wh + reflected_wh
    load_wh = inputs.station_load_w * hours
    net_wh = gross_wh - load_wh
    load_unserved = max(0.0, -net_wh)
    surplus_wh = max(0.0, net_wh)

    direct_loss = (direct_incident_w - direct_w) * hours
    reflected_loss = (reflected_incident_w - reflected_w) * hours
    charge_result: ChargeResult | None = None
    stored_wh = charge_loss = curtailed = 0.0
    if inputs.battery_config is not None and inputs.battery_state is not None:
        charge_result = charge(
            inputs.battery_state,
            inputs.battery_config,
            surplus_wh,
            inputs.battery_charge_efficiency,
            duration_s=inputs.duration_s,
        )
        stored_wh = charge_result.stored_energy_wh
        charge_loss = charge_result.loss_energy_wh
        curtailed = charge_result.curtailed_input_energy_wh
    else:
        curtailed = surplus_wh

    total_losses = direct_loss + reflected_loss + charge_loss
    # Electrical generation balances served loads, battery input/loss, and
    # curtailment. Unserved load is reported separately because it was not
    # supplied by generation or battery dispatch.
    served_load = load_wh - load_unserved
    battery_input = stored_wh + charge_loss
    residual = gross_wh - served_load - battery_input - curtailed

    assumed = [
        "direct_array_area_m2", "direct_conversion_efficiency", "reflected_receiver_area_m2",
        "reflected_conversion_efficiency", "station_load_w", "battery_charge_efficiency",
        "battery_config", "battery_state", "reflected_illumination_available",
    ]
    calculated = []
    origin_fields = (
        ("direct_irradiance_w_m2", inputs.direct_irradiance_origin),
        ("direct_incidence_angle_deg", inputs.direct_incidence_origin),
        ("reflected_irradiance_w_m2", inputs.reflected_irradiance_origin),
        ("reflected_incidence_angle_deg", inputs.reflected_incidence_origin),
    )
    for parameter, origin in origin_fields:
        (assumed if origin == "assumed" else calculated).append(parameter)

    return StationPowerResult(
        duration_s=inputs.duration_s,
        direct_generation_w=direct_w,
        direct_generation_wh=direct_wh,
        direct_conversion_loss_wh=direct_loss,
        reflected_generation_w=reflected_w,
        reflected_generation_wh=reflected_wh,
        reflected_conversion_loss_wh=reflected_loss,
        reflected_illumination_available=reflected_is_active,
        gross_generation_w=gross_w,
        gross_generation_wh=gross_wh,
        station_load_w=inputs.station_load_w,
        station_load_energy_wh=load_wh,
        net_energy_wh=net_wh,
        load_unserved_energy_wh=load_unserved,
        battery_charge=charge_result,
        battery_stored_energy_wh=stored_wh,
        battery_charge_loss_wh=charge_loss,
        curtailed_energy_wh=curtailed,
        total_conversion_losses_wh=total_losses,
        energy_balance_residual_wh=residual,
        assumed_inputs=tuple(assumed),
        calculated_inputs=tuple(calculated),
    )
