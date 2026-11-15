"""Rover battery and mission-energy policy, with all energy in Wh.

Mission energy quantities are explicit inputs. Safe-return energy is a
decision threshold, not an extra battery withdrawal. Orbital rescue or
reflected-light recovery is intentionally outside this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Literal

from .energy_storage import (
    BatteryConfig,
    BatteryState,
    ChargeResult,
    DischargeResult,
    charge as battery_charge,
    discharge as battery_discharge,
    operational_floor_wh,
    usable_capacity_wh,
)


EnergyOrigin = Literal["assumed", "calculated"]


class RoverOperatingState(str, Enum):
    NORMAL = "NORMAL"
    RETURN_REQUIRED = "RETURN_REQUIRED"
    EMERGENCY = "EMERGENCY"


def _nonnegative(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return number


@dataclass(frozen=True)
class RoverConfig:
    """Rover battery/policy assumptions: capacity and reserves in Wh, power in W."""

    battery_config: BatteryConfig = BatteryConfig(
        nominal_capacity_wh=1_000.0,
        maximum_depth_of_discharge=0.80,
        protected_reserve_wh=200.0,
    )
    charging_efficiency: float = 0.90
    return_required_margin_wh: float = 100.0

    def __post_init__(self) -> None:
        if not isinstance(self.battery_config, BatteryConfig):
            raise TypeError("battery_config must be a BatteryConfig.")
        efficiency = float(self.charging_efficiency)
        if not math.isfinite(efficiency) or not 0.0 < efficiency <= 1.0:
            raise ValueError("charging_efficiency must be finite and in (0, 1].")
        object.__setattr__(self, "charging_efficiency", efficiency)
        object.__setattr__(
            self,
            "return_required_margin_wh",
            _nonnegative(self.return_required_margin_wh, "Return-required margin"),
        )


@dataclass(frozen=True)
class RoverState:
    """Current rover battery and operating state."""

    battery_state: BatteryState
    operating_state: RoverOperatingState = RoverOperatingState.NORMAL

    def __post_init__(self) -> None:
        if not isinstance(self.battery_state, BatteryState):
            raise TypeError("battery_state must be a BatteryState.")
        try:
            state = RoverOperatingState(self.operating_state)
        except ValueError as exc:
            raise ValueError("operating_state must be NORMAL, RETURN_REQUIRED, or EMERGENCY.") from exc
        object.__setattr__(self, "operating_state", state)


@dataclass(frozen=True)
class MissionEnergyRequirements:
    """Explicit planned energy terms, all Wh; no energy terms are inferred."""

    outbound_travel_energy_wh: float
    outbound_communication_navigation_energy_wh: float
    outbound_thermal_energy_wh: float
    return_travel_energy_wh: float
    return_communication_navigation_energy_wh: float
    return_thermal_survival_energy_wh: float
    uncertainty_reserve_energy_wh: float
    energy_origins: tuple[tuple[str, EnergyOrigin], ...]

    def __post_init__(self) -> None:
        fields = (
            "outbound_travel_energy_wh",
            "outbound_communication_navigation_energy_wh",
            "outbound_thermal_energy_wh",
            "return_travel_energy_wh",
            "return_communication_navigation_energy_wh",
            "return_thermal_survival_energy_wh",
            "uncertainty_reserve_energy_wh",
        )
        for name in fields:
            object.__setattr__(self, name, _nonnegative(getattr(self, name), name.replace("_", " ")))
        origins = tuple((str(name), origin) for name, origin in self.energy_origins)
        expected = set(fields)
        received = [name for name, _ in origins]
        if set(received) != expected or len(received) != len(expected):
            raise ValueError("energy_origins must label every energy input exactly once.")
        if any(origin not in ("assumed", "calculated") for _, origin in origins):
            raise ValueError("Each energy origin must be 'assumed' or 'calculated'.")
        object.__setattr__(self, "energy_origins", origins)

    @property
    def outbound_operations_energy_wh(self) -> float:
        return (
            self.outbound_travel_energy_wh
            + self.outbound_communication_navigation_energy_wh
            + self.outbound_thermal_energy_wh
        )

    @property
    def safe_return_energy_wh(self) -> float:
        """Return travel + communications/navigation + thermal + uncertainty (Wh)."""
        return (
            self.return_travel_energy_wh
            + self.return_communication_navigation_energy_wh
            + self.return_thermal_survival_energy_wh
            + self.uncertainty_reserve_energy_wh
        )


@dataclass(frozen=True)
class DepartureDecision:
    """Departure gate and its energy accounting, with energies in Wh."""

    departure_allowed: bool
    reason: str
    battery_energy_wh: float
    usable_energy_wh: float
    outbound_operations_energy_wh: float
    predicted_remaining_energy_wh: float
    safe_return_energy_wh: float
    energy_margin_wh: float


@dataclass(frozen=True)
class RoverConsumptionResult:
    """Battery consumption result; requested/accepted/unmet energy in Wh."""

    requested_energy_wh: float
    accepted_energy_wh: float
    unmet_energy_wh: float
    battery_energy_wh: float
    usable_energy_wh: float
    safe_return_energy_wh: float
    energy_margin_wh: float
    operating_state: RoverOperatingState
    reason: str
    battery_discharge: DischargeResult


@dataclass(frozen=True)
class RoverChargingResult:
    """Rover charge result; input, stored, loss and curtailed energy in Wh."""

    requested_input_energy_wh: float
    accepted_input_energy_wh: float
    stored_energy_wh: float
    loss_energy_wh: float
    curtailed_input_energy_wh: float
    battery_energy_wh: float
    usable_energy_wh: float
    safe_return_energy_wh: float
    energy_margin_wh: float
    operating_state: RoverOperatingState
    battery_charge: ChargeResult


def usable_rover_energy_wh(config: RoverConfig) -> float:
    """Planned usable energy from full battery state, in Wh."""
    return usable_capacity_wh(config.battery_config)


def safe_return_energy_wh(requirements: MissionEnergyRequirements) -> float:
    """Calculate safe-return threshold from explicit component inputs (Wh)."""
    if not isinstance(requirements, MissionEnergyRequirements):
        raise TypeError("requirements must be MissionEnergyRequirements.")
    return requirements.safe_return_energy_wh


def evaluate_departure(
    state: RoverState,
    config: RoverConfig,
    requirements: MissionEnergyRequirements,
) -> DepartureDecision:
    """Allow departure only when planned outbound use leaves safe-return energy."""
    _validate_rover_state(state, config)
    if not isinstance(requirements, MissionEnergyRequirements):
        raise TypeError("requirements must be MissionEnergyRequirements.")
    floor = operational_floor_wh(config.battery_config)
    usable = max(0.0, state.battery_state.energy_wh - floor)
    outbound = requirements.outbound_operations_energy_wh
    predicted = state.battery_state.energy_wh - outbound
    available_after_outbound = max(0.0, predicted - floor)
    safe_return = requirements.safe_return_energy_wh
    margin = available_after_outbound - safe_return
    allowed = margin >= 0.0
    reason = (
        "Departure allowed: predicted post-outbound usable energy meets the safe-return requirement."
        if allowed else
        "Departure rejected: predicted post-outbound usable energy is below the safe-return requirement."
    )
    return DepartureDecision(
        departure_allowed=allowed,
        reason=reason,
        battery_energy_wh=state.battery_state.energy_wh,
        usable_energy_wh=usable,
        outbound_operations_energy_wh=outbound,
        predicted_remaining_energy_wh=predicted,
        safe_return_energy_wh=safe_return,
        energy_margin_wh=margin,
    )


def consume_energy(
    state: RoverState,
    config: RoverConfig,
    requested_energy_wh: float,
    requirements: MissionEnergyRequirements,
    *,
    duration_s: float | None = None,
) -> RoverConsumptionResult:
    """Consume rover energy with the shared storage module's battery limits."""
    _validate_rover_state(state, config)
    _require_requirements(requirements)
    requested = _nonnegative(requested_energy_wh, "Requested consumption")
    result = battery_discharge(
        state.battery_state,
        config.battery_config,
        requested,
        "normal",
        duration_s=duration_s,
    )
    next_state = RoverState(result.resulting_state, state.operating_state)
    operating, reason, usable, margin = _evaluate_operating_state(next_state, config, requirements)
    return RoverConsumptionResult(
        requested_energy_wh=requested,
        accepted_energy_wh=result.accepted_energy_wh,
        unmet_energy_wh=result.unmet_energy_wh,
        battery_energy_wh=result.resulting_state.energy_wh,
        usable_energy_wh=usable,
        safe_return_energy_wh=requirements.safe_return_energy_wh,
        energy_margin_wh=margin,
        operating_state=operating,
        reason=reason,
        battery_discharge=result,
    )


def charge_rover(
    state: RoverState,
    config: RoverConfig,
    input_energy_wh: float,
    requirements: MissionEnergyRequirements,
    *,
    duration_s: float | None = None,
) -> RoverChargingResult:
    """Charge from explicit input energy using configured efficiency/limits."""
    _validate_rover_state(state, config)
    _require_requirements(requirements)
    requested = _nonnegative(input_energy_wh, "Requested charging input")
    result = battery_charge(
        state.battery_state,
        config.battery_config,
        requested,
        config.charging_efficiency,
        duration_s=duration_s,
    )
    next_state = RoverState(result.resulting_state, state.operating_state)
    operating, _reason, usable, margin = _evaluate_operating_state(next_state, config, requirements)
    return RoverChargingResult(
        requested_input_energy_wh=requested,
        accepted_input_energy_wh=result.accepted_input_energy_wh,
        stored_energy_wh=result.stored_energy_wh,
        loss_energy_wh=result.loss_energy_wh,
        curtailed_input_energy_wh=result.curtailed_input_energy_wh,
        battery_energy_wh=result.resulting_state.energy_wh,
        usable_energy_wh=usable,
        safe_return_energy_wh=requirements.safe_return_energy_wh,
        energy_margin_wh=margin,
        operating_state=operating,
        battery_charge=result,
    )


def _evaluate_operating_state(
    state: RoverState,
    config: RoverConfig,
    requirements: MissionEnergyRequirements,
) -> tuple[RoverOperatingState, str, float, float]:
    floor = operational_floor_wh(config.battery_config)
    energy = state.battery_state.energy_wh
    usable = max(0.0, energy - floor)
    safe_return = requirements.safe_return_energy_wh
    margin = usable - safe_return
    if margin < 0.0:
        return (
            RoverOperatingState.EMERGENCY,
            "Emergency: available usable energy is below the safe-return requirement.",
            usable,
            margin,
        )
    if margin <= config.return_required_margin_wh:
        return (
            RoverOperatingState.RETURN_REQUIRED,
            "Return required: energy margin is at or within the configured return threshold.",
            usable,
            margin,
        )
    return RoverOperatingState.NORMAL, "Normal: energy margin exceeds the return threshold.", usable, margin


def _validate_rover_state(state: RoverState, config: RoverConfig) -> None:
    if not isinstance(state, RoverState) or not isinstance(config, RoverConfig):
        raise TypeError("state must be RoverState and config must be RoverConfig.")
    floor = operational_floor_wh(config.battery_config)
    energy = state.battery_state.energy_wh
    if energy > config.battery_config.nominal_capacity_wh:
        raise ValueError("Rover battery state exceeds nominal capacity.")
    if energy < floor:
        raise ValueError("Rover battery state is below its configured maximum-DoD floor.")


def _require_requirements(requirements: MissionEnergyRequirements) -> None:
    if not isinstance(requirements, MissionEnergyRequirements):
        raise TypeError("requirements must be MissionEnergyRequirements.")
