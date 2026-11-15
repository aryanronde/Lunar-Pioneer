"""Validated, Wh-based battery state and energy accounting for LUMEN.

Charge and discharge operations are conservatively limited when capacity,
reserve, or power limits are reached. Every result reports the unaccepted
portion so callers can account for curtailed or unmet energy explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


def _finite_nonnegative(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return number


@dataclass(frozen=True)
class BatteryConfig:
    """Battery limits in Wh and W.

    ``maximum_depth_of_discharge`` sets the lowest planned state of charge.
    A protected reserve can raise the normal operational floor. Emergency
    discharge may cross that reserve, but never the planned DoD floor.
    """

    nominal_capacity_wh: float
    maximum_depth_of_discharge: float = 1.0
    protected_reserve_wh: float = 0.0
    maximum_charge_power_w: float | None = None
    maximum_discharge_power_w: float | None = None

    def __post_init__(self) -> None:
        capacity = _finite_nonnegative(self.nominal_capacity_wh, "Nominal capacity")
        if capacity <= 0:
            raise ValueError("Nominal capacity must be greater than zero.")
        dod = float(self.maximum_depth_of_discharge)
        if not math.isfinite(dod) or not 0.0 <= dod <= 1.0:
            raise ValueError("Maximum depth of discharge must be finite and in [0, 1].")
        reserve = _finite_nonnegative(self.protected_reserve_wh, "Protected reserve")
        if reserve > capacity:
            raise ValueError("Protected reserve cannot exceed nominal capacity.")
        planned_floor = capacity * (1.0 - dod)
        if reserve < planned_floor:
            raise ValueError("Protected reserve cannot be below the maximum-DoD state-of-charge floor.")
        for name in ("maximum_charge_power_w", "maximum_discharge_power_w"):
            value = getattr(self, name)
            if value is not None:
                power = _finite_nonnegative(value, name.replace("_", " ").capitalize())
                object.__setattr__(self, name, power)
        object.__setattr__(self, "nominal_capacity_wh", capacity)
        object.__setattr__(self, "maximum_depth_of_discharge", dod)
        object.__setattr__(self, "protected_reserve_wh", reserve)


@dataclass(frozen=True)
class BatteryState:
    """Immutable battery energy state, constrained to [0, nominal] Wh."""

    energy_wh: float

    def __post_init__(self) -> None:
        energy = _finite_nonnegative(self.energy_wh, "Battery energy")
        object.__setattr__(self, "energy_wh", energy)


@dataclass(frozen=True)
class ChargeResult:
    """Charge accounting; energies are Wh and duration is seconds."""

    requested_input_energy_wh: float
    accepted_input_energy_wh: float
    stored_energy_wh: float
    loss_energy_wh: float
    curtailed_input_energy_wh: float
    resulting_state: BatteryState


@dataclass(frozen=True)
class DischargeResult:
    """Discharge accounting; accepted energy is removed from the battery."""

    purpose: str
    emergency_authorized: bool
    requested_energy_wh: float
    accepted_energy_wh: float
    unmet_energy_wh: float
    loss_energy_wh: float
    resulting_state: BatteryState


def usable_capacity_wh(config: BatteryConfig) -> float:
    """Return planned discharge capacity from full, based on maximum DoD."""
    return config.nominal_capacity_wh * config.maximum_depth_of_discharge


def operational_floor_wh(config: BatteryConfig) -> float:
    """Return the minimum state permitted during normal operation, in Wh."""
    return max(
        config.nominal_capacity_wh * (1.0 - config.maximum_depth_of_discharge),
        config.protected_reserve_wh,
    )


def _check_state(state: BatteryState, config: BatteryConfig) -> None:
    if not isinstance(state, BatteryState) or not isinstance(config, BatteryConfig):
        raise TypeError("state must be BatteryState and config must be BatteryConfig.")
    if state.energy_wh > config.nominal_capacity_wh:
        raise ValueError("Battery state exceeds nominal capacity.")
    if state.energy_wh < config.nominal_capacity_wh * (1.0 - config.maximum_depth_of_discharge):
        raise ValueError("Battery state is below the configured maximum-DoD floor.")


def charge(
    state: BatteryState,
    config: BatteryConfig,
    input_energy_wh: float,
    efficiency: float,
    *,
    duration_s: float | None = None,
) -> ChargeResult:
    """Apply charging energy, limiting by battery capacity and charge power.

    ``input_energy_wh`` is energy presented at the charger input. Accepted
    input is converted by ``efficiency`` into stored energy; conversion losses
    and curtailed input are reported separately. If a charge-power limit is
    configured, ``duration_s`` is required to enforce it.
    """
    _check_state(state, config)
    requested = _finite_nonnegative(input_energy_wh, "Input energy")
    eta = float(efficiency)
    if not math.isfinite(eta) or not 0.0 < eta <= 1.0:
        raise ValueError("Charging efficiency must be finite and in (0, 1].")
    duration = _validated_duration(duration_s, config.maximum_charge_power_w)
    power_limited_input = requested
    if config.maximum_charge_power_w is not None:
        power_limited_input = min(requested, config.maximum_charge_power_w * duration / 3600.0)
    room_input = (config.nominal_capacity_wh - state.energy_wh) / eta
    accepted = min(power_limited_input, room_input)
    stored = accepted * eta
    losses = accepted - stored
    curtailed = requested - accepted
    result_state = BatteryState(min(config.nominal_capacity_wh, state.energy_wh + stored))
    return ChargeResult(requested, accepted, stored, losses, curtailed, result_state)


def discharge(
    state: BatteryState,
    config: BatteryConfig,
    requested_energy_wh: float,
    purpose: str,
    *,
    emergency_authorized: bool = False,
    duration_s: float | None = None,
) -> DischargeResult:
    """Remove battery energy, enforcing reserve and configured power limits.

    Normal discharge cannot cross ``operational_floor_wh``. ``purpose`` must
    be ``"normal"`` or ``"emergency"``. Emergency use must pass the explicit
    ``emergency_authorized=True`` flag and may cross only the protected
    reserve, never the maximum-DoD floor. Requests are safe-limited and any
    unmet amount is returned in the result.
    """
    _check_state(state, config)
    requested = _finite_nonnegative(requested_energy_wh, "Requested discharge energy")
    mode = str(purpose).strip().lower()
    if mode not in {"normal", "emergency"}:
        raise ValueError("purpose must be 'normal' or 'emergency'.")
    if mode == "emergency" and emergency_authorized is not True:
        raise PermissionError("Emergency discharge requires emergency_authorized=True.")
    if mode == "normal" and emergency_authorized:
        raise ValueError("Emergency authorization is only valid for emergency discharge.")
    duration = _validated_duration(duration_s, config.maximum_discharge_power_w)
    floor = operational_floor_wh(config) if mode == "normal" else (
        config.nominal_capacity_wh * (1.0 - config.maximum_depth_of_discharge)
    )
    available = max(0.0, state.energy_wh - floor)
    power_limit = requested
    if config.maximum_discharge_power_w is not None:
        power_limit = min(requested, config.maximum_discharge_power_w * duration / 3600.0)
    accepted = min(requested, available, power_limit)
    resulting = BatteryState(max(floor, state.energy_wh - accepted))
    return DischargeResult(
        purpose=mode,
        emergency_authorized=(mode == "emergency"),
        requested_energy_wh=requested,
        accepted_energy_wh=accepted,
        unmet_energy_wh=requested - accepted,
        loss_energy_wh=0.0,
        resulting_state=resulting,
    )


def _validated_duration(duration_s: float | None, power_limit_w: float | None) -> float:
    if duration_s is None:
        if power_limit_w is not None:
            raise ValueError("duration_s is required when a battery power limit is configured.")
        return 0.0
    duration = float(duration_s)
    if not math.isfinite(duration) or duration < 0.0:
        raise ValueError("duration_s must be finite and nonnegative.")
    return duration
