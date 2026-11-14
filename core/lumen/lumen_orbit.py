"""Preliminary circular lunar orbit kinematics (assumed, not propagated).

Orbit positions are Moon-centred vectors expressed in J2000. This matches
``SpiceGeometryEngine.sun_position_km`` and avoids confusing lunar-fixed PA
or ME frames with inertial axes. The two-body circular model is a geometric
assumption; it performs no perturbation analysis and makes no stability claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Protocol


ORBIT_FRAME = "J2000"
ORBIT_ORIGIN = "MOON"
ORBIT_MODEL_LABEL = "assumed kinematic circular orbit; not SPICE-propagated or stability-validated"
DEFAULT_MOON_REFERENCE_RADIUS_KM = 1_737.4
DEFAULT_MOON_MU_KM3_S2 = 4_902.800066


class _UtcToEt(Protocol):
    def utc_to_et(self, utc_timestamp: str) -> float: ...


@dataclass(frozen=True)
class LunarOrbitPosition:
    """Moon-centred spacecraft position in km in the named reference frame."""

    utc_timestamp: str
    epoch_et_s: float
    et_s: float
    position_km: tuple[float, float, float]
    frame: str = ORBIT_FRAME
    origin: str = ORBIT_ORIGIN
    model: str = ORBIT_MODEL_LABEL


@dataclass(frozen=True)
class CircularLunarOrbit:
    """Circular two-body orbit defined by explicit angular elements.

    ``altitude_km`` is above ``reference_radius_km``. ``inclination_deg``,
    ``ascending_node_longitude_deg`` and ``initial_orbital_phase_deg`` are
    degrees. Epoch conversion and UTC/ET interpretation are delegated to a
    loaded SPICE geometry engine supplied to :meth:`position_at_utc`.
    """

    altitude_km: float
    inclination_deg: float
    ascending_node_longitude_deg: float
    initial_orbital_phase_deg: float
    epoch_utc: str
    reference_radius_km: float = DEFAULT_MOON_REFERENCE_RADIUS_KM
    gravitational_parameter_km3_s2: float = DEFAULT_MOON_MU_KM3_S2

    def __post_init__(self) -> None:
        for name in (
            "altitude_km", "inclination_deg", "ascending_node_longitude_deg",
            "initial_orbital_phase_deg", "reference_radius_km", "gravitational_parameter_km3_s2",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
            object.__setattr__(self, name, value)
        if self.altitude_km <= 0:
            raise ValueError("altitude_km must be greater than zero.")
        if not 0.0 <= self.inclination_deg <= 180.0:
            raise ValueError("inclination_deg must be in [0, 180].")
        if self.reference_radius_km <= 0 or self.gravitational_parameter_km3_s2 <= 0:
            raise ValueError("Moon reference radius and gravitational parameter must be positive.")
        try:
            epoch = datetime.fromisoformat(self.epoch_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("epoch_utc must be a valid ISO-8601 UTC timestamp.") from exc
        if epoch.tzinfo is None:
            raise ValueError("epoch_utc must include a UTC offset or Z suffix.")

    @property
    def orbital_radius_km(self) -> float:
        """Circular radius from Moon centre in km."""
        return self.reference_radius_km + self.altitude_km

    @property
    def period_s(self) -> float:
        """Ideal two-body circular period in seconds."""
        return 2.0 * math.pi * math.sqrt(
            self.orbital_radius_km**3 / self.gravitational_parameter_km3_s2
        )

    def position_at_et(self, et_s: float, epoch_et_s: float) -> tuple[float, float, float]:
        """Return Moon-centred position in J2000 at ET, in km."""
        et = float(et_s)
        epoch_et = float(epoch_et_s)
        if not math.isfinite(et) or not math.isfinite(epoch_et):
            raise ValueError("ET values must be finite seconds.")
        elapsed_s = et - epoch_et
        phase = math.radians(self.initial_orbital_phase_deg) + 2.0 * math.pi * elapsed_s / self.period_s
        inclination = math.radians(self.inclination_deg)
        node = math.radians(self.ascending_node_longitude_deg)
        # R3(node) R1(inclination) applied to a circular perifocal position.
        x_orbit = self.orbital_radius_km * math.cos(phase)
        y_orbit = self.orbital_radius_km * math.sin(phase)
        x = math.cos(node) * x_orbit - math.sin(node) * math.cos(inclination) * y_orbit
        y = math.sin(node) * x_orbit + math.cos(node) * math.cos(inclination) * y_orbit
        z = math.sin(inclination) * y_orbit
        return (float(x), float(y), float(z))

    def position_at_utc(self, utc_timestamp: str, spice_engine: _UtcToEt) -> LunarOrbitPosition:
        """Return a timestamped assumed orbit position, using SPICE UTC→ET only.

        The supplied engine must have its kernels loaded. SPICE provides time
        conversion here; it does not propagate this assumed orbit.
        """
        if not hasattr(spice_engine, "utc_to_et"):
            raise TypeError("spice_engine must provide utc_to_et(utc_timestamp).")
        et = float(spice_engine.utc_to_et(utc_timestamp))
        epoch_et = float(spice_engine.utc_to_et(self.epoch_utc))
        position = self.position_at_et(et, epoch_et)
        return LunarOrbitPosition(
            utc_timestamp=utc_timestamp,
            epoch_et_s=epoch_et,
            et_s=et,
            position_km=position,
        )
