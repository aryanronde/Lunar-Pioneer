"""Ideal flat-mirror redirection geometry and first-order optical estimates.

Vectors are Moon-centred Cartesian positions in one explicitly tagged frame,
in km. A flat mirror redirects the diverging light from the finite Sun; it
does not focus or concentrate sunlight.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal


PositionFrame = Literal["J2000", "MOON_PA_DE440", "MOON_ME_DE421"]
GEOMETRY_ORIGINS = ("spice-derived", "assumed")
AU_KM = 149_597_870.7
SOLAR_IRRADIANCE_REFERENCE_DISTANCE_KM = AU_KM


def _vec3(values: tuple[float, float, float], label: str) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{label} must have three Cartesian components.")
    vector = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{label} components must be finite.")
    return vector


def _subtract(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(vector: tuple[float, ...]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(vector: tuple[float, ...], label: str) -> tuple[float, float, float]:
    length = _norm(vector)
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError(f"{label} must be nonzero and numerically stable.")
    return tuple(component / length for component in vector)  # type: ignore[return-value]


@dataclass(frozen=True)
class TaggedLunarPosition:
    """Moon-centred position in km with frame and origin explicitly tagged."""

    position_km: tuple[float, float, float]
    frame: PositionFrame
    origin: str = "MOON"

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_km", _vec3(self.position_km, "Position"))
        if self.frame not in ("J2000", "MOON_PA_DE440", "MOON_ME_DE421"):
            raise ValueError("frame must be J2000, MOON_PA_DE440, or MOON_ME_DE421.")
        if self.origin != "MOON":
            raise ValueError("LUMEN positions must be Moon-centred (origin='MOON').")


@dataclass(frozen=True)
class MirrorOpticsAssumptions:
    """Flat rectangular mirror and solar-source assumptions.

    ``solar_irradiance_reference_w_m2`` is the reference flux at
    ``solar_irradiance_reference_distance_km`` and is inverse-square scaled to
    the Sun-to-mirror range when ``solar_irradiance_w_m2`` is omitted. The
    latter is an explicit instantaneous irradiance at the mirror (legacy API
    and time-varying override). Dimensions are metres; omitted aperture
    dimensions imply a square. Reflectivity is uniform, specular and
    wavelength-averaged.
    """

    mirror_area_m2: float
    reflectivity: float
    pointing_error_deg: float = 0.0
    solar_irradiance_w_m2: float | None = None
    solar_radius_km: float = 695_700.0
    solar_irradiance_reference_distance_km: float = SOLAR_IRRADIANCE_REFERENCE_DISTANCE_KM
    solar_irradiance_reference_w_m2: float = 1361.0
    mirror_width_m: float | None = None
    mirror_height_m: float | None = None
    mirror_aperture_u_unit: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        for name in ("mirror_area_m2", "solar_radius_km", "solar_irradiance_reference_distance_km", "solar_irradiance_reference_w_m2"):
            value = float(getattr(self, name))
            lower_ok = value >= 0.0 if name in ("mirror_area_m2", "solar_irradiance_reference_w_m2") else value > 0.0
            if not math.isfinite(value) or not lower_ok:
                raise ValueError(f"{name} must be finite and valid for its physical domain.")
            object.__setattr__(self, name, value)
        if self.solar_irradiance_w_m2 is not None:
            irradiance = float(self.solar_irradiance_w_m2)
            if not math.isfinite(irradiance) or irradiance < 0:
                raise ValueError("solar_irradiance_w_m2 must be finite and nonnegative when supplied.")
            object.__setattr__(self, "solar_irradiance_w_m2", irradiance)
        reflectivity = float(self.reflectivity)
        if not math.isfinite(reflectivity) or not 0.0 <= reflectivity <= 1.0:
            raise ValueError("reflectivity must be in [0, 1].")
        object.__setattr__(self, "reflectivity", reflectivity)
        error = float(self.pointing_error_deg)
        if not math.isfinite(error) or not 0.0 <= error < 45.0:
            raise ValueError("pointing_error_deg (mirror-normal error) must be finite and in [0, 45).")
        object.__setattr__(self, "pointing_error_deg", error)
        if self.solar_radius_km <= 0.0:
            raise ValueError("solar_radius_km must be positive.")
        width, height = self.mirror_width_m, self.mirror_height_m
        if width is None and height is None:
            width = height = math.sqrt(self.mirror_area_m2)
        elif width is None or height is None:
            raise ValueError("Specify both mirror_width_m and mirror_height_m, or neither.")
        width, height = float(width), float(height)
        if self.mirror_area_m2 == 0 and self.mirror_width_m is None and self.mirror_height_m is None:
            width = height = 0.0
        if not all(math.isfinite(v) and v >= 0 for v in (width, height)) or (self.mirror_area_m2 > 0 and min(width, height) == 0):
            raise ValueError("Mirror aperture dimensions must be finite positive metres for nonzero area.")
        if not math.isclose(width * height, self.mirror_area_m2, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Mirror dimensions product must equal mirror_area_m2.")
        object.__setattr__(self, "mirror_width_m", width)
        object.__setattr__(self, "mirror_height_m", height)
        if self.mirror_aperture_u_unit is not None:
            object.__setattr__(self, "mirror_aperture_u_unit", _unit(self.mirror_aperture_u_unit, "Mirror aperture u axis"))


@dataclass(frozen=True)
class MirrorOpticsResult:
    """Ideal geometry and optical estimates; distances km, power W, angles rad."""

    frame: str
    sun_to_mirror_unit: tuple[float, float, float]
    mirror_to_target_unit: tuple[float, float, float]
    ideal_mirror_normal_unit: tuple[float, float, float]
    reflected_unit: tuple[float, float, float]
    reflection_law_residual: float
    sun_mirror_distance_km: float
    mirror_target_distance_km: float
    solar_angular_radius_rad: float
    beam_footprint_diameter_km: float
    pointing_error_deg: float
    pointing_error_displacement_km: float
    incidence_angle_deg: float
    incident_optical_power_w: float
    reflected_optical_power_w: float
    geometry_origin: str
    assumed_parameters: tuple[str, ...]
    solar_irradiance_at_mirror_w_m2: float = 0.0
    projected_illuminated_area_m2: float = 0.0
    mirror_width_m: float = 0.0
    mirror_height_m: float = 0.0
    solar_radius_km: float = 695_700.0
    mirror_to_sun_unit: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mirror_aperture_u_unit: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mirror_aperture_v_unit: tuple[float, float, float] = (0.0, 0.0, 0.0)


def calculate_mirror_optics(
    sun: TaggedLunarPosition,
    mirror: TaggedLunarPosition,
    target: TaggedLunarPosition,
    assumptions: MirrorOpticsAssumptions,
    *,
    geometry_origin: str,
) -> MirrorOpticsResult:
    """Calculate bisector normal, reflection residual, beam and optical power.

    ``sun_to_mirror_unit`` (legacy field name) points from mirror to Sun;
    ``mirror_to_target_unit`` points from mirror to receiver. Photon
    propagation is the negative of the former. The ideal normal bisects these
    outward vectors. The solar angular radius is ``asin(R_sun/r)``. The beam
    bounding diameter adds solar-disc spread to the rectangular aperture's
    diagonal extent; it is not a uniform illuminated circle. Pointing error is
    mirror-normal angular error and causes a 2x reflected-ray deviation.
    """
    _check_positions(sun, mirror, target)
    if not isinstance(assumptions, MirrorOpticsAssumptions):
        raise TypeError("assumptions must be MirrorOpticsAssumptions.")
    if geometry_origin not in GEOMETRY_ORIGINS:
        raise ValueError("geometry_origin must be 'spice-derived' or 'assumed'.")

    sun_delta = _subtract(sun.position_km, mirror.position_km)
    target_delta = _subtract(target.position_km, mirror.position_km)
    source_unit = _unit(sun_delta, "Mirror-to-Sun direction")
    target_unit = _unit(target_delta, "Mirror-to-target direction")
    normal = _unit(
        (source_unit[0] + target_unit[0], source_unit[1] + target_unit[1], source_unit[2] + target_unit[2]),
        "Ideal mirror normal (source and target directions cannot be antiparallel)",
    )
    aperture_candidate = assumptions.mirror_aperture_u_unit
    if aperture_candidate is None:
        ref = (0.0, 0.0, 1.0) if abs(normal[2]) < 0.9 else (0.0, 1.0, 0.0)
        aperture_candidate = _cross3(normal, ref)
    aperture_projected = tuple(aperture_candidate[i] - _dot(aperture_candidate, normal) * normal[i] for i in range(3))
    aperture_u = _unit(aperture_projected, "Mirror aperture u axis must not be parallel to mirror normal")
    aperture_v = _unit(_cross3(normal, aperture_u), "Mirror aperture v axis")
    incoming = tuple(-component for component in source_unit)
    reflected = tuple(
        incoming[index] - 2.0 * _dot(incoming, normal) * normal[index]
        for index in range(3)
    )
    residual = _norm(_subtract(reflected, target_unit))

    sun_range = _norm(sun_delta)
    target_range = _norm(target_delta)
    ratio = assumptions.solar_radius_km / sun_range
    if ratio >= 1.0:
        raise ValueError("Sun range must exceed the assumed solar radius.")
    angular_radius = math.asin(ratio)
    aperture_diagonal_km = math.hypot(assumptions.mirror_width_m, assumptions.mirror_height_m) / 1000.0
    diameter = 2.0 * target_range * math.tan(angular_radius) + aperture_diagonal_km
    # pointing_error_deg is mirror-normal angular error; reflection doubles it.
    pointing_displacement = target_range * math.tan(2.0 * math.radians(assumptions.pointing_error_deg))
    incidence_cos = max(0.0, min(1.0, _dot(source_unit, normal)))
    incidence = math.degrees(math.acos(incidence_cos))
    irradiance_at_mirror = (
        assumptions.solar_irradiance_w_m2
        if assumptions.solar_irradiance_w_m2 is not None
        else assumptions.solar_irradiance_reference_w_m2
        * (assumptions.solar_irradiance_reference_distance_km / sun_range) ** 2
    )
    incident_power = irradiance_at_mirror * assumptions.mirror_area_m2 * incidence_cos
    reflected_power = incident_power * assumptions.reflectivity

    return MirrorOpticsResult(
        frame=mirror.frame,
        sun_to_mirror_unit=source_unit,
        mirror_to_target_unit=target_unit,
        ideal_mirror_normal_unit=normal,
        reflected_unit=reflected,
        reflection_law_residual=residual,
        sun_mirror_distance_km=sun_range,
        mirror_target_distance_km=target_range,
        solar_angular_radius_rad=angular_radius,
        beam_footprint_diameter_km=diameter,
        pointing_error_deg=assumptions.pointing_error_deg,
        pointing_error_displacement_km=pointing_displacement,
        incidence_angle_deg=incidence,
        incident_optical_power_w=incident_power,
        reflected_optical_power_w=reflected_power,
        geometry_origin=geometry_origin,
        assumed_parameters=(
            "orbit geometry unless geometry_origin is spice-derived",
            "solar_radius_km",
            "solar_irradiance_w_m2 (explicit local override) or reference irradiance and distance",
            "mirror_area_m2",
            "reflectivity",
            "pointing_error_deg",
            "flat mirror; no focusing",
            "uniform rectangular aperture with uniform specular reflectivity",
            "irradiance inverse-square scaled from explicit reference distance",
            "rectangular aperture in supplied in-plane orientation; default orientation is deterministic if omitted",
        ),
        solar_irradiance_at_mirror_w_m2=irradiance_at_mirror,
        projected_illuminated_area_m2=assumptions.mirror_area_m2 * incidence_cos,
        mirror_width_m=assumptions.mirror_width_m,
        mirror_height_m=assumptions.mirror_height_m,
        solar_radius_km=assumptions.solar_radius_km,
        mirror_to_sun_unit=source_unit,
        mirror_aperture_u_unit=aperture_u,
        mirror_aperture_v_unit=aperture_v,
    )


def _check_positions(*positions: TaggedLunarPosition) -> None:
    if any(not isinstance(position, TaggedLunarPosition) for position in positions):
        raise TypeError("Sun, mirror and target must be TaggedLunarPosition values.")
    frames = {position.frame for position in positions}
    origins = {position.origin for position in positions}
    if len(frames) != 1:
        raise ValueError("Sun, mirror and target positions must use the same coordinate frame.")
    if origins != {"MOON"}:
        raise ValueError("Sun, mirror and target positions must share the Moon-centred origin.")


def _cross3(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])
