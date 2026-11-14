from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
import math
import unittest

import numpy as np

from core.lumen.lumen_optics import MirrorOpticsAssumptions
from core.lumen.lumen_orbit import CircularLunarOrbit
from core.lumen.lumen_visibility import SiteHorizonProfile
from core.lumen.station_site import (
    CandidateSite,
    CriterionAssessment,
    CriterionStatus as Status,
    RoverAccessPlan,
    SITE06_NOBILE_RIM_1,
    SITE07_PEAK_NEAR_SHACKLETON_EXPERIMENTAL,
    SiteFeasibilityReport,
    assess_reflector_visibility,
    assess_solar_illumination,
    assess_terrain,
    candidate_me_position,
    compare_site_reports,
    combine_criteria,
)
from core.terrain_provider import (
    MOON_ME_DE421,
    MOON_REFERENCE_RADIUS_M,
    TaggedMoonPosition,
    TerrainMetadata,
    TerrainWindow,
)


class FakeProvider:
    def __init__(self, *, nodata_at=None, outside=False):
        self.x = np.arange(-45.0, 46.0, 10.0)
        self.y = np.arange(45.0, -46.0, -10.0)
        self.z = np.tile(0.1 * self.x, (len(self.y), 1))
        self.nodata = np.zeros(self.z.shape, dtype=bool)
        if nodata_at is not None:
            self.nodata[nodata_at] = True
        self.outside = outside
        self.metadata = TerrainMetadata(
            crs="EPSG:4326", projection="test", affine=(10, 0, -50, 0, -10, 50),
            width=10, height=10, pixel_size_m=(10, 10), nodata=-9999,
            vertical_reference="test datum", source_frame=MOON_ME_DE421,
            bounds_m=(-50, -50, 50, 50), pixel_registration="AREA",
        )

    def site_position_to_dem_xy(self, position: TaggedMoonPosition):
        assert position.frame == MOON_ME_DE421
        return (200.0, 0.0) if self.outside else (0.0, 0.0)

    def read_window(self, center_x_m, center_y_m, width_px, height_px):
        return TerrainWindow(
            center_x_m, center_y_m, self.x, self.y, self.z, self.nodata, 0, 0
        )


class FakeGeometry:
    sun = (200_000.0, 0.0, 0.0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def utc_to_et(self, utc_timestamp):
        value = datetime.fromisoformat(utc_timestamp.replace("Z", "+00:00"))
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()

    def sun_position_km(self, et):
        return self.sun

    def j2000_to_moon_fixed(self, vector, et):
        return tuple(vector)


class FakeBridge:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def vector_dem_to_geometry(self, vector, et):
        return tuple(vector)


def site(**changes):
    args = dict(
        identifier="T", name="Test site", latitude_deg=0.0, longitude_east_deg=0.0,
        coordinate_frame=MOON_ME_DE421, coordinate_provenance="synthetic test coordinates",
        elevation_m=None, elevation_source=None, terrain_coverage_radius_m=None,
        terrain_resolution_m=None,
    )
    args.update(changes)
    return CandidateSite(**args)


def horizon(profile=(0.0,) * 360, *, status="complete", range_km=20.0):
    radius = MOON_REFERENCE_RADIUS_M / 1000.0
    return SiteHorizonProfile(
        profile_deg=profile,
        site_position_km=(radius, 0.0, 0.0),
        frame="MOON_PA_DE440",
        east_axis=(0.0, 1.0, 0.0), north_axis=(0.0, 0.0, 1.0), up_axis=(1.0, 0.0, 0.0),
        range_km=range_km, status=status, site_name="Test site",
    )


class StationSiteTests(unittest.TestCase):
    def test_known_candidates_preserve_published_coordinates_and_experimental_site07(self):
        self.assertEqual((SITE06_NOBILE_RIM_1.latitude_deg, SITE06_NOBILE_RIM_1.longitude_east_deg), (-85.4381, 37.36667))
        self.assertIsNone(SITE06_NOBILE_RIM_1.elevation_m)
        self.assertIsNone(SITE06_NOBILE_RIM_1.terrain_resolution_m)
        self.assertTrue(SITE07_PEAK_NEAR_SHACKLETON_EXPERIMENTAL.experimental)
        self.assertIsNone(SITE07_PEAK_NEAR_SHACKLETON_EXPERIMENTAL.source_version_consistent)
        with self.assertRaises(FrozenInstanceError):
            SITE06_NOBILE_RIM_1.name = "selected"


    def test_coordinate_frame_validation_and_reference_sphere_does_not_fill_elevation(self):
        with self.assertRaises(ValueError):
            site(coordinate_frame="J2000")
        vector = candidate_me_position(SITE06_NOBILE_RIM_1)
        self.assertTrue(math.isclose(math.sqrt(sum(value * value for value in vector)), MOON_REFERENCE_RADIUS_M / 1000.0))
        self.assertIsNone(SITE06_NOBILE_RIM_1.elevation_m)


    def test_terrain_coverage_slope_roughness_and_explicit_route(self):
        plan = RoverAccessPlan(operating_area_radius_m=30, route_points_xy_m=((0, 0), (20, 0)), corridor_width_m=10, maximum_slope_deg=6, route_name="supplied route")
        assessment = assess_terrain(site(), FakeProvider(), evaluation_radius_m=20, maximum_station_slope_deg=6, rover_access_plan=plan)
        self.assertEqual(assessment.coverage.status, Status.PASS)
        self.assertEqual(assessment.slope.status, Status.PASS)
        self.assertEqual(assessment.rover_access.status, Status.PASS)
        self.assertAlmostEqual(assessment.slope_max_deg, math.degrees(math.atan(0.1)))
        self.assertAlmostEqual(assessment.roughness_rms_m, 0.0)
        self.assertEqual(assessment.route_covered_fraction, 1.0)


    def test_nodata_and_missing_slope_neighbors_remain_unknown(self):
        assessment = assess_terrain(site(), FakeProvider(nodata_at=(4, 4)), evaluation_radius_m=20, maximum_station_slope_deg=10)
        self.assertEqual(assessment.coverage.status, Status.UNKNOWN)
        self.assertGreater(assessment.nodata_sample_count, 0)
        self.assertLess(assessment.slope_sample_count, assessment.valid_sample_count)


    def test_dem_boundary_is_not_treated_as_covered(self):
        assessment = assess_terrain(site(), FakeProvider(outside=True), evaluation_radius_m=10)
        self.assertEqual(assessment.coverage.status, Status.FAIL)
        self.assertIsNone(assessment.coverage_fraction)


    def test_solar_visibility_requires_horizon_and_limited_range_stays_unknown(self):
        kwargs = dict(geometry_engine_factory=FakeGeometry, frame_bridge_factory=FakeBridge)
        missing = assess_solar_illumination(site(), "2026-01-01T00:00:00Z", 60, 30, None, **kwargs)
        self.assertEqual(missing.criterion.status, Status.UNKNOWN)
        self.assertTrue(all(item.status == Status.UNKNOWN for item in missing.samples))
        limited = assess_solar_illumination(site(), "2026-01-01T00:00:00Z", 60, 30, horizon(status="limited-range", range_km=1), **kwargs)
        self.assertEqual(limited.criterion.status, Status.UNKNOWN)
        self.assertEqual(limited.terrain_radius_km, 1)
        self.assertAlmostEqual(limited.samples[0].sun_elevation_deg, 90.0)
        complete = assess_solar_illumination(site(), "2026-01-01T00:00:00Z", 60, 30, horizon(), **kwargs)
        self.assertEqual(complete.criterion.status, Status.PASS)
        self.assertTrue(complete.visibility_intervals)


    def test_known_horizon_block_produces_fail(self):
        profile = [0.0] * 360
        profile[0] = 90.0
        result = assess_solar_illumination(
            site(), "2026-01-01T00:00:00Z", 30, 30, horizon(tuple(profile)),
            geometry_engine_factory=FakeGeometry, frame_bridge_factory=FakeBridge,
        )
        self.assertEqual(result.criterion.status, Status.FAIL)
        self.assertLessEqual(result.samples[0].solar_clearance_deg, 0)


    def test_reflector_missing_or_limited_terrain_evidence_is_unknown(self):
        orbit = CircularLunarOrbit(altitude_km=100, inclination_deg=0, ascending_node_longitude_deg=0,
                                   initial_orbital_phase_deg=0, epoch_utc="2026-01-01T00:00:00Z")
        optics = MirrorOpticsAssumptions(mirror_area_m2=10, reflectivity=0.8)
        kwargs = dict(receiver_normal_local_enu=(0, 0, 1), geometry_engine_factory=FakeGeometry, frame_bridge_factory=FakeBridge)
        missing = assess_reflector_visibility(site(), orbit, "2026-01-01T00:00:00Z", 10, 10, None, optics, **kwargs)
        self.assertEqual(missing.criterion.status, Status.UNKNOWN)
        self.assertTrue(missing.orbit_model.startswith("assumed kinematic"))
        limited = assess_reflector_visibility(site(), orbit, "2026-01-01T00:00:00Z", 10, 10, horizon(status="limited-range", range_km=1), optics, **kwargs)
        self.assertEqual(limited.criterion.status, Status.UNKNOWN)
        self.assertIn(limited.samples[0].terrain_status, {"unknown-beyond-limited-range", "limited-range-clearance-only"})


    def test_lunar_occultation_is_reported_as_reflector_failure(self):
        # Put reflector on the far side of the Moon from a site at +X; the ray is occulted.
        orbit = CircularLunarOrbit(altitude_km=100, inclination_deg=0, ascending_node_longitude_deg=0,
                                   initial_orbital_phase_deg=180, epoch_utc="2026-01-01T00:00:00Z")
        result = assess_reflector_visibility(
            site(), orbit, "2026-01-01T00:00:00Z", 1, 1, horizon(),
            MirrorOpticsAssumptions(mirror_area_m2=10, reflectivity=0.8),
            receiver_normal_local_enu=(0, 0, 1), geometry_engine_factory=FakeGeometry,
            frame_bridge_factory=FakeBridge,
        )
        self.assertTrue(result.samples[0].lunar_occulted)
        self.assertEqual(result.samples[0].status, Status.FAIL)


    def test_route_with_nodata_corridor_is_unknown_and_no_implicit_route(self):
        no_plan = assess_terrain(site(), FakeProvider(), evaluation_radius_m=20)
        self.assertEqual(no_plan.rover_access.status, Status.UNKNOWN)
        plan = RoverAccessPlan(30, ((0, 0), (20, 0)), 20, 10, "corridor with gap")
        result = assess_terrain(site(), FakeProvider(nodata_at=(4, 4)), evaluation_radius_m=20, rover_access_plan=plan)
        self.assertEqual(result.rover_access.status, Status.UNKNOWN)


    def test_pass_fail_unknown_combination_comparison_and_report_consistency(self):
        def criterion(name, status):
            return CriterionAssessment(name, status)
        self.assertEqual(combine_criteria((criterion("known", Status.PASS), criterion("missing", Status.UNKNOWN))), Status.UNKNOWN)
        self.assertEqual(combine_criteria((criterion("bad", Status.FAIL), criterion("missing", Status.UNKNOWN))), Status.FAIL)
        passing = SiteFeasibilityReport(site(), *(criterion(name, Status.PASS) for name in (
            "solar", "reflector", "terrain", "slope", "access")), overall_status=Status.PASS)
        unknown_mixed_with_fail = SiteFeasibilityReport(site(identifier="U"),
            criterion("solar", Status.FAIL), criterion("reflector", Status.UNKNOWN),
            criterion("terrain", Status.PASS), criterion("slope", Status.PASS), criterion("access", Status.PASS),
            overall_status=Status.FAIL)
        comparison = compare_site_reports((passing, unknown_mixed_with_fail))
        self.assertFalse(comparison.selection_made)
        self.assertEqual(comparison.comparison_status, Status.UNKNOWN)
        with self.assertRaises(ValueError):
            SiteFeasibilityReport(site(), *(criterion(name, Status.PASS) for name in (
                "solar", "reflector", "terrain", "slope", "access")), overall_status=Status.UNKNOWN)
