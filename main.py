"""Run the V0.1 Sun/Earth geometry calculation for one UTC instant."""

from core import LocalHorizonFrame, LunarSite, SpiceGeometryEngine


def main() -> None:
    # Same documented reference instant as the integration test.
    utc_timestamp = "2026-10-05T00:00:00"
    site = LunarSite(latitude_deg=-85.99, longitude_deg=357.07)
    with SpiceGeometryEngine() as engine:
        et = engine.utc_to_et(utc_timestamp)
        sun_j2000 = engine.sun_position_km(et)
        earth_j2000 = engine.earth_position_km(et)
        sun_fixed = engine.j2000_to_moon_fixed(sun_j2000, et)
        earth_fixed = engine.j2000_to_moon_fixed(earth_j2000, et)
        local_frame = LocalHorizonFrame.at_site(site)
        sun_azimuth, sun_elevation = local_frame.direction_angles(sun_fixed)
        earth_azimuth, earth_elevation = local_frame.direction_angles(earth_fixed)

        # Preserve the V0.1 inertial outputs and report the added frame results.
        print(f"UTC: {utc_timestamp} (UTC scale)")
        print(f"ET (TDB seconds past J2000): {et:.6f}")
        print(f"Sun relative to Moon, J2000 (km): {sun_j2000}")
        print(f"Earth relative to Moon, J2000 (km): {earth_j2000}")
        print(f"Sun relative to Moon, {engine.MOON_FIXED_FRAME} (km): {sun_fixed}")
        print(f"Earth relative to Moon, {engine.MOON_FIXED_FRAME} (km): {earth_fixed}")
        print(f"Geodetic site (latitude, east longitude; degrees): ({site.latitude_deg}, {site.longitude_deg})")
        print(f"Sun geometric horizon: azimuth={sun_azimuth:.6f}°, elevation={sun_elevation:.6f}°")
        print(f"Earth geometric horizon: azimuth={earth_azimuth:.6f}°, elevation={earth_elevation:.6f}°")


if __name__ == "__main__":
    main()
