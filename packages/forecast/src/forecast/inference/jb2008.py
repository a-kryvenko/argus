from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np

# pyatmos otherwise downloads IERS files during import. JB2008 only needs the
# bundled Astropy algorithms for the Sun and sidereal time; production can
# provide IERS independently when higher Earth-orientation accuracy is needed.
os.environ.setdefault("ENABLE_IERS_LOAD", "false")

from astropy.coordinates import get_sun
from astropy.time import Time
from astropy.utils import iers
# pyatmos imports its unrelated COESA model, which uses pkg_resources.
# Compatibility is pinned in pyproject; suppress only this known import warning.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore", message=r"pkg_resources is deprecated as an API\.",
        category=UserWarning, module=r"pyatmos\.standardatmos\.coesa76",
    )
    from pyatmos.jb2008.JB2008_subfunc import JB2008
    from pyatmos.utils.utils import ydhms_days

iers.conf.auto_download = False
# Use bundled predictive Earth orientation for the operational density estimate.
iers.conf.auto_max_age = None


@dataclass(frozen=True)
class JB2008Drivers:
    f10: float
    f10_81c: float
    s10: float
    s10_81c: float
    m10: float
    m10_81c: float
    y10: float
    y10_81c: float
    dtc: float


def jb2008_density(
    valid_time: datetime,
    latitude_deg: float,
    longitude_deg: float,
    altitude_km: float,
    drivers: JB2008Drivers,
) -> float:
    """Evaluate JB2008 total neutral mass density in kg/m³."""
    if not 90 <= altitude_km <= 2500:
        raise ValueError("JB2008 altitude must be between 90 and 2500 km")
    if not -90 <= latitude_deg <= 90:
        raise ValueError("latitude_deg must be between -90 and 90")

    time = Time(valid_time, location=(f"{longitude_deg}d", f"{latitude_deg}d"))
    amjd = time.mjd
    yrday = ydhms_days(np.asarray(time.yday.split(":"), dtype=float))
    sun = get_sun(time)
    sun_position = (sun.ra.rad, sun.dec.rad)
    satellite = (
        time.sidereal_time("mean").rad,
        np.deg2rad(latitude_deg),
        altitude_km,
    )

    _, rho = JB2008(
        amjd,
        yrday,
        sun_position,
        satellite,
        drivers.f10,
        drivers.f10_81c,
        drivers.s10,
        drivers.s10_81c,
        drivers.m10,
        drivers.m10_81c,
        drivers.y10,
        drivers.y10_81c,
        drivers.dtc,
    )
    return float(rho)


def jb2008_density_grid(valid_time, latitudes_deg, longitudes_deg, altitudes_km,
                        drivers: JB2008Drivers) -> np.ndarray:
    """Evaluate [altitude, latitude, longitude], sharing astronomy per timestamp.

    EarthLocation and sidereal time retain the scalar wrapper's conventions,
    including latitude-dependent Earth-orientation corrections.
    """
    from astropy import units as u
    from astropy.coordinates import EarthLocation

    latitudes = np.asarray(latitudes_deg, dtype=float)
    longitudes = np.asarray(longitudes_deg, dtype=float)
    altitudes = np.asarray(altitudes_km, dtype=float)
    for values, lower, upper, name in (
        (latitudes, -90, 90, "latitude"),
        (longitudes, -180, 180, "longitude"),
        (altitudes, 90, 2500, "altitude"),
    ):
        if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not ((values >= lower) & (values <= upper)).all():
            raise ValueError(f"Invalid {name} grid")
    scalar_time = Time(valid_time)
    sun = get_sun(scalar_time)
    sun_position = (float(sun.ra.rad), float(sun.dec.rad))
    amjd = float(scalar_time.mjd)
    yrday = float(ydhms_days(np.asarray(scalar_time.yday.split(":"), dtype=float)))
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)
    locations = EarthLocation.from_geodetic(longitude_grid * u.deg, latitude_grid * u.deg)
    position_times = Time(
        np.full(longitude_grid.shape, scalar_time.jd1),
        np.full(longitude_grid.shape, scalar_time.jd2),
        format="jd", scale=scalar_time.scale, location=locations,
    )
    sidereal = position_times.sidereal_time("mean").rad
    latitude_radians = np.deg2rad(latitudes)
    parameters = tuple(float(value) for value in vars(drivers).values())
    densities = np.empty((len(altitudes), len(latitudes), len(longitudes)))
    for a, altitude in enumerate(altitudes):
        for b, latitude in enumerate(latitude_radians):
            for c in range(len(longitudes)):
                satellite = (float(sidereal[b, c]), float(latitude), float(altitude))
                _, rho = JB2008(amjd, yrday, sun_position, satellite, *parameters)
                densities[a, b, c] = rho
    if not np.isfinite(densities).all() or not (densities > 0).all():
        raise RuntimeError("JB2008 produced invalid density samples")
    return densities
