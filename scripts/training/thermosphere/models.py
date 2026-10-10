"""Research adapters; no production registry or serving changes."""
import ctypes
from pathlib import Path

import numpy as np
import pandas as pd

AP = np.array([0, 2, 3, 4, 5, 6, 7, 9, 12, 15, 18, 22, 27, 32, 39,
               48, 56, 67, 80, 94, 111, 132, 154, 179, 207, 236, 300, 400.])
KP = np.arange(28) / 3


class DTM2020:
    """Single-threaded interface to unchanged pinned CNES Fortran routines."""
    def __init__(self, directory):
        directory = Path(directory)
        self.lib = ctypes.CDLL(str(directory / "libdtm2020.so"))
        pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
        self.lib.argus_dtm_init.argtypes = [pointer]
        self.lib.argus_dtm_init.restype = None
        self.lib.argus_dtm_eval.argtypes = [pointer, pointer]
        self.lib.argus_dtm_eval.restype = None
        # Fixed-width public coefficient format: index, nine (value, uncertainty) pairs.
        lines = (directory / "upstream/DTM_2020_F107_Kp.dat").read_text().splitlines()
        count = int(lines[1][:4])
        coefficients = np.zeros((96, 9), order="F")
        if count != 95:
            raise ValueError("Unexpected DTM2020 coefficient count")
        for i, line in enumerate(lines[2:2 + count]):
            if int(line[:4]) != i + 1:
                raise ValueError("Invalid DTM coefficient ordering")
            for j in range(9):
                coefficients[i, j] = float(line[4 + 22*j:17 + 22*j])
        self.lib.argus_dtm_init(coefficients)

    def evaluate(self, day, altitude_km, latitude_rad, longitude_rad, lst_rad,
                 f107, f107_mean, kp, kp_mean):
        values = np.array([day, altitude_km, latitude_rad, longitude_rad, lst_rad,
                           f107, f107_mean, kp, kp_mean], dtype=np.float64)
        if (not np.isfinite(values).all() or not 120 < altitude_km <= 1500
                or not -np.pi/2 <= latitude_rad <= np.pi/2
                or not 0 <= kp <= 9 or not 0 <= kp_mean <= 9
                or f107 <= 0 or f107_mean <= 0):
            raise ValueError("Invalid DTM2020 inputs")
        result = np.empty(3, dtype=np.float64)
        self.lib.argus_dtm_eval(values, result)
        if not np.isfinite(result).all() or result[0] <= 0:
            raise ValueError("Invalid DTM2020 output")
        return result


def causal_interpolate(series, times, max_age):
    """Interpolate known records; hold latest only within explicit freshness bound."""
    series = series.sort_index()
    times = pd.DatetimeIndex(times)
    if (series.empty or times.min() < series.index[0]
            or times.max() - series.index[-1] > max_age
            or not np.isfinite(series).all()):
        raise ValueError("Insufficient interpolation history")
    # Subtract origin before float conversion to retain timestamp precision.
    origin = series.index[0].value
    return np.interp((times.asi8-origin)/1e9, (series.index.asi8-origin)/1e9, series.values)


def dtm_snapshot(measurements, issue_time):
    """Causal operational approximation to documented DTM input interpolation.

    Archive interval labels remain start times; available_at enforces closed bins.
    F107 is daily, labelled noon and available after day end. Mean uses last 81
    completed days; no future-centered mean. Interpolation never invents the next
    unavailable bin. Latest known ap is held across the open bin.
    """
    from forecast_core.models.jb2008_drivers import fill_background_gaps
    issue = pd.Timestamp(issue_time)
    known = measurements.loc[(measurements.available_at <= issue)
                             & (measurements.observed_at <= issue)]
    f = known.loc[known.metric.eq("f10_7")].set_index("observed_at").value.sort_index()
    if f.empty or issue-f.index[-1] > pd.Timedelta(hours=48):
        raise ValueError("Missing recent F107")
    daily = f.resample("D").last()
    daily = daily.reindex(pd.date_range(daily.index[-1]-pd.Timedelta(days=80), daily.index[-1]))
    daily, filled = fill_background_gaps(daily)
    if not np.isfinite(daily).all() or not daily.gt(0).all():
        raise ValueError("DTM requires 81 valid F107 daily values")
    flux = causal_interpolate(f, [issue-pd.Timedelta(days=1)], pd.Timedelta(days=1))[0]
    ap = known.loc[known.metric.eq("ap")].set_index("observed_at").value.sort_index()
    ap = ap.loc[ap.index >= issue-pd.Timedelta(hours=30)]
    if (len(ap) < 8 or not ap.between(0, 400).all()
            or not ap.index.to_series().diff().dropna().eq(pd.Timedelta(hours=3)).all()):
        raise ValueError("Incomplete DTM ap history")
    times = pd.DatetimeIndex([issue-pd.Timedelta(hours=3),
                             *pd.date_range(issue-pd.Timedelta(hours=23), issue, freq="h")])
    values = causal_interpolate(ap, times, pd.Timedelta(hours=6))
    kp = np.interp(values, AP, KP)
    return {"f107": float(flux), "f107_mean": float(daily.mean()),
            "kp": float(kp[0]), "kp_mean": float(kp[1:].mean()),
            "interpolated_days": filled}


def jb_geometry(observations):
    """Vectorized astronomy, identical conventions to the production JB wrapper."""
    # Import production wrapper first: disables pyatmos network IERS loading.
    from forecast_core.models import jb2008  # noqa: F401
    from astropy import units as u
    from astropy.coordinates import EarthLocation, get_sun
    from astropy.time import Time
    locations = EarthLocation.from_geodetic(observations.longitude_deg.to_numpy()*u.deg,
                                            observations.latitude_deg.to_numpy()*u.deg)
    times = Time(observations.valid_time.to_numpy(dtype="datetime64[ns]"), location=locations)
    sun = get_sun(times)
    days = observations.valid_time.dt.dayofyear.to_numpy() + (
        observations.valid_time-observations.valid_time.dt.floor("D")).dt.total_seconds().to_numpy()/86400
    return np.column_stack([times.mjd, days, sun.ra.rad, sun.dec.rad,
                            times.sidereal_time("mean").rad,
                            np.deg2rad(observations.latitude_deg), observations.altitude_km])


def jb_evaluate(geometry, drivers):
    from forecast_core.models.jb2008 import JB2008
    mjd, day, ra, dec, sidereal, lat, alt = geometry
    _, rho = JB2008(mjd, day, (ra, dec), (sidereal, lat, alt),
                   drivers["f10_7"], drivers["f10_7_81mean"],
                   drivers["s10"], drivers["s10_81mean"],
                   drivers["m10"], drivers["m10_81mean"],
                   drivers["y10"], drivers["y10_81mean"], drivers["dtc"])
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("Invalid JB2008 output")
    return float(rho)
