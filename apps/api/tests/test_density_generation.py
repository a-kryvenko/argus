from datetime import UTC, datetime

import pandas as pd

from forecast.inference.AtmosphericDensityForecastService import AtmosphericDensityForecastService
from app.services.density_observations import observed_driver_frame
from test_density_observations import observations, ISSUE


def test_internal_observations_produce_finite_density_grid():
    drivers = observed_driver_frame(observations(), ISSUE).iloc[:2]
    grid = AtmosphericDensityForecastService().forecast_grid(
        drivers, altitudes_km=[400], latitudes_deg=[0], longitudes_deg=[0, 90])
    assert len(grid) == 2
    assert grid.rho_kg_m3.gt(0).all()
    assert grid.rho_lon_p10_kg_m3.le(grid.rho_kg_m3).all()
    assert grid.rho_lon_p90_kg_m3.ge(grid.rho_kg_m3).all()
    assert grid.driver_mode.eq('observed_persistence').all()
    assert grid.observed_at.eq(pd.Timestamp(ISSUE) - pd.Timedelta(days=5)).all()


def test_batch_density_matches_scalar_reference():
    import numpy as np
    from forecast.inference.jb2008 import JB2008Drivers, jb2008_density, jb2008_density_grid
    inputs = JB2008Drivers(140., 125., 115., 110., 120., 105., 135., 112., 80.)
    latitudes, longitudes, altitudes = [-90., -22., 45., 90.], [-180., -45., 0., 135.], [200., 400., 800.]
    for valid_time in (datetime(2022, 6, 19, 18, 35, tzinfo=UTC), ISSUE):
        batch = jb2008_density_grid(valid_time, latitudes, longitudes, altitudes, inputs)
        scalar = np.array([[[jb2008_density(valid_time, lat, lon, alt, inputs)
                             for lon in longitudes] for lat in latitudes] for alt in altitudes])
        np.testing.assert_allclose(batch, scalar, rtol=1e-10, atol=0)


def test_grid_reports_completed_hours():
    drivers = observed_driver_frame(observations(), ISSUE).iloc[:2]
    progress = []
    AtmosphericDensityForecastService().forecast_grid(
        drivers, altitudes_km=[400], latitudes_deg=[0], longitudes_deg=[0],
        progress=lambda done, total: progress.append((done, total)))
    assert progress == [(1, 2), (2, 2)]
