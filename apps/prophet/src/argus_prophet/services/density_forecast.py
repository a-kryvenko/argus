"""Calculate atmospheric density in memory from an explicit input snapshot."""
import logging
from datetime import datetime
from time import perf_counter

from common.schemas.forecast_inputs import ForecastInputs
from argus_prophet.services.density_observations import load_density_drivers
from forecast.calculation import ForecastResult
from forecast_core.api import (
    AtmosphericDensityForecastService,
)


logger = logging.getLogger(__name__)


def calculate_density(*, inputs: ForecastInputs, issue_time: datetime) -> ForecastResult:
    started = perf_counter()
    service = AtmosphericDensityForecastService()
    drivers = load_density_drivers(inputs, issue_time)
    logger.info('JB2008: drivers ready in %.1fs; calculating %d hourly grids',
                perf_counter() - started, len(drivers))
    frame = service.forecast_grid(
        drivers=drivers,
        progress=lambda done, total: logger.info('JB2008: grid %d/%d, elapsed %.1fs',
                                                 done, total, perf_counter() - started),
    )
    frame.insert(0, "issue_time", issue_time)
    frame.insert(
        2,
        "lead_hours",
        ((frame["valid_time"] - issue_time).dt.total_seconds() / 3600).astype(int),
    )

    logger.info('Calculated %d JB2008 density rows in %.1fs', len(frame), perf_counter() - started)
    return ForecastResult(service.registry_name, frame,
                          {"backend": "forecast_core", "registry_name": service.registry_name,
                           "issue_time": issue_time.isoformat()})
