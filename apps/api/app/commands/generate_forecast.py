import logging

from app.commands.generate_atmospheric_density_forecast import main as generate_density
from app.services.forecast_products import ArtifactNotReadyError

from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector

from app.commands._runner import run_command
from app.commands._sensor_observations import load_sensor_observations


def main() -> None:
    director = ForecastDirector()
    observations = load_sensor_observations()
    director.refresh_forecasts(
        [
            ForecastServiceRegistry.get(ForecastService.KP_INDEX_THRESHOLD),
            ForecastServiceRegistry.get(ForecastService.AP_INDEX_QUANTILE),
            ForecastServiceRegistry.get(ForecastService.DST_QUANTILE),
            ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_QUANTILE),
            ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_THRESHOLD),
            ForecastServiceRegistry.get(ForecastService.PLASMA_DENSITY_QUANTILE),
            ForecastServiceRegistry.get(ForecastService.HMF_TOTAL_THRESHOLD),
            ForecastServiceRegistry.get(ForecastService.HMF_SOUTH_THRESHOLD),
        ],
        observations,
    )

    try:
        generate_density()
    except ArtifactNotReadyError as exc:
        logging.getLogger(__name__).warning("Density forecast unavailable: %s", exc)


if __name__ == "__main__":
    run_command(main)
