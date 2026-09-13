import logging

from argus_prophet.commands.generate_atmospheric_density_forecast import main as generate_density
from forecast.exceptions import ArtifactNotReadyError

from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector

from argus_prophet.commands._runner import run_command
from argus_prophet.observations import load_inputs


def main() -> None:
    director = ForecastDirector()
    inputs = load_inputs()
    observations = inputs.observations
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
        generate_density(inputs)
    except ArtifactNotReadyError as exc:
        logging.getLogger(__name__).warning("Density forecast unavailable: %s", exc)


if __name__ == "__main__":
    run_command(main)
