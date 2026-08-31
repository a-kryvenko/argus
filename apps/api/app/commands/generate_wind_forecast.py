from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector

from app.commands._runner import run_command
from app.commands._sensor_observations import refresh_sensor_observations


def main() -> None:
    director = ForecastDirector()
    observations = refresh_sensor_observations()
    director.refresh_forecasts(
        [
            ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_QUANTILE),
            ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_THRESHOLD),
        ],
        observations,
    )


if __name__ == "__main__":
    run_command(main)
