from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector

from argus_prophet.commands._runner import run_command
from argus_prophet.commands._sensor_observations import load_sensor_observations


def main(inputs=None, recorder=None) -> None:
    director = ForecastDirector(on_result=recorder.store if recorder else None,
                                on_csv_written=recorder.csv_written if recorder else None)
    observations = inputs.observations if inputs is not None else load_sensor_observations()
    director.refresh_forecasts(
        [
            ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_QUANTILE),
            ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_THRESHOLD),
        ],
        observations,
    )


if __name__ == "__main__":
    run_command(main)
