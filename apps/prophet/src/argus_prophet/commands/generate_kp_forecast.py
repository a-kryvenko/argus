from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector

from argus_prophet.commands._runner import run_command
from argus_prophet.commands._sensor_observations import load_sensor_observations


def main(inputs=None, recorder=None) -> None:
    director = ForecastDirector(on_result=recorder.store if recorder else None,
                                publish_csv=recorder is None)
    observations = inputs.observations if inputs is not None else load_sensor_observations()
    director.refresh_forecasts(
        [ForecastServiceRegistry.get(ForecastService.KP_INDEX_THRESHOLD),
         ForecastServiceRegistry.get(ForecastService.AP_INDEX_QUANTILE)],
        observations,
    )


if __name__ == "__main__":
    run_command(main)
