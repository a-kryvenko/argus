from _runner import run_command
from _sensor_observations import refresh_sensor_observations
from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector


def main():
    director = ForecastDirector()
    observations = refresh_sensor_observations()
    director.refresh_forecasts([
        ForecastServiceRegistry.get(ForecastService.HMF_TOTAL_THRESHOLD),
        ForecastServiceRegistry.get(ForecastService.HMF_SOUTH_THRESHOLD),
    ], observations)

if __name__ == "__main__":
    run_command(main)
