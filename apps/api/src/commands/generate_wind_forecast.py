from _runner import run_command
from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector


def main():
    director = ForecastDirector()
    director.refresh_forecasts([
        ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_QUANTILE),
        ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_THRESHOLD),
    ])

if __name__ == "__main__":
    run_command(main)