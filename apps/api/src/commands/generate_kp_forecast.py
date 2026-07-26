from _runner import run_command
from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector


def main():
    director = ForecastDirector()
    director.refresh_forecasts([
        ForecastServiceRegistry.get(ForecastService.KP_INDEX_THRESHOLD),
    ])

if __name__ == "__main__":
    run_command(main)