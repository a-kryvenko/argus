from _runner import run_command
from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector


def main():
    director = ForecastDirector()
    director.refresh_forecast([
        ForecastServiceRegistry.get(ForecastService.HMF_TOTAL_THRESHOLD),
        ForecastServiceRegistry.get(ForecastService.HMF_SOUTH_THRESHOLD),
    ])

if __name__ == "__main__":
    run_command(main)