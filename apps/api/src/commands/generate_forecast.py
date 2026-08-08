from _runner import run_command
from forecast.forecast_services import ForecastService, ForecastServiceRegistry
from forecast.ForecastDirector import ForecastDirector


def main():
    director = ForecastDirector()
    director.refresh_forecasts([
        ForecastServiceRegistry.get(ForecastService.KP_INDEX_THRESHOLD),
        ForecastServiceRegistry.get(ForecastService.AP_INDEX_QUANTILE),
        ForecastServiceRegistry.get(ForecastService.DST_QUANTILE),
        ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_QUANTILE),
        ForecastServiceRegistry.get(ForecastService.PLASMA_SPEED_THRESHOLD),
        ForecastServiceRegistry.get(ForecastService.PLASMA_DENSITY_QUANTILE),
        ForecastServiceRegistry.get(ForecastService.HMF_TOTAL_THRESHOLD),
        ForecastServiceRegistry.get(ForecastService.HMF_SOUTH_THRESHOLD),
    ])

if __name__ == "__main__":
    run_command(main)