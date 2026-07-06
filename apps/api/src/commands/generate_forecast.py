from forecast.ForecastDirector import ForecastDirector
from forecast.inference.KpForecastService import KpForecastService
from forecast.inference.PlasmaStateForecastService import PlasmaStateForecastService
from forecast_core.inference.HmfForecastService import HmfForecastService

from _runner import run_command

def main():
    director = ForecastDirector()
    director.refresh_forecasts([KpForecastService, PlasmaStateForecastService, HmfForecastService])

if __name__ == "__main__":
    run_command(main)