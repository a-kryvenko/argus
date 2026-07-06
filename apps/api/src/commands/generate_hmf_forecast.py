from forecast.ForecastDirector import ForecastDirector
from forecast_core.inference.HmfForecastService import HmfForecastService

from _runner import run_command

def main():
    director = ForecastDirector()
    director.refresh_forecast(HmfForecastService)

if __name__ == "__main__":
    run_command(main)