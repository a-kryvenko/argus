from forecast.ForecastDirector import ForecastDirector
from forecast.inference.KpForecastService import KpForecastService

from _runner import run_command

def main():
    director = ForecastDirector()
    director.refresh_forecast(KpForecastService)

if __name__ == "__main__":
    run_command(main)