from forecast.ForecastDirector import ForecastDirector
from forecast.inference.PlasmaStateForecastService import PlasmaStateForecastService

from _runner import run_command

def main():
    director = ForecastDirector()
    director.refresh_forecast(PlasmaStateForecastService)

if __name__ == "__main__":
    run_command(main)