from forecast.ForecastDirector import ForecastDirector
from forecast.inference.KpForecastService import KpForecastService

import sentry_sdk

from common.config import get_config

import os

config = get_config()

if not config.debug:
    sentry_sdk.init(
        dsn=os.getenv("SENTRY_COLLECT_POINT"),
        send_default_pii=True,
    )

def main():
    try:
        director = ForecastDirector()
        director.refresh_forecast(KpForecastService)
    except Exception as exc:
        if not config.debug:
            sentry_sdk.capture_exception(exc)
        else:
            raise exc

if __name__ == "__main__":
    main()