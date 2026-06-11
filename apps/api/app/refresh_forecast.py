from forecast_core.predictor import refresh_forecast
import sentry_sdk

from common.config import get_config

import os

config = get_config()

sentry_sdk.init(
    dsn=os.getenv("SENTRY_COLLECT_POINT"),
    send_default_pii=True,
)

def main():
    try:
        refresh_forecast()
    except Exception as exc:
        sentry_sdk.capture_exception(exc)

if __name__ == "__main__":
    main()