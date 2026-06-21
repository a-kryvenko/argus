import os
from collections.abc import Callable

import sentry_sdk

from common.config import get_config

config = get_config()

def setup_sentry() -> None:
    if config.debug:
        return

    sentry_sdk.init(
        dsn=os.getenv("SENTRY_COLLECT_POINT"),
        send_default_pii=True,
    )


def run_command(command: Callable[[], None]) -> None:
    setup_sentry()

    try:
        command()
    except Exception as exc:
        if config.debug:
            raise

        sentry_sdk.capture_exception(exc)
        raise