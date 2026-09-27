"""CLI error reporting shared by services that install sentry-sdk."""
import os

from common.config import get_config


def setup_sentry():
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.excepthook import ExcepthookIntegration
    config = get_config()
    if not config.debug and not sentry_sdk.is_initialized():
        # The command boundary (or a continuing worker loop) reports failures.
        # Logging and the final traceback must not report the same failure again.
        sentry_sdk.init(
            dsn=os.getenv('SENTRY_COLLECT_POINT'),
            send_default_pii=True,
            integrations=[LoggingIntegration(event_level=None)],
            disabled_integrations=[ExcepthookIntegration()],
        )


def run_command(command):
    setup_sentry()
    try:
        return command()
    except Exception as exc:
        if not get_config().debug:
            import sentry_sdk
            sentry_sdk.capture_exception(exc)
        raise
