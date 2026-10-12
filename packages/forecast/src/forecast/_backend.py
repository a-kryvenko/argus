"""Single optional boundary to the separately installed proprietary backend."""
from importlib import import_module


def resolve(module, name):
    if name.startswith("__"):
        raise AttributeError(name)
    try:
        backend = import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == "forecast_core" or (exc.name and exc.name.startswith("forecast_core.")):
            raise ImportError("Model inference requires the private forecast-core package") from exc
        raise
    return getattr(backend, name)
