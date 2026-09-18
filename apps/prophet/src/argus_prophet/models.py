"""Load a model and fingerprint the exact bytes used for calculation."""
import hashlib
import io
from pathlib import Path

import joblib
from common.exceptions import ConfigurationException
from forecast.inference._forecast_service import DefaultForecastService


def load_model(service: type[DefaultForecastService], *, workdir: Path, registry: dict):
    entry = registry.get(service.registry_name)
    if not entry:
        raise ConfigurationException(f'Not found registry for {service.registry_name}')
    path = workdir / 'data/models' / (entry['model'] + '.joblib')
    if not path.is_file():
        raise ConfigurationException(f'{path} not exists')
    content = path.read_bytes()
    model_info = {'registry_name': service.registry_name, 'model': entry['model'],
                  'sha256': hashlib.sha256(content).hexdigest()}
    return service(joblib.load(io.BytesIO(content))), model_info
