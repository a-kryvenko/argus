"""Load a model and fingerprint the exact bytes used for calculation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

from common.exceptions import ConfigurationException
if TYPE_CHECKING:
    from forecast.calculation import SnapshotService


def load_model(service: type[SnapshotService], *, workdir: Path, registry: dict):
    entry = registry.get(service.registry_name)
    if not entry:
        raise ConfigurationException(f'Not found registry for {service.registry_name}')
    path = workdir / entry.get('artifact_path', 'data/models/' + entry['model'] + '.joblib')
    if not path.is_file():
        raise ConfigurationException(f'{path} not exists')
    content = path.read_bytes()
    model_info = {'registry_name': service.registry_name, 'model': entry['model'],
                  'sha256': hashlib.sha256(content).hexdigest()}
    if path.suffix == '.json':
        bundle = json.loads(content)
    else:
        import joblib
        bundle = joblib.load(io.BytesIO(content))
    return service(bundle), model_info
