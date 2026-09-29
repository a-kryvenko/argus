"""Coordinate isolated calculations and persist their exact serialized bytes."""
from dataclasses import dataclass
from datetime import UTC

from common.config import get_config
from argus_prophet.services.generation.products import PRODUCTS


@dataclass(frozen=True)
class Artifact:
    name: str
    content: bytes
    model_info: dict
    row_count: int
    columns: list[str]


def serialize(result):
    return Artifact(result.name, result.frame.to_csv(index=False).encode('utf-8'),
                    result.model_info, len(result.frame), list(result.frame.columns))


def calculate_product(product, inputs, workdir, registry):
    """Child entry point: only model files are read; no configuration, HTTP or DB."""
    from forecast.api import calculate_snapshot
    from argus_prophet.services.generation.models import load_model
    issue_time = inputs.as_of.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    artifacts = []
    for model in PRODUCTS[product].models:
        if model.bundled:
            service, metadata = load_model(model.service_class(), workdir=workdir, registry=registry)
        else:
            service = model.service_class()()
            metadata = {'backend': 'forecast_core', 'registry_name': service.registry_name,
                        'issue_time': issue_time.isoformat()}
        result = calculate_snapshot(service, inputs, issue_time=issue_time, model_info=metadata)
        if result.name != model.artifact:
            raise ValueError(f'Expected artifact {model.artifact}, received {result.name}')
        artifact = serialize(result)
        if len(artifact.content) > 32 * 1024 * 1024:
            raise ValueError('Forecast artifact exceeds 32 MiB')
        artifacts.append(artifact)
    if sum(len(item.content) for item in artifacts) > 64 * 1024 * 1024:
        raise ValueError('Forecast product exceeds 64 MiB')
    return artifacts


def calculate(product: str, *, inputs, recorder):
    from argus_prophet.config import ProphetConfig, InputPolicy
    from argus_prophet.scheduling.execution import execute
    config = get_config()
    policy = ProphetConfig.model_validate(getattr(config, 'project_config', {}).get('prophet', {}))
    policy.inputs.get(product, InputPolicy()).validate_inputs(inputs)
    artifacts = execute(calculate_product, product, inputs, config.workdir, config.models_registry['models'],
                        timeout_seconds=policy.calculation_timeout_seconds)
    for artifact in artifacts:
        recorder.store(artifact.name, artifact.content, artifact.model_info, artifact.row_count, artifact.columns)


def store_result(recorder, result):
    artifact = serialize(result)
    recorder.store(artifact.name, artifact.content, artifact.model_info, artifact.row_count, artifact.columns)
