"""Calculate one product and return its exact serialized artifact bytes."""
from dataclasses import dataclass
from datetime import UTC

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
