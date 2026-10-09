"""Calculate one product and return its exact serialized artifact bytes."""
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

from common.schemas.forecast_inputs import ForecastInputs

from argus_prophet.services.generation.products import PRODUCTS


@dataclass(frozen=True)
class CalculationRequest:
    """Serializable input for a child process; never carries a database writer."""

    product: str
    inputs: ForecastInputs
    workdir: Path
    registry: dict


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


def calculate_product(request: CalculationRequest) -> list[Artifact]:
    """Child entry point: only model files are read; no configuration, HTTP or DB."""
    from forecast.api import calculate_snapshot
    from argus_prophet.services.generation.models import load_model
    product, inputs = request.product, request.inputs
    workdir, registry = request.workdir, request.registry
    issue_time = inputs.as_of.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    artifacts = []
    for model in PRODUCTS[product].models:
        if model.bundled:
            service, metadata = load_model(model.service_class(), workdir=workdir, registry=registry)
        else:
            service = model.service_class()()
            metadata = {'backend': 'forecast_core', 'registry_name': service.registry_name,
                        'issue_time': issue_time.isoformat()}
        if product == 'solar-wind-speed' and getattr(service, 'uses_aia', False):
            from forecast.inference.proswin_blend import VERSION, KNOTS, WEIGHTS
            from forecast.inference.proswin_runtime import CHECKPOINT_SHA
            import hashlib
            import json
            identity = {'base_bundle_sha256': metadata['sha256'], 'version': VERSION,
                        'proswin_checkpoint_sha256': CHECKPOINT_SHA, 'knots': KNOTS, 'weights': WEIGHTS}
            identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
            metadata = {**metadata, **identity, 'sha256': identity_sha,
                        'model': VERSION, 'speed_point_model': VERSION,
                        'uncertainty': 'DLinear empirical offsets; blend calibration pending',
                        'proswin_input_records': len(inputs.proswin_predictions),
                        'input_cutoff': inputs.as_of.isoformat(),
                        'proswin_ready_at': inputs.proswin_ready_at.isoformat() if inputs.proswin_ready_at else None,
                        'proswin_job': {k:v for k,v in inputs.proswin_job.items() if k != 'predictions'}}
        if getattr(service, 'uses_density_blend', False):
            metadata = {**metadata, 'density_point_model': 'DLinear + PROSWIN-speed-assisted LightGBM',
                        'uncertainty': 'NRT 2025 residual q10/q90; DLinear intervals on fallback',
                        'proswin_input_records': len(inputs.proswin_predictions),
                        'input_cutoff': inputs.as_of.isoformat(),
                        'proswin_ready_at': inputs.proswin_ready_at.isoformat() if inputs.proswin_ready_at else None,
                        'proswin_job': {k:v for k,v in inputs.proswin_job.items() if k != 'predictions'}}
        if getattr(service, 'uses_temperature_proswin', False):
            metadata = {**metadata, 'temperature_point_model': 'LightGBM with T/n/v history and PROSWIN speed',
                        'uncertainty': 'Per-head NRT 2025 residual q10/q90',
                        'proswin_input_records': len(inputs.proswin_predictions),
                        'input_cutoff': inputs.as_of.isoformat(),
                        'proswin_job': {k:v for k,v in inputs.proswin_job.items() if k != 'predictions'}}
        result = calculate_snapshot(service, inputs, issue_time=issue_time, model_info=metadata)
        if result.name != model.artifact:
            raise ValueError(f'Expected artifact {model.artifact}, received {result.name}')
        if getattr(service, 'uses_density_blend', False):
            result.model_info.update(service._density.last_status)
        if getattr(service, 'uses_temperature_proswin', False):
            result.model_info.update(service.last_status)
        artifact = serialize(result)
        if len(artifact.content) > 32 * 1024 * 1024:
            raise ValueError('Forecast artifact exceeds 32 MiB')
        artifacts.append(artifact)
    if sum(len(item.content) for item in artifacts) > 64 * 1024 * 1024:
        raise ValueError('Forecast product exceeds 64 MiB')
    return artifacts


def calculate_serialized(request):
    """Share the memory budget with the optional remote-container child."""
    from argus_prophet.services.heavy_task import heavy_task
    with heavy_task(request.workdir / 'data'):
        return calculate_product(request)
