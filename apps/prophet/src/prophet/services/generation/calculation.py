"""Calculate one product and return its exact serialized artifact bytes."""
from dataclasses import dataclass
from datetime import UTC
import hashlib
import json
from pathlib import Path

from forecast.inputs import ForecastInputs

from prophet.services.generation.products import PRODUCTS


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


def proswin_metadata(product, inputs, metadata):
    details = {
        'proswin_input_records': len(inputs.proswin_predictions),
        'input_cutoff': inputs.as_of.isoformat(),
        'proswin_ready_at': inputs.proswin_ready_at.isoformat() if inputs.proswin_ready_at else None,
        'proswin_job': {key: value for key, value in inputs.proswin_job.items() if key != 'predictions'},
    }
    if product == 'solar-wind-speed':
        from forecast_core.inference.plasma.speed.dlinear_proswin import VERSION, KNOTS, WEIGHTS
        from forecast_core.models.proswin import CHECKPOINT_SHA
        identity = {'base_bundle_sha256': metadata['sha256'], 'version': VERSION,
                    'proswin_checkpoint_sha256': CHECKPOINT_SHA, 'knots': KNOTS, 'weights': WEIGHTS}
        details.update(identity, sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
                       model=VERSION, speed_point_model=VERSION,
                       uncertainty='DLinear empirical offsets; blend calibration pending')
    elif product == 'solar-wind-density':
        details.update(density_point_model='DLinear + PROSWIN-speed-assisted LightGBM',
                       uncertainty='NRT 2025 residual q10/q90; DLinear intervals on fallback')
    elif product == 'solar-wind-temperature':
        details.update(temperature_point_model='LightGBM with T/n/v history and PROSWIN speed',
                       uncertainty='Per-head NRT 2025 residual q10/q90')
    return details


def calculate_product(request: CalculationRequest) -> list[Artifact]:
    """Child entry point: only model files are read; no configuration, HTTP or DB."""
    if request.product == 'solar-wind-events':
        from prophet.services.events.calculation import calculate
        from prophet.services.events.speed import calculate_speed
        return calculate(request.inputs, registry=request.registry,
                         speed_calculator=lambda events: calculate_speed(request, events))
    from forecast.calculation import calculate_snapshot
    from prophet.services.generation.models import load_model
    product, inputs = request.product, request.inputs
    workdir, registry = request.workdir, request.registry
    issue_time = inputs.as_of.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    results, services = {}, {}
    for model in PRODUCTS[product].models:
        service_class = model.service_class()
        if model.bundled:
            service, metadata = load_model(service_class, workdir=workdir, registry=registry)
        else:
            entry = registry[model.artifact]
            service = service_class(
                density_model=entry.get('model', 'JB2008'),
                runtime_path=workdir / entry.get('runtime_path', 'data/models/dtm2020'))
            metadata = {'backend': 'forecast_core', 'registry_name': service.registry_name,
                        'issue_time': issue_time.isoformat()}
        if product in ('solar-wind-speed', 'solar-wind-density', 'solar-wind-temperature'):
            metadata.update(proswin_metadata(product, inputs, metadata))
        result = calculate_snapshot(service, inputs, issue_time=issue_time, model_info=metadata)
        if result.name != model.artifact:
            raise ValueError(f'Expected artifact {model.artifact}, received {result.name}')
        if product in ('solar-wind-density', 'solar-wind-temperature'):
            result.model_info.update(service.last_status)
        if product == 'atmospheric-density':
            result.model_info.update(service.model_info)
        if product == 'solar-radiation' and model.artifact != 'f10_7_quantile':
            result.model_info.update(service.model_info)
        results[result.name] = result
        services[result.name] = service
    if product == 'solar-wind-speed':
        from prophet.services.events.speed import attach_speed_outlook, integrate_speed_forecast
        outlook = attach_speed_outlook(request, results['plasma_speed_quantile'])
        integrate_speed_forecast(request, results, services, outlook)
    artifacts = []
    for result in results.values():
        artifact = serialize(result)
        if len(artifact.content) > 32 * 1024 * 1024:
            raise ValueError('Forecast artifact exceeds 32 MiB')
        artifacts.append(artifact)
    if sum(len(item.content) for item in artifacts) > 64 * 1024 * 1024:
        raise ValueError('Forecast product exceeds 64 MiB')
    return artifacts


def calculate_serialized(request):
    """Share the memory budget with the optional remote-container child."""
    if request.product == 'solar-wind-events':
        return calculate_product(request)
    from prophet.services.heavy_task import heavy_task
    with heavy_task(request.workdir / 'data'):
        return calculate_product(request)
