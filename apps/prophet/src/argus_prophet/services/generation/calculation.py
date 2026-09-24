"""One calculation path for scheduled runs and local development."""
from datetime import UTC

from argus_prophet.services.generation.products import PRODUCTS
from common.config import get_config
from argus_prophet.services.generation.models import load_model
from forecast.api import ForecastResult, calculate_forecast


def calculate(product: str, *, inputs, recorder):
    definition = PRODUCTS[product]
    issue_time = inputs.as_of.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    if definition.backend == 'density':
        from argus_prophet.services.density.forecast import calculate_density
        store_result(recorder, calculate_density(inputs=inputs, issue_time=issue_time))
        return
    config = get_config()
    for model in definition.models:
        service, model_info = load_model(model.service_class(), workdir=config.workdir,
                                         registry=config.models_registry['models'])
        extra = {}
        if getattr(service, '_dlinear', None) is not None:
            import pandas as pd
            extra['speed_history'] = pd.DataFrame(
                [point.model_dump() for point in inputs.speed_observations], columns=['issue_time', 'v'])
        if getattr(service, 'uses_aia', False):
            import pandas as pd
            extra['aia_features'] = pd.DataFrame([
                {**point.features, 'slot_at': point.slot_at,
                 'observed_at': point.observed_at, 'available_at': point.available_at}
                for point in inputs.aia_frames])
        result = calculate_forecast(service, inputs.observations,
                                    issue_time=issue_time, model_info=model_info, **extra)
        store_result(recorder, result)


def store_result(recorder, result: ForecastResult) -> None:
    frame = result.frame
    content = frame.to_csv(index=False).encode('utf-8')
    recorder.store(result.name, content, result.model_info, len(frame), list(frame.columns))
