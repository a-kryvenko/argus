"""Load approved artifacts and delegate speed forecasting to private forecast-core."""
import hashlib
import io
import logging

from common.exceptions import ConfigurationException

logger = logging.getLogger(__name__)


def calculate_speed(request, events, *, baseline=None):
    entry=request.registry.get('solar_event_speed')
    unavailable=dict(model_version='cme-speed-v2',state='unavailable')
    source=request.inputs.solar_wind_events
    if source is None or not (any(p.endpoint in ('CME', 'WSAEnlilSimulations') and
                                 p.received_at <= request.inputs.as_of for p in source.polls)
                              or any(e.arrival_model for e in events)):
        return dict(unavailable,reason='CME arrival source inputs unavailable')
    if not entry:
        return dict(unavailable,reason='CME speed model is not configured')
    try:
        import joblib
        import pandas as pd
        from forecast.api import SWSpeedFS, calculate_snapshot
        from forecast_core.api import forecast_cme_speed, cme_speed_model_bytes
        from argus_prophet.services.generation.models import load_model
        from datetime import UTC
        path=request.workdir/'data/models'/(entry['model']+'.joblib')
        content=path.read_bytes() if path.is_file() else cme_speed_model_bytes(entry['model'])
        sha=hashlib.sha256(content).hexdigest()
        if sha!=entry['sha256']:
            raise ValueError('CME model fingerprint differs from approved release')
        bundle=joblib.load(io.BytesIO(content))
        if baseline is None:
            service,metadata=load_model(SWSpeedFS,workdir=request.workdir,registry=request.registry)
            issue=request.inputs.as_of.astimezone(UTC).replace(minute=0,second=0,microsecond=0)
            baseline=calculate_snapshot(service,request.inputs,issue_time=issue,model_info=metadata)
        metadata=baseline.model_info
        f=baseline.frame.sort_values('valid_time')
        f=f[(pd.to_datetime(f.valid_time,utc=True)>request.inputs.as_of) & (f.lead_hours<=96)]
        output=forecast_cme_speed(request.inputs,events,bundle,pd.to_datetime(f.valid_time,utc=True),f['v_q50'].to_numpy())
        output.update(model_sha256=sha,background_model=dict(metadata),as_of=request.inputs.as_of)
        return output
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, ConfigurationException) as exc:
        logger.exception('CME speed forecast unavailable; retaining source arrival predictions')
        return dict(unavailable,reason=type(exc).__name__)


def attach_speed_outlook(request, result):
    """Store the scenario with its own baseline release, never from another run."""
    from common.schemas.solar_wind_events import CMESpeedOutlook
    from argus_prophet.services.cme_arrival import forecast_events
    inputs=request.inputs
    if inputs.solar_wind_events is None:
        output=dict(model_version='cme-speed-v2',state='unavailable',reason='Event inputs unavailable')
    else:
        try:
            report=forecast_events(inputs,registry=request.registry,include_hss=False,include_dbem=False)
            output=calculate_speed(request,report.events,baseline=result)
        except ValueError:
            logger.exception('Normalized event inputs unavailable; retaining background speed')
            output=dict(model_version='cme-speed-v2',state='unavailable',reason='Normalized event inputs unavailable')
    outlook=CMESpeedOutlook.model_validate(output)
    result.frame['cme_speed_json']=''
    result.frame.loc[result.frame.index[0],'cme_speed_json']=outlook.model_dump_json()
    result.model_info['cme_speed']={'state':outlook.state,'model_sha256':outlook.model_sha256}
    return outlook


def integrate_speed_forecast(request, results, services, outlook):
    """Apply one same-issue CME increment to speed quantiles and probabilities."""
    import json
    import numpy as np
    import pandas as pd
    from forecast_core.api import translated_speed_probabilities

    policy = request.registry.get('solar_event_speed', {}).get('integration', {})
    if not policy.get('enabled', False):
        return
    if policy.get('version') != 'cme-arrival-window-v2':
        raise ValueError('Unsupported CME speed integration policy')
    quantiles = results['plasma_speed_quantile']
    probabilities = results['plasma_speed_threshold']
    q, p = quantiles.frame.copy(), probabilities.frame.copy()
    if not q.valid_time.equals(p.valid_time) or not q.lead_hours.equals(p.lead_hours):
        raise ValueError('Speed distributions have different time grids')
    if request.registry['plasma_speed_quantile']['model'] != request.registry['plasma_speed_threshold']['model']:
        raise ValueError('CME translation requires a shared background distribution')
    background = q.v_q50.to_numpy().copy()
    increment = np.zeros(len(q))
    indexed = {pd.Timestamp(point.valid_time): point for point in outlook.points}
    for j, valid in enumerate(pd.to_datetime(q.valid_time, utc=True)):
        point = indexed.get(valid)
        if point is None:
            continue
        if point.speed_integrated is None:
            raise ValueError('Missing weighted CME scenario')
        if not np.isclose(point.speed_background, background[j], rtol=0, atol=1e-8):
            raise ValueError('CME integration uses a different background')
        increment[j] = point.speed_integrated-background[j]
    if (increment < 0).any() or not np.isfinite(increment).all():
        raise ValueError('Invalid CME speed increment')
    active = increment > 0
    if active.any():
        translated = translated_speed_probabilities(
            services['plasma_speed_threshold'].models_bundle, p, background, increment)
        for column, values in translated.items():
            p.loc[active, column] = values[active]
        for column in ('v_q10', 'v_q50', 'v_q90'):
            q.loc[active, column] += increment[active]
    q['v_background_q50'] = background
    outlook.integrated_into_speed = True
    q.loc[q.index[0], 'cme_speed_json'] = outlook.model_dump_json()
    integration = dict(version='cme-arrival-window-v2', peak_weight=1., window_width_multiplier=1.,
        late_extension_hours=12., corrected_hours=int(active.sum()), source_state=outlook.state,
        cme_model_sha256=outlook.model_sha256,
        uncertainty='Conditional translation of background residuals; CME timing/amplitude uncertainty is not calibrated')
    # Commit both outputs together after the entire distribution has been checked.
    quantiles.frame, probabilities.frame = q, p
    for result in (quantiles, probabilities):
        identity = dict(background_sha256=result.model_info['sha256'],
            arrival_model=request.registry.get('cme_arrival', {}).get('model', 'argus-cme-arrival-enlil-v1'),
            cme_model_sha256=request.registry['solar_event_speed']['sha256'],
            integration_version=integration['version'], peak_weight=1., late_extension_hours=12.)
        result.model_info.update(background_model=dict(result.model_info), cme_integration=integration,
            model=result.model_info['model']+'+cme-window-v2',
            sha256=hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest(),
            uncertainty=integration['uncertainty'])
