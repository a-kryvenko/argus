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
    if source is None or not any(p.endpoint in ('CME', 'WSAEnlilSimulations') and
                                 p.received_at <= request.inputs.as_of for p in source.polls):
        return dict(unavailable,reason='ENLIL source inputs unavailable')
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
    from forecast_core.api import forecast_solar_wind_events
    inputs=request.inputs
    if inputs.solar_wind_events is None:
        output=dict(model_version='cme-speed-v2',state='unavailable',reason='Event inputs unavailable')
    else:
        report=forecast_solar_wind_events(inputs,include_hss=False,include_dbem=False)
        output=calculate_speed(request,report.events,baseline=result)
    outlook=CMESpeedOutlook.model_validate(output)
    result.frame['cme_speed_json']=''
    result.frame.loc[result.frame.index[0],'cme_speed_json']=outlook.model_dump_json()
    result.model_info['cme_speed']={'state':outlook.state,'model_sha256':outlook.model_sha256}
