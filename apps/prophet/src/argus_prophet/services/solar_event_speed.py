"""Load approved artifacts and delegate speed forecasting to private forecast-core."""
import hashlib
import io
import logging

from common.exceptions import ConfigurationException

logger = logging.getLogger(__name__)


def calculate_speed(request, events):
    entry=request.registry.get('solar_event_speed')
    unavailable=dict(model_version='cme-speed-v2',state='unavailable')
    if not entry:
        return dict(unavailable,reason='CME speed model is not configured')
    try:
        import joblib
        import pandas as pd
        from forecast.api import SWSpeedFS, calculate_snapshot
        from forecast_core.api import forecast_cme_speed
        from argus_prophet.services.generation.models import load_model
        from datetime import UTC
        content=(request.workdir/'data/models'/(entry['model']+'.joblib')).read_bytes()
        sha=hashlib.sha256(content).hexdigest()
        if sha!=entry['sha256']:
            raise ValueError('CME model fingerprint differs from approved release')
        bundle=joblib.load(io.BytesIO(content))
        service,metadata=load_model(SWSpeedFS,workdir=request.workdir,registry=request.registry)
        issue=request.inputs.as_of.astimezone(UTC).replace(minute=0,second=0,microsecond=0)
        baseline=calculate_snapshot(service,request.inputs,issue_time=issue,model_info=metadata)
        f=baseline.frame.sort_values('valid_time')
        f=f[(pd.to_datetime(f.valid_time,utc=True)>request.inputs.as_of) & (f.lead_hours<=96)]
        output=forecast_cme_speed(request.inputs,events,bundle,pd.to_datetime(f.valid_time,utc=True),f['v_q50'].to_numpy())
        output.update(model_sha256=sha,background_model=metadata)
        return output
    except (OSError, ValueError, KeyError, RuntimeError, ImportError, ConfigurationException) as exc:
        logger.exception('CME speed forecast unavailable; retaining source arrival predictions')
        return dict(unavailable,reason=type(exc).__name__)
