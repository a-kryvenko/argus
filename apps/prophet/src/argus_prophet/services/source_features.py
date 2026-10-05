"""Prophet derives model inputs exclusively from Clio's recorded originals."""
import io
import logging
from datetime import timedelta

import httpx
import numpy as np
import pandas as pd

from common.config import get_config
from common.schemas.forecast_inputs import GONGFeatureFrame, SourceMeasurement

from argus_prophet.services.source_cache import original, prune_source_cache
from argus_prophet.services.aia.cache import aia_record

logger = logging.getLogger(__name__)


def solar_measurements(records, as_of):
    """Calibrate source samples here; absence never fabricates an index."""
    from forecast_core.api import load_solar_index_calibrations, extract_solar_indices, extract_solar_index_observations
    config = get_config()
    setting = config.models_registry.get('models', {}).get('solar_index_calibration', {})
    if not setting.get('calibration_path') or not records:
        return []
    try:
        calibration = load_solar_index_calibrations(config.workdir / setting['calibration_path'])
        samples = pd.concat(records, ignore_index=True)
        samples['timestamp'] = pd.to_datetime(samples.timestamp, utc=True)
        samples = samples.loc[samples.timestamp.le(as_of)].drop_duplicates('timestamp', keep='last').sort_values('timestamp')
        completed = samples.loc[samples.timestamp.lt(pd.Timestamp(as_of).floor('D'))]
        rows = []
        if not completed.empty:
            daily = extract_solar_indices(completed, calibration).rename(columns={'timestamp': 'observed_at'})
            rows.extend(daily.to_dict('records'))
        # Current hourly estimate uses only samples up to that hour boundary.
        current = samples.loc[samples.timestamp.ge(pd.Timestamp(as_of).floor('D') - pd.Timedelta(days=1))]
        if not current.empty:
            hourly = extract_solar_index_observations(current, calibration, pd.Timestamp(as_of))
            rows.extend(hourly.loc[hourly.observed_at.eq(pd.Timestamp(as_of).floor('h'))].to_dict('records'))
        values = {(row['observed_at'], metric): SourceMeasurement(metric=metric, value=float(row[metric]),
                  observed_at=row['observed_at']) for row in rows for metric in ('s10', 'm10', 'y10')
                  if pd.notna(row[metric]) and np.isfinite(row[metric])}
        return list(values.values())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        logger.exception('Solar indices unavailable: Prophet calibration failed')
        return []


def select_sources(inputs):
    """Select causal receipts before downloading or extracting any features."""
    refs = sorted((r for r in inputs.files if r.observed_at <= inputs.as_of and r.available_at <= inputs.as_of),
                  key=lambda r: (r.observed_at, r.available_at))
    gong = [r for r in refs if r.kind == 'gong']
    # One most-recently received live frame per day, plus completed-day archives.
    selected_goes = {}
    for ref in refs:
        if ref.kind == 'goes':
            selected_goes[(ref.source_product, ref.slot_at.date())] = ref
    selected = [r for r in refs if r.kind == 'aia' and r.slot_at.hour % 6 == 0
                and r.observed_at >= inputs.as_of - timedelta(days=40)]
    selected += gong[-1:] + sorted(selected_goes.values(), key=lambda r: r.available_at)
    return selected


def enrich_inputs(inputs, client, url, token):
    """Keep optional-product failures isolated; model policies enforce coverage."""
    aia, goes = [], []
    for ref in select_sources(inputs):
        try:
            path = original(client, url, token, ref)
            if ref.kind == 'gong':
                from forecast_core.api import extract_gong_features
                features = extract_gong_features(path)
                if not features:
                    raise ValueError('GONG extraction returned no features')
                inputs.gong = GONGFeatureFrame(**{k: getattr(ref, k) for k in (
                    'observed_at', 'available_at', 'sha256', 'source_product')}, features=features)
            elif ref.kind == 'aia':
                aia.append(aia_record(ref, path))
            else:
                goes.append(pd.read_json(io.StringIO(path.read_text()), orient='records'))
        except (OSError, ValueError, KeyError, TypeError, RuntimeError, httpx.HTTPError):
            logger.exception('Cannot prepare %s model inputs from %s', ref.kind, ref.sha256)
    inputs.aia_frames = []
    if aia:
        from argus_prophet.services.aia.features import feature_frames
        inputs.aia_frames = feature_frames(aia, inputs.as_of)
    apply_solar_measurements(inputs, solar_measurements(goes, inputs.as_of) if goes else [])
    prune_source_cache()
    return inputs


def apply_solar_measurements(inputs, calibrated):
    inputs.measurements = [row for row in inputs.measurements if row.metric not in ('s10', 'm10', 'y10')] + calibrated
    by_time = {}
    for row in calibrated:
        by_time.setdefault(row.observed_at, {})[row.metric] = row.value
    for point in inputs.observations.points:
        for metric in ('s10', 'm10', 'y10'):
            setattr(point, metric, by_time.get(point.issue_time, {}).get(metric))
