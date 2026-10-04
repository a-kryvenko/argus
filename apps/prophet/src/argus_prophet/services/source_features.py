"""Prophet derives model inputs exclusively from Clio's recorded originals."""
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from datetime import UTC, datetime, timedelta

import httpx
import numpy as np
import pandas as pd

from common.config import get_config
from common.schemas.forecast_inputs import GONGFeatureFrame, SourceMeasurement, RawObservationFile

logger = logging.getLogger(__name__)
MAX_ORIGINAL_BYTES = 128 * 1024 * 1024


def cache_root():
    config = get_config()
    return Path(os.getenv('PROPHET_FEATURE_CACHE', str(config.data_root / 'prophet/source-cache')))


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.')
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(content)
        Path(name).replace(path)
    finally:
        Path(name).unlink(missing_ok=True)


def original(client, url, token, reference):
    path = cache_root() / reference.kind / (reference.sha256 + ('.json' if reference.kind == 'goes' else '.fits'))
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() == reference.sha256:
            return path
    # Build the path locally; owner responses cannot redirect credentials.
    endpoint = f'{url.rstrip("/")}/internal/v1/observations/files/{reference.kind}/{reference.sha256}'
    content = bytearray()
    params = {'slot_at': reference.slot_at.isoformat()} if reference.kind == 'aia' else {}
    with client.stream('GET', endpoint, params=params, headers={'Authorization': f'Bearer {token}'}) as response:
        response.raise_for_status()
        for block in response.iter_bytes():
            content.extend(block)
            if len(content) > MAX_ORIGINAL_BYTES:
                raise ValueError('Observation original exceeds supported size')
    if hashlib.sha256(content).hexdigest() != reference.sha256:
        raise ValueError('Observation checksum differs from its receipt')
    atomic_write(path, content)
    return path


def aia_record(reference, path):
    from argus_prophet.services.aia.extraction import extract_frame, VERSION
    cache = path.with_suffix(f'.features-v{VERSION}.npz')
    metadata = None
    try:
        with np.load(cache, allow_pickle=False) as saved:
            metadata = json.loads(str(saved['metadata']))
            if metadata['sha256'] != reference.sha256 or saved['dark'].shape != (61, 61):
                metadata = None
    except (OSError, ValueError, KeyError):
        pass
    if metadata is None:
        dark, ratio, metadata = extract_frame(path, allow_nrt=reference.source_product == 'aia.nrt_193')
        content = io.BytesIO()
        np.savez_compressed(content, dark=dark, ratio=ratio, metadata=json.dumps(metadata))
        atomic_write(cache, content.getvalue())
    observed = pd.Timestamp(metadata['observed_at']).to_pydatetime()
    if abs((observed-reference.slot_at).total_seconds()) > 600 or observed > reference.available_at:
        raise ValueError('AIA FITS timestamp conflicts with source receipt')
    return SimpleNamespace(**{**reference.model_dump(), 'observed_at': observed,
        'cache_path': str(cache), **{key: metadata[key] for key in ('b0_deg', 'valid_fraction', 'carrington_lon')}})


def prune_source_cache():
    """Discard old derived/download caches; Clio owns the source history."""
    cutoff = datetime.now(UTC).timestamp() - timedelta(days=45).total_seconds()
    root = cache_root()
    for kind in ('aia', 'gong', 'goes'):
        for path in (root / kind).glob('*'):
            try:
                if path.is_file() and not path.is_symlink() and path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                logger.warning('Cannot prune source cache: %s', path, exc_info=True)


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


def enrich_inputs(inputs, client, url, token):
    """Keep optional-product failures isolated; model policies enforce coverage."""
    refs = sorted((r for r in inputs.files if r.observed_at <= inputs.as_of and r.available_at <= inputs.as_of),
                  key=lambda r: (r.observed_at, r.available_at))
    aia, goes = [], []
    gong = [r for r in refs if r.kind == 'gong']
    # One most-recently received live frame per day, plus completed-day archives.
    selected_goes = {}
    for ref in refs:
        if ref.kind == 'goes':
            selected_goes[(ref.source_product, ref.slot_at.date())] = ref
    selected = [r for r in refs if r.kind == 'aia' and r.slot_at.hour % 6 == 0
                and r.observed_at >= inputs.as_of - timedelta(days=40)]
    selected += gong[-1:] + sorted(selected_goes.values(), key=lambda r: r.available_at)
    for ref in selected:
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
    calibrated = solar_measurements(goes, inputs.as_of) if goes else []
    inputs.measurements = [row for row in inputs.measurements if row.metric not in ('s10', 'm10', 'y10')] + calibrated
    by_time = {}
    for row in calibrated:
        by_time.setdefault(row.observed_at, {})[row.metric] = row.value
    for point in inputs.observations.points:
        for metric in ('s10', 'm10', 'y10'):
            setattr(point, metric, by_time.get(point.issue_time, {}).get(metric))
    prune_source_cache()
    return inputs
