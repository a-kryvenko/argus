"""Build isolated historical demo bundles. Never write operational releases or scores.

python -m argus_prophet.demo --manifest event.json --observations observations.json
    --snapshots snapshots/ --output /absolute/path/current.json
Without --snapshots, read historical inputs through Clio's as_of contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

from common.publication import atomic_publish

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, HttpUrl, field_validator

HOUR = timedelta(hours=1)
REPLAY_START_HOURS = -96
REPLAY_END_HOURS = 72
# Demo products are isolated from the operational visibility/access policy.
VARIABLES = {
    'solar-wind-speed': [('plasma_speed_quantile', 'v', True, []),
                         ('plasma_speed_threshold', 'v', False, [450, 500, 600])],
    'solar-wind-density': [('plasma_density_quantile', 'n', True, [])],
    'geomagnetic-activity': [('kp_threshold', 'kp', False, [4, 5, 6]),
                             ('ap_quantile', 'ap', True, [])],
    'dst': [('dst_quantile', 'dst', True, [])],
    'hmf': [('hmf_total_threshold', 'bt', False, [5, 10, 15]),
            ('hmf_southward_threshold', 'bs', False, [5, 10, 15])],
    'solar-radiation': [('f10_7_quantile', 'f10_7', True, [])],
}
HORIZONS = {'solar-wind-speed': 96, 'solar-wind-density': 96, 'geomagnetic-activity': 48, 'dst': 48,
            'hmf': 48, 'solar-radiation': 48}


def selected_models(manifest, product):
    from argus_prophet.services.generation.products import PRODUCTS, Model
    models = (Model('f10_7_quantile', 'forecast_core.api.F107FS'),) if product == 'solar-radiation' else PRODUCTS[product].models
    return [model for model in models if model.artifact in manifest.models]


class Event(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(min_length=1)
    starts_at: AwareDatetime
    source_url: HttpUrl
    description: str = Field(min_length=1)

    @field_validator('starts_at')
    @classmethod
    def year_2026(cls, value):
        value = value.astimezone(UTC)
        if value.year != 2026:
            raise ValueError('Demo event must be in 2026')
        return value


class ModelEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    # Includes labels used in fitting, calibration and model selection.
    training_end: AwareDatetime
    training_evidence: str = Field(min_length=1)


class ProswinArchive(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    model_version: Literal['proswin-fold1-nrt-v1', 'proswin-fold1-science-v1']
    description: str = Field(min_length=1)


class EventArchive(BaseModel):
    model_config = ConfigDict(extra='forbid')
    path: str
    sha256: str = Field(pattern=r'^[a-f0-9]{64}$')
    arrival_model: str
    speed_model: ModelEvidence


class Manifest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event: Event
    observation_source: str = Field(min_length=1)
    input_source: str = Field(min_length=1)
    products: list[str] = Field(min_length=1)
    models: dict[str, ModelEvidence]
    proswin_archive: ProswinArchive | None = None
    event_archive: EventArchive | None = None

    @field_validator('products')
    @classmethod
    def supported_products(cls, value):
        if len(value) != len(set(value)) or set(value) - set(VARIABLES):
            raise ValueError('Expected distinct supported demo products')
        return value


def issue_times(event):
    anchor = event.starts_at.replace(minute=0, second=0, microsecond=0)
    return [anchor + offset * HOUR for offset in range(REPLAY_START_HOURS, REPLAY_END_HOURS + 1)]


def validate_evidence(manifest):
    required = {artifact for product in manifest.products for artifact, *_ in VARIABLES[product]}
    if set(manifest.models) - required or any(not selected_models(manifest, product) for product in manifest.products):
        raise ValueError('Training evidence must cover exactly the models used, with at least one model per product')
    for name, evidence in manifest.models.items():
        if evidence.training_end >= issue_times(manifest.event)[0]:
            raise ValueError(f'{name}: training/calibration overlaps the replay window')
    if manifest.event_archive:
        if manifest.event_archive.speed_model.training_end >= issue_times(manifest.event)[0]:
            raise ValueError('CME speed training/calibration overlaps replay')
        if not {'plasma_speed_quantile', 'plasma_speed_threshold'} <= manifest.models.keys():
            raise ValueError('CME integration requires both speed models')


def event_archive(manifest, workdir):
    spec = manifest.event_archive
    content = (workdir / spec.path).read_bytes()
    if hashlib.sha256(content).hexdigest() != spec.sha256:
        raise ValueError('ENLIL demo archive checksum mismatch')
    archive = json.loads(content)
    times = issue_times(manifest.event)
    if (archive.get('availability_basis') != 'provider_notification_publication_not_recorded_receipt'
            or datetime.fromisoformat(archive['coverage_start'].replace('Z', '+00:00')) > times[0]
            or datetime.fromisoformat(archive['coverage_end'].replace('Z', '+00:00')) < times[-1]):
        raise ValueError('ENLIL archive availability/coverage mismatch')
    return archive


def event_snapshot(archive, issue):
    from common.schemas.solar_wind_events import EventInputs, SolarWindEvent, EnlilArrivalInput
    events, arrivals = {}, []
    for row in archive['records']:
        available = datetime.fromisoformat(row['available_at'])
        completed = datetime.fromisoformat(row['model_completed_at'])
        observed = datetime.fromisoformat(row['observed_at'])
        if max(available, completed, observed) > issue:
            continue
        event = SolarWindEvent(event_id=row['event_id'], kind='cme', status='detected', source='DONKI',
            source_ids=[row['event_id']], available_at=available, observed_at=observed,
            initial_speed_km_s=row['initial_speed_km_s'], source_url=row['source_url'])
        old = events.get(event.event_id)
        if old is None or event.available_at >= old.available_at: events[event.event_id] = event
        arrivals.append(EnlilArrivalInput(**{k: row[k] for k in ('simulation_id', 'available_at',
            'model_completed_at', 'earth_arrival_at', 'duration_hours', 'glancing_blow', 'minor_impact', 'source_url')},
            cme_ids=[row['event_id']]))
    return EventInputs(schema_version=2, as_of=issue, read_at=issue, events=list(events.values()), enlil_arrivals=arrivals)


def validate_inputs(inputs, issue):
    if inputs.as_of != issue:
        raise ValueError('Input snapshot has a different as_of')
    for row in [*inputs.observations.points, *inputs.speed_observations, *inputs.density_observations]:
        if row.issue_time.tzinfo is None or row.issue_time >= issue:
            raise ValueError('Input contains an unclosed or future observation hour')
    for row in inputs.measurements:
        if row.observed_at > issue:
            raise ValueError('Input contains a future measurement')
    for row in [*inputs.files, *inputs.aia_frames, *([inputs.gong] if inputs.gong else [])]:
        if row.observed_at > issue or row.available_at > issue:
            raise ValueError('Source was not available at issue time')
    for row in getattr(inputs, 'proswin_predictions', []):
        if (row.available_at > issue or row.image_slot > issue or
                row.valid_time-row.image_slot != 96*HOUR or
                (row.source_cutoff is not None and row.source_cutoff > issue)):
            raise ValueError('PROSWIN input is unavailable or has invalid native horizon')
    if getattr(inputs, 'solar_wind_hourly', None) is not None:
        history = inputs.solar_wind_hourly
        if history.get('schema') != 'omni-hourly-v1':
            raise ValueError('Demo IMF requires explicitly identified OMNI inputs')
        seen = set()
        for row in history['rows']:
            time = datetime.fromisoformat(row['time'])
            if time.tzinfo is None or time.minute or time.second or time.microsecond or time >= issue or time in seen:
                raise ValueError('Invalid or future IMF observation')
            for key in ('bx', 'by', 'bz', 'v', 'n', 't'):
                value = row.get(key)
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                    raise ValueError('Invalid IMF observation value')
            seen.add(time)


def calculate_demo_snapshot(service, inputs, issue, metadata):
    from forecast.api import calculate_snapshot, ForecastResult
    if service.registry_name not in ('hmf_total_threshold', 'hmf_southward_threshold'):
        return calculate_snapshot(service, inputs, issue_time=issue, model_info=metadata)
    if service.registry_name == 'hmf_southward_threshold' and service.models_bundle.get('format') != 'argus-imf-revalidated-v1':
        return calculate_snapshot(service, inputs, issue_time=issue, model_info=metadata)
    # The trained feature definition also accepts OMNI. Do not label Earth-shifted
    # archive values as native L1 measurements just to pass the live adapter.
    import numpy as np
    import pandas as pd
    from forecast_core.api import build_imf_features
    if not inputs.solar_wind_hourly or inputs.solar_wind_hourly.get('schema') != 'omni-hourly-v1':
        raise ValueError('Missing historical OMNI IMF inputs')
    frame = pd.DataFrame(inputs.solar_wind_hourly['rows'])
    frame.index = pd.to_datetime(frame.pop('time'), utc=True)
    grid = pd.date_range(issue - 168 * HOUR, issue, freq='h', inclusive='left')
    frame = frame.reindex(grid)
    if any(not frame[key].notna().any() for key in ('bx', 'by', 'bz', 'v', 'n', 't')):
        raise ValueError('Insufficient historical IMF observations')
    features, _ = build_imf_features(frame)
    horizon = service.models_bundle['lead_hours']
    variable = 'bt' if service.registry_name == 'hmf_total_threshold' else 'bs'
    rows = pd.concat([features.loc[[issue]]] * horizon, ignore_index=True)
    rows['lead_hours'] = np.arange(1, horizon + 1)
    probabilities = service.predict_features(rows[service.models_bundle['columns']])
    result = pd.DataFrame({'issue_time': issue, 'lead_hours': rows.lead_hours})
    result['valid_time'] = issue + pd.to_timedelta(rows.lead_hours, unit='h')
    for index, threshold in enumerate((5, 10, 15)):
        result[f'p_{variable}_ge_{threshold}'] = probabilities[:, index]
    return ForecastResult(service.registry_name, result, metadata)


def number(value):
    value = float(value)
    return value if math.isfinite(value) else None


def forecast_payload(product, frames, issue):
    points = {}
    available = []
    for artifact, variable, quantiles, thresholds in VARIABLES[product]:
        if artifact not in frames:
            continue
        frame = frames[artifact]
        seen = set()
        if variable not in available:
            available.append(variable)
        for row in frame.to_dict('records'):
            lead = int(row['lead_hours'])
            if lead != float(row['lead_hours']) or lead < 1 or lead in seen:
                raise ValueError('Invalid or duplicate lead hour')
            seen.add(lead)
            valid = datetime.fromisoformat(str(row['valid_time']).replace('Z', '+00:00'))
            issued = datetime.fromisoformat(str(row['issue_time']).replace('Z', '+00:00'))
            if issued != issue or valid != issue + lead * HOUR:
                raise ValueError('Forecast times disagree with the historical issue')
            if lead > HORIZONS[product]:
                continue
            point = points.setdefault(lead, {'valid_time': valid.isoformat(), 'lead_hours': lead, 'variables': {}})
            values = point['variables'].setdefault(variable, {'continuous': None, 'binary': []})
            if quantiles:
                quantile = {q: number(row[f'{variable}_{q}']) for q in ('q10', 'q50', 'q90')}
                if all(value is not None for value in quantile.values()):
                    if not quantile['q10'] <= quantile['q50'] <= quantile['q90']:
                        raise ValueError('Unordered forecast quantiles')
                    values['continuous'] = quantile
            for threshold in thresholds:
                probability = number(row[f'p_{variable}_ge_{threshold}'])
                if probability is not None:
                    if not 0 <= probability <= 1:
                        raise ValueError('Probability outside [0, 1]')
                    values['binary'].append({'threshold': threshold, 'probability': probability})
    if not points or not any(value['continuous'] or value['binary'] for point in points.values() for value in point['variables'].values()):
        raise ValueError(f'{product}: no forecast values')
    return {'target': product, 'issue_time': issue.isoformat(), 'horizon_hours': max(points),
            'available_variables': available, 'predictions': [points[lead] for lead in sorted(points)]}


def observation_grid(rows, manifest, releases):
    variables = {variable for product in manifest.products for artifact, variable, *_ in VARIABLES[product]
                 if artifact in manifest.models}
    start = issue_times(manifest.event)[0] - 24 * HOUR
    end = max(datetime.fromisoformat(point['valid_time']) for forecasts in releases.values()
              for forecast in forecasts for point in forecast['predictions'])
    indexed = {}
    for row in rows:
        time = datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
        if time.tzinfo is None or time.minute or time.second or time.microsecond or time in indexed:
            raise ValueError('Observations must have unique timezone-aware UTC hour timestamps')
        values = {}
        for key in variables:
            value = row['values'].get(key)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError('Observed values must be finite numbers or null')
            values[key] = value
        indexed[time] = values
    if not indexed or min(indexed) > start or max(indexed) < end:
        raise ValueError('Observation window must cover history through the final forecast horizon')
    for key in variables:
        if not any(start <= time <= end and values[key] is not None for time, values in indexed.items()):
            raise ValueError(f'No observations for {key}')
    result = []
    while start <= end:
        result.append({'time': start.isoformat(), 'values': indexed.get(start, {key: None for key in variables})})
        start += HOUR
    return result


def build(manifest, observations, output, snapshots=None):
    from common.config import get_config
    from common.schemas.forecast_inputs import ForecastInputs
    from argus_prophet.services.generation.models import load_model
    from argus_prophet.config import InputPolicy, load_config

    validate_evidence(manifest)
    config = get_config()
    if manifest.event_archive:
        from forecast_core.api import cme_speed_model_bytes
        spec = manifest.event_archive
        registry = config.models_registry['models']
        entry = registry['solar_event_speed']
        path = config.workdir / 'data/models' / (entry['model'] + '.joblib')
        content = path.read_bytes() if path.is_file() else cme_speed_model_bytes(entry['model'])
        if (hashlib.sha256(content).hexdigest() != spec.speed_model.sha256
                or entry['sha256'] != spec.speed_model.sha256
                or registry['cme_arrival']['model'] != spec.arrival_model):
            raise ValueError('CME model changed; update demo evidence')
        event_archive(manifest, config.workdir)
    policy = load_config()
    services = {}
    for product in manifest.products:
        for model in selected_models(manifest, product):
            service, metadata = load_model(model.service_class(), workdir=config.workdir, registry=config.models_registry['models'])
            if metadata['sha256'] != manifest.models[model.artifact].sha256:
                raise ValueError(f'{model.artifact}: model changed; update its training evidence before rebuilding')
            if manifest.proswin_archive:
                if getattr(service, 'uses_aia', False):
                    service._aia.allowed_model_versions = (manifest.proswin_archive.model_version,)
                if getattr(service, 'uses_density_blend', False):
                    service._density.allowed_model_versions = (manifest.proswin_archive.model_version,)
            services[model.artifact] = (service, metadata)
    releases = {product: [] for product in manifest.products}
    event_releases = []
    evidence = []
    for index, issue in enumerate(issue_times(manifest.event)):
        if snapshots is not None:
            inputs = ForecastInputs.model_validate_json((snapshots / f'{index:03d}.json').read_text())
        else:
            from argus_prophet.services.inputs import load_inputs
            inputs = load_inputs(issue)
            # Source hourly aggregations may include the partial current hour.
            inputs.observations.points = [row for row in inputs.observations.points if row.issue_time < issue]
            inputs.speed_observations = [row for row in inputs.speed_observations if row.issue_time < issue]
            inputs.density_observations = [row for row in inputs.density_observations if row.issue_time < issue]
        if not {'hmf_total_threshold', 'hmf_southward_threshold'} & manifest.models.keys():
            inputs.solar_wind_hourly = None
        validate_inputs(inputs, issue)
        report = None
        if manifest.event_archive:
            from argus_prophet.services.cme_arrival import forecast_events
            report = forecast_events(inputs, registry=config.models_registry['models'], include_hss=False, include_dbem=False)
            report.model_version = 'solar-wind-events-v4'
            report.source_status['WSAEnlilSimulations'] = dict(state='retrospective',
                availability_basis='provider_notification_publication_not_recorded_receipt',
                scope='single_CME_notifications')
        evidence.append({'issue_time': issue.isoformat(), 'sha256': hashlib.sha256(inputs.model_dump_json().encode()).hexdigest(),
                         'aia_frames': len(inputs.aia_frames), 'proswin_records': len(inputs.proswin_predictions),
                         'models': {}})
        for product in manifest.products:
            # Only legacy Bs requires GONG; revalidated Bt/Bs use hourly history.
            if product != 'hmf' or ('hmf_southward_threshold' in manifest.models and
                    services['hmf_southward_threshold'][0].models_bundle.get('format') != 'argus-imf-revalidated-v1'):
                policy.inputs.get(product, InputPolicy()).validate_inputs(inputs)
            frames = {}
            results = {}
            for model in selected_models(manifest, product):
                service, metadata = services[model.artifact]
                result = calculate_demo_snapshot(service, inputs, issue, metadata)
                if result.name != model.artifact:
                    raise ValueError('Unexpected model output')
                frames[result.name] = result.frame
                results[result.name] = result
                if getattr(service, 'uses_density_blend', False):
                    count = service._density.last_status['candidate_leads']
                    evidence[-1]['models'][model.artifact] = {'candidate_leads': count,
                        'fallback_leads': HORIZONS[product]-count}
                elif getattr(service, 'uses_aia', False):
                    from forecast.inference.proswin_blend import select_predictions, VERSION, KNOTS, WEIGHTS
                    import pandas as pd
                    values = select_predictions(pd.Timestamp(issue), result.frame.valid_time,
                        inputs.proswin_predictions, allowed_model_versions=service._aia.allowed_model_versions)
                    count = int(sum(math.isfinite(v) for v in values[:96]))
                    evidence[-1]['models'][model.artifact] = {'candidate_leads': count, 'fallback_leads': 96-count,
                        'version': VERSION, 'knots': KNOTS, 'weights': WEIGHTS}
            outlook = None
            if product == 'solar-wind-speed' and report is not None:
                from argus_prophet.services.generation.calculation import CalculationRequest
                from argus_prophet.services.solar_event_speed import calculate_speed, integrate_speed_forecast
                from common.schemas.solar_wind_events import CMESpeedOutlook
                request = CalculationRequest(product, inputs, config.workdir, config.models_registry['models'])
                outlook = CMESpeedOutlook.model_validate(calculate_speed(request, report.events,
                    baseline=results['plasma_speed_quantile']))
                integrate_speed_forecast(request, results, {k: v[0] for k, v in services.items()}, outlook)
                frames = {name: result.frame for name, result in results.items()}
                report.cme_speed = outlook
                evidence[-1]['models']['cme_integration'] = results['plasma_speed_quantile'].model_info.get('cme_integration', {})
            payload = forecast_payload(product, frames, issue)
            if outlook is not None: payload['cme_speed'] = outlook.model_dump(mode='json')
            releases[product].append(payload)
        if report is not None:
            report.limitations.extend(['Historical notification publication substitutes for unknown Argus receipt times.',
                'Only unambiguously attributed single-CME notifications are included; missing context is not absence of incoming CME.'])
            event_releases.append(report.model_dump(mode='json'))
        print(f'Demo {index + 1}/{len(issue_times(manifest.event))}: {issue.isoformat()}', flush=True)
    observations = observation_grid(observations, manifest, releases)
    generated = datetime.now(UTC)
    bundle = {'schema_version': 1, 'version': generated.strftime('%Y%m%dT%H%M%S%fZ'),
              'generated_at': generated.isoformat(), **manifest.model_dump(mode='json', exclude={'products'}),
              'observations': observations, 'releases': releases, 'inputs': evidence}
    if manifest.event_archive: bundle['event_releases'] = event_releases
    atomic_publish(bundle, output)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--observations', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(Manifest.model_validate_json(args.manifest.read_text()), json.loads(args.observations.read_text()),
          args.output, args.snapshots)


if __name__ == '__main__':
    main()
