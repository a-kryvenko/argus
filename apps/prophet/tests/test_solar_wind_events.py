import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from common.schemas.forecast_inputs import ForecastInputs
from common.schemas.forecast_release import ForecastArtifact, ForecastRelease
from common.schemas.observation import Observation
from common.schemas.solar_wind_events import EventInputs, L1Sample, SourcePoll, SolarWindEvent
from argus_prophet.services.events import forecast, l1_detections
from argus_prophet.services.generation.calculation import CalculationRequest, calculate_product

NOW = datetime(2026, 10, 13, 12, 5, tzinfo=UTC)


def poll(endpoint, rows, received=NOW):
    return SourcePoll(endpoint=endpoint, poll_id=uuid4(), received_at=received,
        sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        request_start=NOW-timedelta(days=30), request_end=NOW+timedelta(days=1), records=rows)


def simulation(ids=('cme1',), arrival='2026-10-14T12:00Z', completed='2026-10-13T10:00Z', link='run1'):
    return dict(cmeInputs=[{'CMEID': i} for i in ids], link=link,
                modelCompletionTime=completed, estimatedShockArrivalTime=arrival,
                estimatedDuration=20, isEarthGB=True, isEarthMinorImpact=False)


def source(*polls, samples=()):
    # Prophet receives normalized events from Clio, independently of raw provider parsing.
    events = []
    for p in polls:
        if p.endpoint == 'WSAEnlilSimulations':
            for row in p.records:
                events.append(SolarWindEvent(event_id='cme1', kind='cme', status='predicted_hit',
                    source='DONKI/WSA-ENLIL', source_ids=['cme1'], available_at=p.received_at,
                    model_completed_at=datetime.fromisoformat(row['modelCompletionTime'].replace('Z', '+00:00')),
                    arrival_at=datetime.fromisoformat(row['estimatedShockArrivalTime'].replace('Z', '+00:00')),
                    arrival_target='Earth', observed_at=NOW-timedelta(hours=4), initial_speed_km_s=900))
    return EventInputs(as_of=NOW, read_at=NOW, polls=list(polls), events=events, l1_samples=list(samples))


def samples():
    onset = NOW-timedelta(minutes=30)
    rows = []
    for minute in range(-40, 20):
        time = onset+timedelta(minutes=minute)
        values = dict(v=500 if minute >= 0 else 400, n=10 if minute >= 0 else 4,
                      bx=8 if minute >= 0 else 3, by=0, bz=-2)
        rows.extend(L1Sample(observed_at=time, available_at=time+timedelta(seconds=30),
            metric=m, value=v, quality='unverified', spacecraft='ACE') for m, v in values.items())
    return onset, rows


def test_sustained_l1_compression_is_one_unattributed_detection():
    onset, rows = samples()
    events = source(poll('WSAEnlilSimulations', [simulation(arrival=onset.isoformat())])).events
    found = l1_detections(source(samples=rows), events)
    assert len(found) == 1 and found[0].onset_at == onset
    assert found[0].detected_at == onset+timedelta(minutes=10)
    assert found[0].attribution == 'unresolved' and found[0].candidate_event_ids == ['cme1']


def test_l1_can_associate_own_ensemble_without_enlil_prediction():
    from types import SimpleNamespace
    onset, rows = samples()
    own = SimpleNamespace(cme_id='cme1', arrival_q10=onset-timedelta(hours=1),
                          arrival_q90=onset+timedelta(hours=1))
    result = l1_detections(source(samples=rows), [], [own])
    assert result[0].candidate_event_ids == ['cme1'] and result[0].attribution == 'unresolved'


@pytest.mark.parametrize('failure', ['spike', 'gap', 'flagged', 'late_receipt', 'source_switch', 'missing_vector'])
def test_l1_rejects_spikes_gaps_unavailable_and_switched_sources(failure):
    onset, rows = samples()
    if failure == 'gap':
        rows = [r for r in rows if not onset-timedelta(minutes=10) <= r.observed_at < onset]
    elif failure == 'missing_vector':
        rows = [r for r in rows if r.metric != 'bz']
    else:
        for row in rows:
            if row.observed_at >= onset:
                if failure == 'source_switch':
                    row.spacecraft = 'DSCOVR'
                if failure == 'flagged':
                    row.quality = 'flagged'
                if failure == 'late_receipt':
                    row.available_at = NOW+timedelta(minutes=1)
                if failure == 'spike' and row.observed_at >= onset+timedelta(minutes=1):
                    row.value = dict(v=400, n=4, bx=3, by=0, bz=-2)[row.metric]
    assert l1_detections(source(samples=rows), []) == []


def test_empty_report_publishes_through_existing_release_contract():
    inputs = ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[]), solar_wind_events=source())
    artifacts = calculate_product(CalculationRequest('solar-wind-events', inputs, Path('/unused'), {}))
    artifact = artifacts[0]
    validated = ForecastArtifact(name=artifact.name, csv_text=artifact.content.decode(),
        sha256=hashlib.sha256(artifact.content).hexdigest(), row_count=artifact.row_count,
        columns=artifact.columns, model_info=artifact.model_info)
    release = ForecastRelease(release_id=uuid4(), run_id=uuid4(), product='solar-wind-events',
        published_at=NOW, issue_time=NOW, artifacts=[validated])
    assert release.issue_time == NOW  # Minute precision, no hourly flooring.
    report = forecast(inputs)
    assert report.events == [] and report.source_status['CME']['state'] == 'unavailable'
    assert report.source_status['L1']['state'] == 'unavailable'


def test_empty_successful_feed_is_distinct_from_missing_or_stale_feed():
    inputs = ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[]),
        solar_wind_events=source(poll('CME', []), poll('HSS', [], NOW-timedelta(hours=2))))
    report = forecast(inputs)
    assert report.source_status['CME']['state'] == 'available'
    assert report.source_status['HSS']['state'] == 'stale'
    assert report.source_status['IPS']['state'] == 'unavailable'


def test_event_inputs_require_no_normalized_data_or_image_preparation(monkeypatch):
    import httpx
    from argus_prophet.services import inputs as loader
    from argus_prophet.services import hss_inputs
    monkeypatch.setattr(hss_inputs, 'enrich_hss_inputs', lambda inputs, *args: inputs)
    monkeypatch.setenv('OBSERVATIONS_URL', 'http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    real_client = httpx.Client
    def handle(request):
        assert request.url.path == '/internal/v1/observations/solar-wind-events/inputs'
        assert request.url.params['as_of'] == NOW.isoformat()
        assert request.headers['Authorization'] == 'Bearer test-token'
        return httpx.Response(200, json=source().model_dump(mode='json'))
    monkeypatch.setattr(loader.httpx, 'Client', lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs))
    result = loader.load_event_inputs(NOW)
    assert result.observations.points == [] and result.solar_wind_events.as_of == NOW


def test_speed_context_failure_preserves_arrival_inputs(monkeypatch):
    import httpx
    from argus_prophet.services import inputs as loader
    monkeypatch.setenv('OBSERVATIONS_URL','http://clio')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN','test-token')
    real=httpx.Client
    def handle(request):
        if request.url.path.endswith('/solar-wind-events/inputs'):
            return httpx.Response(200,json=source().model_dump(mode='json'))
        return httpx.Response(503)
    monkeypatch.setattr(loader.httpx,'Client',lambda **kw:real(transport=httpx.MockTransport(handle),**kw))
    result=loader.load_event_inputs(NOW,include_speed=True)
    assert result.solar_wind_events.as_of==NOW and not result.observations.points


def test_changed_model_fingerprint_disables_only_speed(tmp_path):
    from argus_prophet.services.solar_event_speed import calculate_speed
    models=tmp_path/'data/models';models.mkdir(parents=True)
    (models/'cme.joblib').write_bytes(b'not a valid model')
    inputs=ForecastInputs(as_of=NOW,read_at=NOW,observations=Observation(points=[]),solar_wind_events=source(poll('WSAEnlilSimulations', [])))
    result=calculate_speed(CalculationRequest('solar-wind-events',inputs,tmp_path,
        {'solar_event_speed':{'model':'cme','sha256':'a'*64}}),[])
    assert result['state']=='unavailable'
    assert result['reason']=='ValueError'


def test_CME_only_release_excludes_DBEM_and_verifies_ensemble_count():
    import csv
    import io
    from common.schemas.forecast_release import ForecastArtifact
    from pydantic import ValidationError
    from argus_prophet.services.events import calculate
    inputs = ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[]), solar_wind_events=source(poll('CME', [dict(
        activityID='cme1', startTime='2026-10-13T00:00Z', cmeAnalyses=[dict(
            isMostAccurate=True, featureCode='LE', time21_5='2026-10-13T03:00Z',
            submissionTime='2026-10-13T04:00Z', speed=900, latitude=0, longitude=5, halfAngle=45)])])))
    artifact = calculate(inputs)[0]
    payload = dict(name=artifact.name, csv_text=artifact.content.decode(),
        sha256=hashlib.sha256(artifact.content).hexdigest(), row_count=artifact.row_count,
        columns=artifact.columns, model_info=artifact.model_info)
    validated = ForecastArtifact(**payload)
    rows = list(csv.DictReader(io.StringIO(validated.csv_text)))
    assert rows[0]['cme_ensemble_count'] == '0'
    assert json.loads(rows[0]['report_json'])['cme_ensembles'] == []
    rows[0]['cme_ensemble_count'] = '1'
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=artifact.columns)
    writer.writeheader()
    writer.writerows(rows)
    text = stream.getvalue()
    payload.update(csv_text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValidationError, match='does not match'):
        ForecastArtifact(**payload)



def test_main_speed_scenario_reuses_baseline_and_bundled_model(tmp_path, monkeypatch):
    import pandas as pd
    from common.schemas.forecast_inputs import SpeedObservation, DensityObservation
    from forecast.api import ForecastResult
    from argus_prophet.services import solar_event_speed
    from argus_prophet.services.generation import models
    from argus_prophet.services.generation.calculation import serialize
    import csv
    import io
    src=source(poll('CME',[dict(activityID='cme1',startTime=(NOW-timedelta(hours=4)).isoformat(),
        cmeAnalyses=[dict(speed=900,isMostAccurate=True)])]),
        poll('WSAEnlilSimulations',[simulation(arrival=(NOW+timedelta(hours=12)).isoformat())]))
    inputs=ForecastInputs(as_of=NOW,read_at=NOW,observations=Observation(points=[]),solar_wind_events=src,
        speed_observations=[SpeedObservation(issue_time=NOW.replace(minute=0)-timedelta(hours=h),v=400) for h in range(1,30)],
        density_observations=[DensityObservation(issue_time=NOW.replace(minute=0)-timedelta(hours=h),n=5) for h in range(1,30)])
    issue=NOW.replace(minute=0)
    frame=pd.DataFrame(dict(issue_time=issue,valid_time=pd.date_range(issue+timedelta(hours=1),periods=24,freq='h'),
        lead_hours=range(1,25),v_q10=350.,v_q50=400.,v_q90=450.))
    original=frame.copy()
    baseline=ForecastResult('plasma_speed_quantile',frame,{'model':'same-release-background','sha256':'b'*64})
    registry={'solar_event_speed':{'model':'argus-cme-speed-v2',
        'sha256':'9eee53f8996b044f95d523ac79fc8ca6508a7c6d29ede2fe8d9b923526b48379'}}
    request=CalculationRequest('solar-wind-speed',inputs,tmp_path,registry)
    monkeypatch.setattr(models,'load_model',lambda *a,**kw:pytest.fail('Background must not be recalculated'))
    solar_event_speed.attach_speed_outlook(request,baseline)
    pd.testing.assert_frame_equal(frame[original.columns],original)
    artifact=serialize(baseline)
    ForecastArtifact(name=artifact.name,csv_text=artifact.content.decode(),sha256=hashlib.sha256(artifact.content).hexdigest(),
        row_count=artifact.row_count,columns=artifact.columns,model_info=artifact.model_info)
    rows=list(csv.DictReader(io.StringIO(artifact.content.decode())))
    outlook=json.loads(rows[0]['cme_speed_json'])
    assert outlook['state']=='available' and len(outlook['events'])==1
    assert outlook['background_model']['model']=='same-release-background'
    assert len(outlook['points'])==24 and all(p['speed_background']==400 for p in outlook['points'])
    assert max(p['speed_nominal'] for p in outlook['points'])>400
    assert all(not row['cme_speed_json'] for row in rows[1:])
    # A scenario from another baseline must never be accepted into this release.
    outlook['points'][0]['speed_background']=399
    rows[0]['cme_speed_json']=json.dumps(outlook)
    stream=io.StringIO(); writer=csv.DictWriter(stream,fieldnames=artifact.columns);writer.writeheader();writer.writerows(rows)
    corrupted=stream.getvalue()
    with pytest.raises(ValueError,match='this release'):
        ForecastArtifact(name=artifact.name,csv_text=corrupted,sha256=hashlib.sha256(corrupted.encode()).hexdigest(),
            row_count=artifact.row_count,columns=artifact.columns,model_info=artifact.model_info)


def test_old_clio_snapshot_preserves_background_and_marks_cme_unavailable(tmp_path):
    import pandas as pd
    from forecast.api import ForecastResult
    from argus_prophet.services.solar_event_speed import attach_speed_outlook
    src = source(poll('CME', []))
    src.events = None  # Legacy Clio contract, before normalized events were supplied.
    inputs = ForecastInputs(as_of=NOW, read_at=NOW, observations=Observation(points=[]), solar_wind_events=src)
    frame = pd.DataFrame(dict(valid_time=[NOW+timedelta(hours=1)], v_q50=[400.]))
    result = ForecastResult('plasma_speed_quantile', frame, {})
    attach_speed_outlook(CalculationRequest('solar-wind-speed', inputs, tmp_path, {}), result)
    assert result.frame.v_q50.tolist() == [400.]
    outlook = json.loads(result.frame.cme_speed_json.iloc[0])
    assert outlook['state'] == 'unavailable' and 'Normalized' in outlook['reason']
