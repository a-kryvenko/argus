import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import pytest

from common.schemas.forecast_inputs import ForecastInputs
from common.schemas.forecast_release import ForecastArtifact, ForecastRelease
from common.schemas.observation import Observation
from common.schemas.solar_wind_events import EventInputs, L1Sample, SourcePoll
from argus_prophet.services.events import external_events, forecast, l1_detections
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
    return EventInputs(as_of=NOW, read_at=NOW, polls=list(polls), l1_samples=list(samples))


def test_external_prediction_has_no_invented_probabilities_or_quantiles():
    result = external_events(source(poll('WSAEnlilSimulations', [simulation()])))
    assert len(result) == 1
    event = result[0]
    assert event.event_id == 'cme1' and event.status == 'predicted_hit'
    assert event.arrival_target == 'Earth' and event.glancing_blow
    assert 'probability' not in event.model_dump() and 'arrival_q10' not in event.model_dump()


def test_latest_simulation_can_retract_hit_and_multiple_cmes_stay_a_group():
    rows = [simulation(), simulation(arrival=None, completed='2026-10-13T11:00Z', link='run2')]
    result = external_events(source(poll('WSAEnlilSimulations', rows)))
    assert len(result) == 1 and result[0].status == 'no_predicted_hit'
    combined = external_events(source(poll('WSAEnlilSimulations', [simulation(('cme1', 'cme2'))])))
    assert len(combined) == 1 and combined[0].source_ids == ['cme1', 'cme2']


def test_nested_simulations_restore_missing_standalone_ids_without_order_dependency():
    nested = simulation()
    nested.pop('cmeInputs')
    nested['cmeIDs'] = ['cme1']
    cme = dict(activityID='cme1', startTime='2026-10-12T00:00Z',
               cmeAnalyses=[dict(enlilList=[nested])])
    result = external_events(source(poll('WSAEnlilSimulations', [simulation(ids=())]), poll('CME', [cme])))
    assert len(result) == 1 and result[0].kind == 'cme' and result[0].event_id == 'cme1'
    unknown = external_events(source(poll('WSAEnlilSimulations', [simulation(ids=())])))
    assert unknown[0].kind == 'solar_wind_disturbance'


@pytest.mark.parametrize('change', ['receipt', 'completion', 'submission', 'invalid_arrival'])
def test_future_or_invalid_information_is_not_used(change):
    row = simulation()
    received = NOW
    if change == 'receipt':
        received += timedelta(minutes=1)
    if change == 'completion':
        row['modelCompletionTime'] = '2026-10-13T13:00Z'
    if change == 'submission':
        row['submissionTime'] = '2026-10-13T13:00Z'
    if change == 'invalid_arrival':
        row['estimatedShockArrivalTime'] = 'invalid'
    assert external_events(source(poll('WSAEnlilSimulations', [row], received))) == []


def test_hss_and_ips_are_observations_only_and_other_targets_are_excluded():
    hss = dict(hssID='hss1', eventTime='2026-10-13T01:00Z', instruments=[{'displayName': 'ACE: SWEPAM'}])
    ips = dict(activityID='ips1', eventTime='2026-10-13T01:00Z', location='STEREO A')
    result = external_events(source(poll('HSS', [hss]), poll('IPS', [ips])))
    assert len(result) == 1 and result[0].kind == 'hss' and result[0].status == 'observed'
    assert result[0].arrival_at is None
    hss['eventTime'] = '2026-10-14T01:00Z'
    assert external_events(source(poll('HSS', [hss]))) == []


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
    events = external_events(source(poll('WSAEnlilSimulations', [simulation(arrival=onset.isoformat())])))
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
    inputs=ForecastInputs(as_of=NOW,read_at=NOW,observations=Observation(points=[]),solar_wind_events=source())
    result=calculate_speed(CalculationRequest('solar-wind-events',inputs,tmp_path,
        {'solar_event_speed':{'model':'cme','sha256':'a'*64}}),[])
    assert result['state']=='unavailable'


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

