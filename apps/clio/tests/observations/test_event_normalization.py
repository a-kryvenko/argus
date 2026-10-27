import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from common.schemas.solar_wind_events import EventInputs, SourcePoll
from clio.observations.solar_wind_events import normalize_events as external_events, normalize_event_inputs

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


def test_external_predictions_are_inputs_without_forecast_status_or_quantiles():
    observations, predictions = normalize_event_inputs(source(poll('WSAEnlilSimulations', [simulation()])))
    assert observations == [] and len(predictions) == 1
    prediction = predictions[0]
    assert prediction.cme_ids == ['cme1'] and prediction.glancing_blow
    assert prediction.earth_arrival_at == datetime(2026, 10, 14, 12, tzinfo=UTC)
    assert not {'status', 'probability', 'arrival_q10'}.intersection(prediction.model_dump())
    assert len(prediction.receipts) == 1 and prediction.receipts[0].sha256


def test_alternatives_and_retractions_are_retained_for_prophet_selection():
    rows = [simulation(), simulation(arrival=None, completed='2026-10-13T11:00Z', link='run2')]
    observations, predictions = normalize_event_inputs(source(poll('WSAEnlilSimulations', rows)))
    assert observations == [] and len(predictions) == 2
    assert predictions[1].simulation_id == 'run2' and predictions[1].earth_arrival_at is None
    _, combined = normalize_event_inputs(source(poll('WSAEnlilSimulations', [simulation(('cme1', 'cme2'))])))
    assert combined[0].cme_ids == ['cme1', 'cme2']


def test_nested_simulations_restore_missing_standalone_ids_without_order_dependency():
    nested = simulation()
    nested.pop('cmeInputs')
    nested['cmeIDs'] = ['cme1']
    cme = dict(activityID='cme1', startTime='2026-10-12T00:00Z',
               cmeAnalyses=[dict(enlilList=[nested])])
    observations, predictions = normalize_event_inputs(source(poll('WSAEnlilSimulations', [simulation(ids=())]), poll('CME', [cme])))
    assert len(observations) == 1 and observations[0].status == 'detected'
    assert predictions[0].cme_ids == ['cme1'] and len(predictions[0].receipts) == 2
    _, unknown = normalize_event_inputs(source(poll('WSAEnlilSimulations', [simulation(ids=())])))
    assert unknown[0].cme_ids == []


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
    observations, predictions = normalize_event_inputs(source(poll('WSAEnlilSimulations', [row], received)))
    assert observations == []
    if change == 'invalid_arrival':
        assert len(predictions) == 1 and predictions[0].invalid_reason == 'invalid_arrival_time'
    else:
        assert predictions == []


def test_hss_and_ips_are_observations_only_and_other_targets_are_excluded():
    hss = dict(hssID='hss1', eventTime='2026-10-13T01:00Z', instruments=[{'displayName': 'ACE: SWEPAM'}])
    ips = dict(activityID='ips1', eventTime='2026-10-13T01:00Z', location='STEREO A')
    result = external_events(source(poll('HSS', [hss]), poll('IPS', [ips])))
    assert len(result) == 1 and result[0].kind == 'hss' and result[0].status == 'observed'
    assert result[0].arrival_at is None
    hss['eventTime'] = '2026-10-14T01:00Z'
    assert external_events(source(poll('HSS', [hss]))) == []


def test_initial_speed_uses_causal_analysis_and_merged_receipt_time():
    received = NOW-timedelta(minutes=5)
    cme = dict(activityID='cme1', startTime='2026-10-12T00:00Z', cmeAnalyses=[
        dict(speed=800, isMostAccurate=False), dict(speed=900, isMostAccurate=True),
        dict(speed=1900, isMostAccurate=True, submissionTime='2026-10-14T00:00Z')])
    observations, predictions = normalize_event_inputs(source(poll('CME', [cme]), poll('WSAEnlilSimulations', [simulation()], received)))
    assert observations[0].initial_speed_km_s == 900 and observations[0].available_at == NOW
    assert predictions[0].simulation_id == 'run1' and predictions[0].available_at == received


def test_latest_full_poll_wins_independently_of_input_order():
    old = poll('WSAEnlilSimulations', [simulation()], NOW-timedelta(minutes=1))
    replacement = poll('WSAEnlilSimulations', [], NOW)
    assert normalize_event_inputs(source(replacement, old)) == ([], [])




def test_clio_does_not_apply_prophet_forecast_horizon():
    _, predictions = normalize_event_inputs(source(poll('WSAEnlilSimulations', [simulation(arrival='2026-11-01T00:00Z')])))
    assert len(predictions) == 1
