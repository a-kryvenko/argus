import hashlib
import json
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from common.schemas.solar_wind_events import EventInputs, SourcePoll
from clio.normalization.solar_wind_events import normalize_events as external_events

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


def test_initial_speed_uses_causal_analysis_and_merged_receipt_time():
    received = NOW-timedelta(minutes=5)
    cme = dict(activityID='cme1', startTime='2026-10-12T00:00Z', cmeAnalyses=[
        dict(speed=800, isMostAccurate=False), dict(speed=900, isMostAccurate=True),
        dict(speed=1900, isMostAccurate=True, submissionTime='2026-10-14T00:00Z')])
    event = external_events(source(poll('CME', [cme]), poll('WSAEnlilSimulations', [simulation()], received)))[0]
    assert event.initial_speed_km_s == 900 and event.simulation_id == 'run1'
    assert event.available_at == NOW


def test_latest_full_poll_wins_independently_of_input_order():
    old = poll('WSAEnlilSimulations', [simulation()], NOW-timedelta(minutes=1))
    replacement = poll('WSAEnlilSimulations', [], NOW)
    assert external_events(source(replacement, old)) == []


