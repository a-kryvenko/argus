"""Normalize versioned external DONKI/ENLIL data; no Argus forecasting."""
import math
from datetime import UTC, datetime

from common.schemas.solar_wind_events import SolarWindEvent, EnlilArrivalInput, ArrivalSourceReceipt


def timestamp(value):
    if not value or not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace('Z', '+00:00'))
        return result.astimezone(UTC) if result.tzinfo else None
    except ValueError:
        return None


def positive(value):
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 else None
    except (TypeError, ValueError):
        return None


def published(row, cutoff):
    value = row.get('submissionTime')
    return value is None or (timestamp(value) is not None and timestamp(value) <= cutoff)


def simulation_key(row):
    # Nested CME responses and standalone simulations use the same DONKI link.
    return row.get('link') or row.get('simulationID')


def normalize_event_inputs(inputs):
    cutoff = inputs.as_of
    polls = {}
    for poll in sorted(inputs.polls, key=lambda p: (p.received_at, str(p.poll_id))):
        if poll.received_at <= cutoff:
            polls[poll.endpoint] = poll
    events, simulations = {}, {}
    receipts = {}
    for endpoint, poll in sorted(polls.items()):
        for row in poll.records:
            if not published(row, cutoff):
                continue
            if endpoint == 'CME':
                event_id, onset = row.get('activityID'), timestamp(row.get('startTime'))
                if not event_id or onset is None or onset > cutoff:
                    continue
                events[event_id] = SolarWindEvent(event_id=event_id, kind='cme', status='detected',
                    source='DONKI', source_ids=[event_id], source_url=row.get('link'),
                    available_at=poll.received_at, observed_at=onset)
                analyses = [a for a in row.get('cmeAnalyses') or [] if published(a, cutoff) and positive(a.get('speed')) is not None]
                if analyses:
                    best = max(analyses, key=lambda a: (bool(a.get('isMostAccurate')), timestamp(a.get('submissionTime')) or datetime(1970, 1, 1, tzinfo=UTC)))
                    events[event_id].initial_speed_km_s = positive(best['speed'])
                for analysis in row.get('cmeAnalyses') or []:
                    if not published(analysis, cutoff):
                        continue
                    for simulation in analysis.get('enlilList') or []:
                        key = simulation_key(simulation)
                        if key:
                            receipts.setdefault(key, {})[poll.poll_id] = ArrivalSourceReceipt(poll_id=poll.poll_id, sha256=poll.sha256, received_at=poll.received_at)
                            old = simulations.get(key)
                            ids = set((old[0].get('cmeIDs') or []) if old else [])
                            ids.update(simulation.get('cmeIDs') or [event_id])
                            simulations[key] = ({**simulation, 'cmeIDs': sorted(ids)}, poll.received_at)
            elif endpoint == 'WSAEnlilSimulations':
                for_simulation = simulation_key(row)
                if for_simulation:
                    receipts.setdefault(for_simulation, {})[poll.poll_id] = ArrivalSourceReceipt(poll_id=poll.poll_id, sha256=poll.sha256, received_at=poll.received_at)
                    old = simulations.get(for_simulation)
                    ids = [c['CMEID'] for c in row.get('cmeInputs') or [] if c.get('CMEID')]
                    ids = ids or (old[0].get('cmeIDs', []) if old else [])
                    received = max(poll.received_at, old[1]) if old and not row.get('cmeInputs') else poll.received_at
                    simulations[for_simulation] = ({**row, 'cmeIDs': ids}, received)
            elif endpoint in ('HSS', 'IPS'):
                event_id = row.get('hssID') if endpoint == 'HSS' else row.get('activityID')
                onset = timestamp(row.get('eventTime'))
                # HSS currently has no location field; require an identified L1 instrument.
                near_earth = row.get('location') in ('Earth', 'L1')
                if endpoint == 'HSS' and row.get('location') is None:
                    near_earth = any(i.get('displayName', '').split(':')[0] in
                        ('ACE', 'DSCOVR', 'Wind', 'SOLAR-1', 'IMAP') for i in row.get('instruments') or [])
                if not event_id or onset is None or onset > cutoff or not near_earth:
                    continue
                linked = [x['activityID'] for x in row.get('linkedEvents') or [] if x.get('activityID')]
                events[event_id] = SolarWindEvent(event_id=event_id,
                    kind='hss' if endpoint == 'HSS' else 'interplanetary_shock', status='observed',
                    source='DONKI', source_ids=linked, source_url=row.get('link'),
                    available_at=poll.received_at, observed_at=onset)

    # Retain external alternatives and retractions without selecting a version.
    predictions = []
    for key, (row, received) in sorted(simulations.items()):
        completed = timestamp(row.get('modelCompletionTime'))
        if completed is None or completed > cutoff:
            continue
        arrival = timestamp(row.get('estimatedShockArrivalTime'))
        invalid = ('invalid_arrival_time' if row.get('estimatedShockArrivalTime') is not None and arrival is None
                   else 'arrival_precedes_completion' if arrival is not None and arrival < completed else None)
        predictions.append(EnlilArrivalInput(simulation_id=key,
            cme_ids=sorted(set(row.get('cmeIDs') or [])),
            available_at=max(received, completed), model_completed_at=completed,
            earth_arrival_at=arrival, source_url=row.get('link'),
            duration_hours=positive(row.get('estimatedDuration')),
            glancing_blow=row.get('isEarthGB'), minor_impact=row.get('isEarthMinorImpact'),
            invalid_reason=invalid,
            receipts=sorted(receipts[key].values(), key=lambda r: (r.received_at, str(r.poll_id)))))
    observations = sorted(events.values(), key=lambda e: (e.observed_at or cutoff, e.event_id))
    return observations, predictions


def normalize_events(inputs):
    """Normalize catalog observations only."""
    return normalize_event_inputs(inputs)[0]
