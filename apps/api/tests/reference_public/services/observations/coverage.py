"""Native UTC slot coverage; never infer why a source record is absent."""
import math
from datetime import UTC, datetime


def coverage(points, start, end, resolution, *, intervals=False, now=None):
    end = min(end, now or datetime.now(UTC))
    first = (math.floor if intervals else math.ceil)(start.timestamp()/resolution)
    stop = math.ceil(end.timestamp()/resolution) if end > start else first
    slots = {slot: 'missing' for slot in range(first, max(first, stop))}
    for point in points:
        slot = math.floor(point['interval_start' if intervals else 'observed_at'].timestamp()/resolution)
        if slot not in slots:
            continue
        state = 'usable' if point['value'] is not None and point['quality'] not in ('flagged', 'missing') else 'invalid'
        if slots[slot] != 'usable':
            slots[slot] = state
    gaps = []
    for slot, state in slots.items():
        if state == 'usable':
            continue
        left = max(start, datetime.fromtimestamp(slot*resolution, UTC))
        right = min(end, datetime.fromtimestamp((slot+1)*resolution, UTC))
        if gaps and gaps[-1]['to'] == left and gaps[-1]['reason'] == state:
            gaps[-1]['to'] = right
            gaps[-1]['slots'] += 1
        else:
            gaps.append({'from': left, 'to': right, 'reason': state, 'slots': 1})
    usable = sum(state == 'usable' for state in slots.values())
    missing = sum(state == 'missing' for state in slots.values())
    return {'expected_slots': len(slots), 'usable_slots': usable, 'missing_slots': missing,
            'invalid_slots': len(slots)-usable-missing,
            'percent': round(100*usable/len(slots), 2) if slots else None,
            'resolution_seconds': resolution, 'evaluated_to': end,
            'basis': 'overlapping_intervals' if intervals else 'sample_timestamps', 'gaps': gaps}
