"""Persist selected native samples once, in the common measurement table."""
import math

from sqlalchemy import or_
from sqlalchemy.dialects.postgresql import insert

from clio.db.models.measurement import Measurement
from clio.providers.solar_wind_loader import FIELDS


def wind_measurements(kind, records):
    selected = {}
    for record in sorted(records, key=lambda r: r['received_at'], reverse=True):
        if record['active']:
            selected.setdefault(record['observed_at'], record)
    return [dict(metric=metric, observed_at=at, value=record['values'].get(metric),
                 received_at=record['received_at'], source_product=f'swpc.rtsw_{kind}',
                 quality='missing' if record['values'].get(metric) is None else
                         'flagged' if record['raw'].get('overall_quality') not in (None, 0) else 'unverified')
            for at, record in selected.items() for metric in FIELDS[kind]]


def geomagnetic_measurements(metric, records):
    rows = []
    for record in records:
        values = {metric: record['value']}
        if metric == 'kp':
            value = record['raw'].get('a_running')
            try:
                value = None if isinstance(value, bool) else float(value)
                if value is not None and (not math.isfinite(value) or value < 0):
                    value = None
            except (ValueError, TypeError):
                value = None
            values['ap'] = value
        for name, value in values.items():
            rows.append(dict(metric=name, value=value, observed_at=record['interval_start'],
                             received_at=record['received_at'], source_product=f'swpc.{metric}',
                             quality='missing' if value is None else record['quality']))
    return rows


async def store_native(session, records):
    # New polling timestamps alone do not revise an observation's receipt.
    fields = ('value', 'source_product', 'quality')
    for offset in range(0, len(records), 1000):
        statement = insert(Measurement).values(records[offset:offset + 1000])
        await session.execute(statement.on_conflict_do_update(
            constraint='uq_measurement_metric',
            set_={name: statement.excluded[name] for name in (*fields, 'received_at')},
            where=or_(*(getattr(Measurement, name).is_distinct_from(statement.excluded[name]) for name in fields))))
