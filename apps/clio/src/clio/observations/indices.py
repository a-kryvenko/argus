"""Read current source indices and their changes over one hour."""
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, tuple_

from clio.db.models.measurement import Measurement
from clio.observations.intervals import interval_end
from clio.observations.summary import value_change


async def current_indices(session, metrics, *, trends=False, now=None):
    # Legacy stored S10/M10/Y10 estimates are not source observations.
    now = now or datetime.now(UTC)
    statement = (select(Measurement)
                 .where(Measurement.metric.in_(metrics), Measurement.observed_at <= now,
                        tuple_(Measurement.metric, Measurement.source_product).in_([
                            ('ap', 'swpc.kp'), ('f10_7', 'swpc.f107'), ('f10_7', 'gfz.f107'),
                        ]))
                 .distinct(Measurement.metric)
                 .order_by(Measurement.metric, Measurement.observed_at.desc()))
    records = (await session.execute(statement)).scalars().all()
    series = dict.fromkeys(metrics)
    for record in records:
        point = {'observed_at': record.observed_at, 'value': record.value,
                 'quality': record.quality or ('missing' if record.value is None else None)}
        end = interval_end(record.metric, record.observed_at)
        if end is not None:
            point['interval_end'] = end
        series[record.metric] = point
    data = {'series': series}
    if trends:
        previous_records = (await session.execute(statement.where(
            Measurement.observed_at <= now-timedelta(hours=1)))).scalars().all()
        previous = {record.metric: {'value': record.value, 'quality': record.quality} for record in previous_records}
        data['changes_1h'] = {metric: value_change(series[metric], previous.get(metric)) for metric in metrics}
    return data
