"""Paginated reads of stored raw and hourly observations."""
from sqlalchemy import select, func

from clio.db.models.measurement import Measurement
from clio.db.models.normalized_observation import NormalizedObservation


async def browse(db, *, kind, page, page_size, order, start=None, end=None, metric=None):
    model = Measurement if kind == 'raw' else NormalizedObservation
    filters = []
    if start:
        filters.append(model.observed_at >= start)
    if end:
        filters.append(model.observed_at <= end)
    if metric and kind == 'raw':
        filters.append(Measurement.metric == metric)
    ordering = [model.observed_at.asc() if order == 'asc' else model.observed_at.desc()]
    if kind == 'raw':
        ordering.append(Measurement.id.asc() if order == 'asc' else Measurement.id.desc())
    rows = (await db.scalars(select(model).where(*filters).order_by(*ordering).offset((page-1)*page_size).limit(page_size))).all()
    columns = [c.name for c in model.__table__.columns]
    return {'columns': columns, 'items': [{c: getattr(row, c) for c in columns} for row in rows],
        'total': await db.scalar(select(func.count()).select_from(model).where(*filters)), 'page': page, 'page_size': page_size}
