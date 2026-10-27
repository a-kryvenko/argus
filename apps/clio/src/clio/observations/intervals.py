"""Fixed geomagnetic intervals derived from measurement timestamps."""
from datetime import timedelta

from sqlalchemy import case

from clio.db.models.measurement import Measurement

INTERVALS = {'kp': timedelta(hours=3), 'ap': timedelta(hours=3), 'dst': timedelta(hours=1)}


def interval_end(metric, observed_at):
    duration = INTERVALS.get(metric)
    return observed_at + duration if duration is not None else None


def interval_end_expression():
    return case(*((Measurement.metric == metric, Measurement.observed_at + duration)
                  for metric, duration in INTERVALS.items()), else_=Measurement.observed_at)
