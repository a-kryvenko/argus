"""Convert source samples into the common measurement format."""
import math

import pandas as pd

from clio.observations.schema import OBSERVATION_METRICS
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


def wide_to_measurements(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp_column = "issue_time" if "issue_time" in frame else "observed_at"
    metrics = [metric for metric in OBSERVATION_METRICS if metric in frame]
    if timestamp_column not in frame or not metrics:
        return pd.DataFrame(columns=["metric", "value", "observed_at"])

    return (
        frame
        .melt(
            id_vars=timestamp_column,
            value_vars=metrics,
            var_name="metric",
            value_name="value",
        )
        .rename(columns={timestamp_column: "observed_at"})
        .dropna(subset=["observed_at", "value"])
    )
