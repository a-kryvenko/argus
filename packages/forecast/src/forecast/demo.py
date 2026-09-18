"""Deterministic wiring example with synthetic observations and constant toy models.

Run with ``python -m forecast.demo``. These values are not operational forecasts
or evidence of model accuracy; no fitted production models or private code are used.
"""
from datetime import UTC, datetime, timedelta
import json

import numpy as np
from common.schemas.observation import Observation, ObservationPoint
from forecast.api import SWSpeedFS, SWSpeedProbaFS, SWDensityFS, calculate_forecast

ISSUE_TIME = datetime(2026, 1, 1, 12, tzinfo=UTC)


class ConstantRegressor:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value, dtype=float)


class ConstantClassifier:
    classes_ = np.array([0, 1])

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        return np.tile([1 - self.probability, self.probability], (len(frame), 1))


def run_demo():
    observations = Observation(points=[
        ObservationPoint(issue_time=ISSUE_TIME - timedelta(hours=7-i),
                         bx=1, by=2, bz=-3, v=400+i, n=5, t=100000,
                         kp=2, ap=5, dst=-10, f10_7=120) for i in range(8)
    ])
    buckets = [(1, 4), (2, 6)]
    def quantiles(values):
        return {'lead_hours': 6, 'buckets': buckets, 'feature_columns': ['v', 'lead_hours'],
                'models': {(bucket, q): ConstantRegressor(value)
                           for bucket in buckets for q, value in zip(('q10', 'q50', 'q90'), values)}}
    speed = SWSpeedFS(quantiles((350, 420, 500)))
    density = SWDensityFS(quantiles((3, 5, 8)))
    thresholds = SWSpeedProbaFS({
        'lead_hours': 6, 'buckets': [(6, 'demo')], 'feature_columns': ['v', 'lead_hours'],
        'models': {(threshold, 'demo'): ConstantClassifier(probability)
                   for threshold, probability in ((450, 0.3), (500, 0.1), (600, 0.01))},
    })
    return [calculate_forecast(service, observations, issue_time=ISSUE_TIME,
                               model_info={'kind': 'synthetic_demo', 'registry_name': service.registry_name})
            for service in (speed, thresholds, density)]


if __name__ == '__main__':
    print(json.dumps({result.name: result.frame.to_dict(orient='records') for result in run_demo()},
                     default=str, indent=2))
