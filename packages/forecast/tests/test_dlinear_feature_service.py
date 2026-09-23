from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from common.schemas.observation import Observation, ObservationPoint
from forecast.api import SWSpeedFS


class EchoFeature:
    def predict(self, frame):
        return frame.dlinear_v.to_numpy()


def bundle():
    dependency = dict(format='rotation_dlinear', version=1,
                      settings=dict(columns=['v'], segments={'rotations': [[-2, 0]]},
                                    horizon=3, ffill_limit_hours=1, mean=[100.], std=[10.]),
                      weights=np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]), bias=np.zeros(3))
    return dict(lead_hours=3, buckets=[(3, 'all')], feature_columns=['dlinear_v'],
                feature_models={'dlinear_v': {'bundle': dependency, 'sha256': 'test'}},
                models={('all', q): EchoFeature() for q in ['q10', 'q50', 'q90']})


def observations():
    start = datetime(2025, 1, 1, tzinfo=UTC)
    return Observation(points=[ObservationPoint(issue_time=start+timedelta(hours=i),
        bx=1, by=2, bz=-3, v=100+i*10, n=5, t=10000, kp=2, dst=-20, ap=5, f10_7=120)
        for i in range(8)])


def test_runtime_uses_embedded_dlinear_per_lead_at_forecast_hour():
    obs = observations()
    issue = obs.points[-1].issue_time + timedelta(minutes=17)
    service = SWSpeedFS(bundle())
    result = service.forecast(obs, issue_time=issue)
    np.testing.assert_allclose([p.v_q50 for p in result.points], [170., 150., 160.])
    assert result.issue_time == issue
    assert result.points[0].valid_time == issue.replace(minute=0) + timedelta(hours=1)
    # Future speed must not affect the DLinear feature.
    obs.points.append(obs.points[-1].model_copy(update={'issue_time': issue+timedelta(hours=1), 'v': 9999.}))
    future_result = service.forecast(obs, issue_time=issue)
    np.testing.assert_allclose([p.v_q50 for p in future_result.points], [170., 150., 160.])


def test_missing_dependency_and_expired_history_fail_explicitly():
    b = bundle()
    del b['feature_models']
    with pytest.raises(ValueError, match='embed'):
        SWSpeedFS(b)
    obs = observations()
    with pytest.raises(ValueError, match='Insufficient hourly speed history'):
        SWSpeedFS(bundle()).forecast(obs, issue_time=obs.points[-1].issue_time+timedelta(days=3))


def test_explicit_raw_speed_history_overrides_interpolated_wide_observations():
    import pandas as pd
    obs = observations()
    history = pd.DataFrame({'issue_time': [p.issue_time for p in obs.points],
                             'v': [p.v for p in obs.points]})
    issue = obs.points[-1].issue_time
    for point in obs.points:
        point.v = 9999.  # Stand-in for independently processed/filled wide data.
    result = SWSpeedFS(bundle()).forecast(obs, issue_time=issue, speed_history=history)
    np.testing.assert_allclose([p.v_q50 for p in result.points], [170., 150., 160.])
    with pytest.raises(ValueError, match='Insufficient hourly speed history'):
        SWSpeedFS(bundle()).forecast(obs, issue_time=issue, speed_history=history.iloc[:0])
