import copy
import sys

import numpy as np
import pandas as pd
import pytest

from forecast.api import RotationDLinearForecaster


def bundle():
    return {
        'format': 'rotation_dlinear', 'version': 1,
        'settings': {
            'columns': ['v'], 'segments': {'rotations': [[-2, 0]]},
            'horizon': 2, 'ffill_limit_hours': 1, 'mean': [100.], 'std': [10.],
        },
        'weights': np.array([[0., 0., 1.], [1., 0., 0.]]),
        'bias': np.zeros(2),
    }


def observations():
    return pd.DataFrame({'issue_time': pd.date_range('2025-01-01', periods=5, freq='h', tz='UTC'),
                         'v': [100., 110., 120., 130., 140.]})


def test_order_index_normalization_and_future_independence():
    model = RotationDLinearForecaster(bundle(), batch_size=1)
    obs = observations()
    request = pd.DataFrame({'issue_time': [obs.issue_time[3], obs.issue_time[2], obs.issue_time[3]],
                            'lead_hours': [2, 1, 1]}, index=[7, 2, 7])
    original = request.copy()
    actual = model.add_rotation_v(request, obs)
    np.testing.assert_allclose(actual.rotation_v, [110., 120., 130.])
    pd.testing.assert_frame_equal(request, original)
    assert actual.index.equals(request.index)
    obs.loc[4, 'v'] = 99999.
    np.testing.assert_allclose(model.add_rotation_v(request, obs).rotation_v, actual.rotation_v)


def test_fill_limit_and_missing_history():
    model = RotationDLinearForecaster(bundle())
    obs = observations()
    request = pd.DataFrame({'issue_time': [obs.issue_time[2]], 'lead_hours': [1]})
    obs.loc[2, 'v'] = np.nan
    assert model.add_rotation_v(request, obs).rotation_v.iloc[0] == 110.
    obs.loc[1, 'v'] = np.nan
    assert model.add_rotation_v(request, obs).rotation_v.isna().all()
    assert model.add_rotation_v(request, obs.iloc[3:]).rotation_v.isna().all()


@pytest.mark.parametrize('lead', [0, 3, 1.5, np.nan])
def test_reject_invalid_lead(lead):
    obs = observations()
    request = pd.DataFrame({'issue_time': [obs.issue_time[2]], 'lead_hours': [lead]})
    with pytest.raises(ValueError, match='lead_hours'):
        RotationDLinearForecaster(bundle()).add_rotation_v(request, obs)


def test_conflicting_observations():
    obs = observations()
    obs = pd.concat([obs, obs.iloc[[2]].assign(v=999.)])
    request = pd.DataFrame({'issue_time': [obs.issue_time.iloc[2]], 'lead_hours': [1]})
    with pytest.raises(ValueError, match='Conflicting'):
        RotationDLinearForecaster(bundle()).add_rotation_v(request, obs)


@pytest.mark.parametrize('change', ['version', 'shape', 'future', 'nan'])
def test_artifact_validation(change):
    b = copy.deepcopy(bundle())
    if change == 'version': b['version'] = 99
    if change == 'shape': b['weights'] = np.zeros((1, 3))
    if change == 'future': b['settings']['segments']['rotations'] = [[0, 2]]
    if change == 'nan': b['bias'][0] = np.nan
    with pytest.raises(ValueError):
        RotationDLinearForecaster(b)


def test_joblib_registry_round_trip_without_torch(tmp_path, monkeypatch):
    joblib = pytest.importorskip('joblib')
    monkeypatch.setitem(sys.modules, 'torch', None)
    path = tmp_path / 'model.joblib'
    joblib.dump(bundle(), path)
    model = RotationDLinearForecaster.from_registry(
        workdir=tmp_path, registry={'plasma_speed_dlinear': {'artifact_path': 'model.joblib'}})
    obs = observations()
    request = pd.DataFrame({'issue_time': [obs.issue_time[2]], 'lead_hours': [1]})
    assert model.add_rotation_v(request, obs).rotation_v.iloc[0] == 120.
