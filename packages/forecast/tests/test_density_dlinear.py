import copy
import sys

import numpy as np
import pandas as pd
import pytest

from common.schemas.forecast_inputs import DensityObservation, ForecastInputs
from common.schemas.observation import Observation
from forecast.api import DensityDLinearForecaster, SWDensityFS, calculate_snapshot


def bundle():
    return {
        'format': 'density_dlinear', 'version': 1,
        'settings': {
            'columns': ['n'], 'segments': {'rotations': [[-2, 0]]},
            'horizon': 2, 'ffill_limit_hours': 1, 'mean': [4.], 'std': [2.],
            'scale': 'original', 'point_postprocessing': 'maximum(point, 0)',
            'quantile_postprocessing': 'maximum(point + residual_quantile, 0)',
        },
        'weights': np.array([[0., 0., 1.], [1., 0., 0.]]),
        'bias': np.zeros(2), 'quantiles': np.array([.1, .5, .9]),
        'residual_offsets': np.array([[-8., .2, 3.], [-1., -.1, 2.]]),
    }


def history():
    return pd.DataFrame({'issue_time': pd.date_range('2025-01-01', periods=4, freq='h', tz='UTC'),
                         'n': [3., 4., 5., 100.]})


def test_affine_quantiles_nonnegative_and_no_future_inputs(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    obs = history()
    model = DensityDLinearForecaster(bundle())
    actual = model.frame(obs.issue_time[2], obs)
    np.testing.assert_allclose(actual[['n_q10', 'n_q50', 'n_q90']], [[0., 5.2, 8.], [2., 2.9, 5.]])
    assert actual.lead_hours.tolist() == [1, 2]
    assert actual.valid_time.iloc[0] == obs.issue_time[2] + pd.Timedelta(1, unit="h")
    obs.loc[3, 'n'] = -1e9
    pd.testing.assert_frame_equal(actual, model.frame(obs.issue_time[2], obs))


@pytest.mark.parametrize('invalid', [999.9, -1., np.inf, -np.inf, np.nan])
def test_bounded_fill_and_history_error(invalid):
    obs = history()
    model = DensityDLinearForecaster(bundle())
    obs.loc[2, 'n'] = invalid
    assert model.frame(obs.issue_time[2], obs).dlinear_n.iloc[0] == 4.
    obs.loc[1, 'n'] = invalid
    with pytest.raises(ValueError, match='Insufficient hourly density history'):
        model.frame(obs.issue_time[2], obs)


def test_valid_large_values_are_not_clipped_like_speed():
    obs = history()
    obs.loc[2, 'n'] = 1001.
    assert DensityDLinearForecaster(bundle()).frame(obs.issue_time[2], obs).dlinear_n.iloc[0] == 1001.


@pytest.mark.parametrize('change', ['shape', 'nan', 'crossing', 'scale', 'quantiles', 'future', 'postprocessing'])
def test_rejects_incompatible_artifacts(change):
    b = copy.deepcopy(bundle())
    if change == 'shape': b['residual_offsets'] = np.zeros((1, 3))
    if change == 'nan': b['residual_offsets'][0, 0] = np.nan
    if change == 'crossing': b['residual_offsets'][0, 0] = 20
    if change == 'scale': b['settings']['scale'] = 'log'
    if change == 'quantiles': b['quantiles'] = [.05, .5, .95]
    if change == 'future': b['settings']['segments']['rotations'] = [[0, 2]]
    if change == 'postprocessing': b['settings']['point_postprocessing'] = 'none'
    with pytest.raises(ValueError):
        DensityDLinearForecaster(b)


def test_snapshot_round_trip_uses_raw_density_not_normalized_observations():
    obs = history().iloc[:3]
    issue = obs.issue_time.iloc[-1].to_pydatetime()
    inputs = ForecastInputs(as_of=issue, read_at=issue, observations=Observation(points=[]),
                            density_observations=[DensityObservation(**r) for r in obs.to_dict('records')])
    # Same JSON boundary used by Clio HTTP and saved Prophet snapshots.
    inputs = ForecastInputs.model_validate_json(inputs.model_dump_json())
    service = SWDensityFS(bundle())
    result = calculate_snapshot(service, inputs, issue_time=issue, model_info={'model': 'test'})
    np.testing.assert_allclose(result.frame.n_q50, [5.2, 2.9])
    assert result.name == 'plasma_density_quantile'
    with pytest.raises(ValueError, match='Insufficient hourly density history'):
        service.forecast_snapshot(inputs.model_copy(update={'density_observations': []}), issue_time=issue)
    with pytest.raises(ValueError, match='unfilled density_history'):
        service.forecast(inputs.observations, issue_time=issue)
