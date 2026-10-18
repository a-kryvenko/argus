from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
import json

import numpy as np
import pandas as pd
import pytest

from common.schemas.forecast_release import ForecastArtifact
from common.schemas.solar_wind_events import CMESpeedOutlook
from forecast.calculation import ForecastResult
from prophet.services.generation.calculation import serialize
from prophet.services.events.speed import integrate_speed_forecast


def setup():
    issue = datetime(2026, 1, 18, 22, tzinfo=UTC)
    clock = pd.date_range(issue+timedelta(hours=1), periods=4, freq='h')
    frame = pd.DataFrame(dict(issue_time=issue, valid_time=clock, lead_hours=range(1, 5)))
    q = frame.assign(v_q10=360., v_q50=400., v_q90=440., cme_speed_json='')
    samples = np.linspace(-50, 50, 101)
    p = frame.copy()
    for t in (450, 500, 600):
        p[f'p_v_ge_{t}'] = 1-np.searchsorted(samples, t-400, side='left')/len(samples)
    results = {name: ForecastResult(name, values, dict(model='base', sha256='b'*64))
               for name, values in [('plasma_speed_quantile', q), ('plasma_speed_threshold', p)]}
    bundle = dict(format='dlinear_aia_ridge_v1', thresholds=[450, 500, 600],
                  uncertainty=[dict(lo=1, hi=4, dlinear=samples)])
    services = {'plasma_speed_threshold': SimpleNamespace(models_bundle=bundle)}
    registry = dict(solar_event_speed=dict(sha256='c'*64, integration=dict(enabled=True, version='cme-arrival-window-v2')),
                    plasma_speed_quantile=dict(model='base'), plasma_speed_threshold=dict(model='base'))
    outlook = CMESpeedOutlook(model_version='cme-speed-v2', state='available', model_sha256='c'*64,
        # Before nominal arrival, the window correction can exceed the nominal profile.
        points=[dict(valid_time=t, speed_background=400, speed_nominal=400, speed_early=400,
                     speed_late=400, scenario_lower=400, scenario_upper=650,
                     speed_integrated=500 if j == 1 else 400) for j,t in enumerate(clock)])
    return SimpleNamespace(registry=registry), results, services, outlook


def test_point_intervals_probabilities_and_serialized_contract_change_together():
    request, results, services, outlook = setup()
    before_q = results['plasma_speed_quantile'].frame.copy()
    before_p = results['plasma_speed_threshold'].frame.copy()
    integrate_speed_forecast(request, results, services, outlook)
    q, p = results['plasma_speed_quantile'].frame, results['plasma_speed_threshold'].frame
    np.testing.assert_array_equal(q.v_q50, [400,500,400,400])
    np.testing.assert_array_equal(q.v_q90-q.v_q10, before_q.v_q90-before_q.v_q10)
    assert p.loc[1,'p_v_ge_450'] == 1 and p.loc[1,'p_v_ge_500'] > .49
    pd.testing.assert_frame_equal(p.drop(index=1), before_p.drop(index=1))
    assert json.loads(q.cme_speed_json.iloc[0])['integrated_into_speed']
    for result in results.values():
        assert result.model_info['cme_integration']['version'] == 'cme-arrival-window-v2'
        assert result.model_info['model'].endswith('+cme-window-v2')
        artifact = serialize(result)
        ForecastArtifact(name=artifact.name, csv_text=artifact.content.decode(),
            sha256=__import__('hashlib').sha256(artifact.content).hexdigest(),
            row_count=artifact.row_count, columns=artifact.columns, model_info=artifact.model_info)


def test_no_cme_and_disabled_policy_preserve_predictions():
    request, results, services, outlook = setup()
    request.registry['solar_event_speed']['integration']['enabled'] = False
    before = results['plasma_speed_quantile'].frame.copy()
    integrate_speed_forecast(request, results, services, outlook)
    pd.testing.assert_frame_equal(before, results['plasma_speed_quantile'].frame)
    request.registry['solar_event_speed']['integration']['enabled'] = True
    outlook = CMESpeedOutlook(model_version='cme-speed-v2', state='unavailable')
    before_p = results['plasma_speed_threshold'].frame.copy()
    integrate_speed_forecast(request, results, services, outlook)
    pd.testing.assert_frame_equal(before_p, results['plasma_speed_threshold'].frame)
    pd.testing.assert_frame_equal(before[['v_q10','v_q50','v_q90']],
        results['plasma_speed_quantile'].frame[['v_q10','v_q50','v_q90']])


def test_background_mismatch_rejected_before_changing_outputs():
    request, results, services, outlook = setup()
    before = results['plasma_speed_quantile'].frame.copy()
    outlook.points[0].speed_background = 399
    with pytest.raises(ValueError, match='different background'):
        integrate_speed_forecast(request, results, services, outlook)
    pd.testing.assert_frame_equal(before, results['plasma_speed_quantile'].frame)
