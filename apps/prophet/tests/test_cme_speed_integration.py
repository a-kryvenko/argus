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


@pytest.mark.parametrize('version', ['cme-arrival-window-v2', 'cme-arrival-arc-v3'])
def test_point_intervals_probabilities_and_serialized_contract_change_together(version):
    request, results, services, outlook = setup()
    request.registry['solar_event_speed']['integration']['version'] = version
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
        assert result.model_info['cme_integration']['version'] == version
        assert result.model_info['model'].endswith('+cme-arc-v3' if version == 'cme-arrival-arc-v3' else '+cme-window-v2')
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


@pytest.mark.parametrize('mode', ['blend', 'fallback'])
@pytest.mark.parametrize('model_version', ['proswin-fold1-nrt-v1', 'proswin-fold1-science-v1'])
def test_new_speed_artifact_uses_causal_residual_mode_for_cme_probabilities(mode, model_version):
    request, results, services, outlook = setup()
    request.registry['solar_event_speed']['integration']['version'] = 'cme-arrival-arc-v3'
    fallback = np.linspace(-50, 50, 101)
    blend = np.linspace(-20, 80, 101)
    services['plasma_speed_threshold'].models_bundle = dict(format='proswin_revalidated_raw',
        point_quantile='raw', thresholds=[450, 500, 600],
        residuals={'blend': [blend]*4, 'fallback': [fallback]*4})
    services['plasma_speed_threshold']._speed = SimpleNamespace(allowed_model_versions=(model_version,))
    frame = results['plasma_speed_threshold'].frame
    issue = frame.issue_time.iloc[0]
    valid = frame.valid_time.iloc[1]
    records = [dict(valid_time=valid, image_slot=valid-pd.Timedelta(hours=96),
        available_at=valid-pd.Timedelta(hours=95), value=400., model_version=model_version)] if mode == 'blend' else []
    request.inputs = SimpleNamespace(as_of=issue, proswin_ready_at=None, proswin_predictions=records)
    if mode == 'blend':
        for threshold in (450, 500, 600):
            frame.loc[1, f'p_v_ge_{threshold}'] = 1-np.searchsorted(blend, threshold-400., side='left')/len(blend)
    integrate_speed_forecast(request, results, services, outlook)
    expected_samples = blend if mode == 'blend' else fallback
    expected = 1-np.searchsorted(expected_samples, 0., side='left')/len(expected_samples)
    assert results['plasma_speed_threshold'].frame.loc[1, 'p_v_ge_500'] == pytest.approx(expected)
    np.testing.assert_array_equal(results['plasma_speed_quantile'].frame.v_q50, [400, 500, 400, 400])
