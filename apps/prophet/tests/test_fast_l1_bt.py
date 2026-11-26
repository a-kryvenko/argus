from hashlib import sha256
from types import SimpleNamespace

import lightgbm
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from prophet.services.generation.fast_l1_bt import FEATURES, FastL1BtForecaster
from prophet.services.generation.fast_l1_model import MINUTE, WIND, find_trigger
from prophet.services.inputs.l1_geomagnetic import STANDALONE_WIND


ISSUE = pd.Timestamp('2026-01-19T19:13:20Z')
FRONT = pd.Timestamp('2026-01-19T18:50:00Z')


@pytest.fixture
def bt_artifact(monkeypatch):
    model = SimpleNamespace(num_feature=lambda: len(FEATURES) + 1)

    def predict(features, *, raw_score, num_threads):
        assert raw_score and num_threads == 1
        model.features = pd.DataFrame(features, columns=[*FEATURES, 'threshold_step'])
        return 4 - features[:, -1]

    model.predict = predict
    monkeypatch.setattr(lightgbm, 'Booster', lambda **_: model)
    text = 'fast Bt model'
    return {'kind': 'tree', 'features': list(FEATURES), 'calibration': [1, 0], 'model_text': text,
            'model_sha256': sha256(text.encode()).hexdigest()}, model


@pytest.fixture
def wind():
    issue = ISSUE.floor('min')
    clock = pd.date_range(issue - pd.Timedelta(hours=4), issue, freq='min')
    frame = pd.DataFrame(index=clock, columns=list(WIND), dtype=float)
    frame.loc[:, list(WIND)] = [400., 5., 3., 2., -1., 5.]
    frame.loc[FRONT:, list(WIND)] = [720., 12., 4., 3., -10., 18.]
    frame['spacecraft'] = 'dscovr'
    frame['received_at'] = frame.index + MINUTE
    for metric, (value, receipt) in STANDALONE_WIND.items():
        frame[value] = frame[metric]
        frame[receipt] = frame.received_at
        frame[f'{value}_spacecraft'] = frame.spacecraft
    return frame


def test_fast_bt_matches_target_clock_and_freezes_independent_wind_at_trigger(bt_artifact, wind):
    bundle, booster = bt_artifact
    trigger = find_trigger(wind, issue_time=ISSUE)
    assert trigger.trigger_time == FRONT + 2 * MINUTE
    wind.loc[trigger.trigger_time + MINUTE:, ['mag_bt', 'mag_bz', 'plasma_speed']] = 300.
    wind.loc[FRONT - pd.Timedelta(minutes=7), list(WIND)] = np.nan
    wind.loc[FRONT - pd.Timedelta(minutes=7), 'mag_bt'] = 14.
    wind.loc[FRONT - MINUTE, 'mag_bt_received_at'] = ISSUE + MINUTE

    forecast = FastL1BtForecaster(bundle).forecast_frame(wind, trigger=trigger, issue_time=ISSUE)

    targets = pd.date_range('2026-01-19T20:00:00Z', periods=5, freq='h')
    assert forecast.valid_time.tolist() == targets.tolist()
    assert forecast.loc[0, ['p_bt_ge_5', 'p_bt_ge_10', 'p_bt_ge_15']].tolist() == pytest.approx(expit([3, 2, 1]))
    row = booster.features.iloc[0]
    assert row.x_bt_mean_5m == pytest.approx((5 + 3 * 18) / 4)
    assert row.x_bt_mean_30m == pytest.approx((26 * 5 + 3 * 18 + 14 - 5) / 29)
    assert row.x_bz_mean_5m == pytest.approx((-2 - 30) / 5)
    assert row.x_speed_5m == 720.
    assert row.x_bt_jump == 13.
    assert row.x_bt_baseline == 5.
    assert row.x_lead_hours == pytest.approx((targets[0] - ISSUE) / pd.Timedelta(hours=1))


def test_fast_bt_respects_magnetic_coverage_and_rejects_stale_trigger(bt_artifact, wind):
    bundle, _ = bt_artifact
    trigger = find_trigger(wind, issue_time=ISSUE)
    model = FastL1BtForecaster(bundle)
    assert model.forecast_frame(wind, trigger=trigger, issue_time=trigger.trigger_time + 32 * MINUTE) is None
    wind.loc[FRONT:FRONT + MINUTE, 'mag_bt_spacecraft'] = 'ace'
    assert model.forecast_frame(wind, trigger=trigger, issue_time=ISSUE) is None


def test_fast_bt_rejects_modified_or_incompatible_artifact(bt_artifact):
    bundle, _ = bt_artifact
    with pytest.raises(ValueError, match='hash mismatch'):
        FastL1BtForecaster({**bundle, 'model_text': 'changed'})
    with pytest.raises(ValueError, match='Invalid fast L1 Bt'):
        FastL1BtForecaster({**bundle, 'features': list(reversed(FEATURES))})
