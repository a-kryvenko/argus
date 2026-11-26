from hashlib import sha256
from types import SimpleNamespace

import lightgbm
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from prophet.services.generation.fast_l1_model import (
    FEATURES, WIND, FastL1GeomagneticForecaster, _known_wind, find_trigger,
)


ISSUE = pd.Timestamp('2026-01-19T19:13:00Z')
FRONT = pd.Timestamp('2026-01-19T18:50:00Z')
TARGET = pd.Timestamp('2026-01-19T21:00:00Z')


@pytest.fixture
def artifact(monkeypatch):
    model = SimpleNamespace(num_feature=lambda: len(FEATURES) + 1)

    def predict(features, *, raw_score, num_threads):
        assert raw_score and num_threads == 1
        model.features = pd.DataFrame(features, columns=[*FEATURES, 'threshold_step'])
        return 4 - features[:, -1]

    model.predict = predict
    monkeypatch.setattr(lightgbm, 'Booster', lambda **_: model)
    text = 'fast ordinal model'
    return {'kind': 'tree_ordinal', 'features': list(FEATURES), 'calibration': [1, 0], 'model_text': text,
            'model_sha256': sha256(text.encode()).hexdigest()}, model


@pytest.fixture
def wind():
    clock = pd.date_range(ISSUE - pd.Timedelta(hours=4), ISSUE, freq='min')
    frame = pd.DataFrame(index=clock, columns=list(WIND), dtype=float)
    frame.loc[:, list(WIND)] = [400., 5., 3., 2., -1., 5.]
    frame.loc[FRONT:, list(WIND)] = [720., 12., 4., 3., -10., 18.]
    frame['spacecraft'] = 'DSCOVR'
    frame['received_at'] = frame.index + pd.Timedelta(minutes=1)
    return frame


def test_fast_forecast_freezes_trigger_features_and_uses_closed_index_history(artifact, wind):
    bundle, booster = artifact
    kp = pd.Series([2., 5.33, 9.], index=pd.date_range('2026-01-19T12:00:00Z', periods=3, freq='3h'))
    dst = pd.Series([-20., -50., -999.], index=pd.date_range('2026-01-19T16:00:00Z', periods=3, freq='h'))
    wind.loc[ISSUE - pd.Timedelta(minutes=2):, 'v'] = 1900.
    forecast = FastL1GeomagneticForecaster(bundle).forecast_frames(wind, kp, dst, issue_time=ISSUE)

    assert forecast.trigger_time == FRONT + pd.Timedelta(minutes=2)
    assert forecast.kp.valid_time.tolist() == forecast.ap.valid_time.tolist() == [TARGET]
    pd.testing.assert_frame_equal(forecast.kp, pd.DataFrame({
        'valid_time': [TARGET], 'p_kp_ge_4': [expit(0)], 'p_kp_ge_5': [expit(-1)], 'p_kp_ge_6': [expit(-2)],
    }))
    assert forecast.ap.loc[0, ['ap_q10', 'ap_q50', 'ap_q90']].tolist() == [6, 22, 80]
    row = booster.features.iloc[0]
    assert row.x_speed == 720.
    assert row.x_kp_latest == pytest.approx(5.33)
    assert row.x_dst_latest == -50.


def test_trigger_requires_received_samples_and_consistent_spacecraft(wind):
    assert find_trigger(wind, issue_time=ISSUE) is not None
    delayed = wind.copy()
    delayed.loc[FRONT:, 'received_at'] = ISSUE + pd.Timedelta(minutes=1)
    assert find_trigger(delayed, issue_time=ISSUE) is None
    changed = wind.copy()
    changed.loc[FRONT:, 'spacecraft'] = 'ACE'
    assert find_trigger(changed, issue_time=ISSUE) is None
    assert find_trigger(wind, issue_time=ISSUE + pd.Timedelta(minutes=30)) is None


def test_training_travel_gate_uses_independent_received_plasma_speed(wind):
    minute = pd.Timedelta(minutes=1)
    sample = ISSUE - 8 * minute
    wind['plasma_speed'] = wind.v
    wind['plasma_received_at'] = wind.received_at
    wind.loc[sample - 4 * minute:sample - minute, list(WIND)] = np.nan
    wind.loc[sample - 4 * minute:sample - 3 * minute, 'plasma_speed'] = np.nan

    assert _known_wind(wind, ISSUE).at[sample, 'travel_ready']
    wind.at[sample - minute, 'plasma_received_at'] = ISSUE + minute
    assert not _known_wind(wind, ISSUE).at[sample, 'travel_ready']


def test_rejects_modified_artifact(artifact):
    bundle, _ = artifact
    with pytest.raises(ValueError, match='hash mismatch'):
        FastL1GeomagneticForecaster({**bundle, 'model_text': 'changed'})
    with pytest.raises(ValueError, match='Invalid fast L1'):
        FastL1GeomagneticForecaster({**bundle, 'features': list(reversed(FEATURES))})
