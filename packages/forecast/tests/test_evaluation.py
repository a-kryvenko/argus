import numpy as np
import pandas as pd
import pytest

from forecast.evaluation import match_observations, score, validate_predictions


def predictions():
    return pd.DataFrame({
        'issue_time': ['2026-01-01T00:00:00Z'] * 3,
        'valid_time': ['2026-01-01T01:00:00Z', '2026-01-01T02:00:00Z', '2026-01-01T03:00:00Z'],
        'lead_hours': [1, 2, 3], 'v_q10': [300., 300., 300.],
        'v_q50': [400., 400., 400.], 'v_q90': [500., 500., 500.],
    })


def test_future_and_missing_targets_are_not_scores_or_zeroes():
    forecast = validate_predictions(predictions(), 'plasma_speed_quantile')
    truth = pd.DataFrame({'valid_time': ['2026-01-01T01:00:00Z', '2026-01-01T03:00:00Z'],
                          'value': [450., 999.]})
    matched = match_observations(forecast, truth, as_of=pd.Timestamp('2026-01-01T03:30:00Z'))
    assert matched.state.tolist() == ['verified', 'missing', 'pending']
    assert np.isnan(matched.iloc[2].value)
    result = score(matched, 'plasma_speed_quantile')
    assert result['counts'] == {'total': 3, 'verified': 1, 'missing': 1, 'pending': 1}
    assert result['tables']['regression.csv'][0]['mae'] == 50
    assert result['tables']['regression.csv'][0]['bias'] == -50


@pytest.mark.parametrize('change', ['duplicate', 'lead', 'naive', 'crossing', 'infinity'])
def test_invalid_prediction_contract_is_rejected(change):
    frame = predictions()
    if change == 'duplicate':
        frame = pd.concat([frame, frame.iloc[[0]]])
    elif change == 'lead':
        frame.loc[0, 'lead_hours'] = 4
    elif change == 'naive':
        frame.loc[0, 'issue_time'] = '2026-01-01T00:00:00'
    elif change == 'crossing':
        frame.loc[0, 'v_q10'] = 600
    else:
        frame.loc[0, 'v_q50'] = np.inf
    with pytest.raises(ValueError):
        validate_predictions(frame, 'plasma_speed_quantile')


def test_duplicate_truth_is_rejected_instead_of_multiplying_scores():
    forecast = validate_predictions(predictions(), 'plasma_speed_quantile')
    truth = pd.DataFrame({'valid_time': ['2026-01-01T01:00:00Z'] * 2, 'value': [450., 460.]})
    with pytest.raises(ValueError, match='unique'):
        match_observations(forecast, truth, as_of=pd.Timestamp('2026-01-02T00:00:00Z'))


def test_binary_equality_and_single_class_metrics():
    pytest.importorskip('sklearn')
    frame = pd.DataFrame({'lead_hours': [1, 1], 'value': [500., 500.], 'state': ['verified'] * 2,
                          'p_v_ge_450': [1., 1.], 'p_v_ge_500': [.5, .5], 'p_v_ge_600': [0., 0.]})
    result = score(frame, 'plasma_speed_threshold')['tables']
    assert result['threshold_500.csv'][0]['brier'] == .25
    assert result['threshold_500.csv'][0]['threat_score'] == 1
    assert result['threshold_600.csv'][0]['roc_auc'] is None
    assert result['threshold_600.csv'][0]['avg_precision'] is None
