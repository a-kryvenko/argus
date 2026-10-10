import numpy as np
import pandas as pd
import pytest

from run_oracle import oracle_trajectory


def test_oracle_uses_valid_time_and_preserves_forecast_clock():
    issue = pd.Timestamp('2025-01-01T00:30Z')
    forecasts = pd.DataFrame({'issue_time': [issue, issue],
        'valid_time': [issue+pd.Timedelta(hours=1), issue+pd.Timedelta(hours=2)],
        'lead_hours': [1, 2], 'ap_q50': [999., 999.]})
    observed = pd.Series([5., 12., 27.], index=pd.date_range('2025-01-01', periods=3, freq='h', tz='UTC'))
    result = oracle_trajectory(forecasts, observed)
    assert result.ap_q50.tolist() == [12., 27.]
    pd.testing.assert_frame_equal(result.drop(columns='ap_q50'), forecasts.drop(columns='ap_q50'))
    assert forecasts.ap_q50.eq(999).all()


def test_oracle_rejects_missing_truth_instead_of_filling():
    issue = pd.Timestamp('2025-01-01T00:00Z')
    forecasts = pd.DataFrame({'issue_time': [issue], 'valid_time': [issue+pd.Timedelta(hours=1)], 'lead_hours': [1]})
    observed = pd.Series([np.nan], index=pd.DatetimeIndex(forecasts.valid_time))
    with pytest.raises(ValueError, match='future observed ap'):
        oracle_trajectory(forecasts, observed)
