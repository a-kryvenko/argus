from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ap_forecast import convert_ap_forecasts, forecast_dtm_snapshot


def forecasts():
    issue = pd.Timestamp("2025-06-01T12:00Z")
    return pd.DataFrame({"issue_time": issue,
        "valid_time": pd.date_range(issue+pd.Timedelta(hours=1), periods=48, freq="h"),
        "ap_q10": 27., "ap_q50": 48., "ap_q90": 80.})


def observations():
    issue = forecasts().issue_time.iloc[0]
    days = pd.date_range(issue.floor("D")-pd.Timedelta(days=90), periods=90)+pd.Timedelta(hours=12)
    hours = pd.date_range(issue-pd.Timedelta(hours=36), issue, freq="3h")
    return pd.concat([
        pd.DataFrame({"metric":"f10_7", "value":100., "observed_at":days,"available_at":days+pd.Timedelta(hours=12)}),
        pd.DataFrame({"metric":"ap", "value":4., "observed_at":hours,"available_at":hours+pd.Timedelta(hours=3)})])


def test_table_and_explicit_clipping():
    frame = forecasts()
    converted = convert_ap_forecasts(frame)
    np.testing.assert_allclose(converted[["kp_q10","kp_q50","kp_q90"]], np.tile([4.,5.,6.],(48,1)))
    frame.loc[0,["ap_q10","ap_q50","ap_q90"]] = [-1.,37.5,401.]
    converted = convert_ap_forecasts(frame)
    assert converted.kp_q50.iloc[0] == pytest.approx(4+1/3+5.5/7/3)
    assert converted.kp_q10.iloc[0] == 0
    assert converted.kp_q90.iloc[0] == 9
    assert converted.ap_q10_clipped.iloc[0]
    assert frame.ap_q10.iloc[0] == -1 # raw input retained


def test_crossed_quantiles_rejected():
    frame=forecasts();frame.loc[0,"ap_q10"]=100.
    with pytest.raises(ValueError,match="ordered"):
        convert_ap_forecasts(frame)


def test_future_observations_and_other_issuances_cannot_change_drivers():
    frame=forecasts();issue=frame.issue_time.iloc[0];valid=issue+pd.Timedelta(hours=24)
    obs=observations()
    reference=forecast_dtm_snapshot(obs,frame,issue,valid)
    obs.loc[obs.available_at>issue,"value"]=400.
    later=frame.copy();later.issue_time+=pd.Timedelta(minutes=30);later[["ap_q10","ap_q50","ap_q90"]]=400.
    assert forecast_dtm_snapshot(obs,pd.concat([frame,later]),issue,valid)==reference
    assert reference["kp"]==5.
    assert reference["kp_mean"]==5.
    assert reference["f107"]==100.


def test_missing_horizon_is_not_persisted():
    frame=forecasts();issue=frame.issue_time.iloc[0]
    with pytest.raises(ValueError,match="horizon"):
        forecast_dtm_snapshot(observations(),frame,issue,issue+pd.Timedelta(hours=49))


def test_missing_hourly_trajectory_is_rejected():
    frame=forecasts();issue=frame.issue_time.iloc[0]
    with pytest.raises(ValueError,match="Gap"):
        forecast_dtm_snapshot(observations(),frame.drop(index=[5,6,7]),issue,issue+pd.Timedelta(hours=24))
