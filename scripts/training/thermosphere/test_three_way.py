from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_ap import FEATURES, causal_features, forecast_ap, training_rows
from run_three_way import common_rows, metrics
from ap_forecast import forecast_dtm_snapshot
from test_ap_forecast import observations, forecasts


def raw_archive():
    clock = pd.date_range("2024-11-01", "2024-12-03", freq="h", tz="UTC")
    return pd.DataFrame({"bx": 3., "by": 4., "bz": -2., "v": 450., "n": 5., "t": 1e5,
                         "kp": 1., "ap": 4., "dst": -10., "f10_7": 100.}, index=clock)


def test_in_progress_and_future_bins_do_not_change_issue_features():
    raw = raw_archive()
    issue = pd.Timestamp("2024-11-20T01:00Z")
    before = causal_features(raw).loc[issue].copy()
    raw.loc[raw.index >= issue, ["bx", "by", "bz", "v", "dst"]] = 99.
    raw.loc[raw.index >= issue.floor("3h"), ["ap", "kp"]] = 9.
    raw.loc[raw.index >= issue.floor("D"), "f10_7"] = 400.
    pd.testing.assert_series_equal(causal_features(raw).loc[issue], before)
    # This guards against accidentally using the current OMNI hour directly.
    assert before.southward_bz == 2.


def test_training_purges_full_target_bin_and_excludes_test_changes():
    raw = raw_archive()
    original = training_rows(raw, causal_features(raw))
    cutoff = pd.Timestamp("2024-12-01T00:00Z")
    assert original.label_available_at.max() < cutoff
    assert original.valid_time.max() < cutoff
    raw.loc[raw.index >= cutoff, :] = 99.
    changed = training_rows(raw, causal_features(raw))
    pd.testing.assert_frame_equal(original, changed)


def test_median_only_driver_and_zero_lead_match_persistence():
    frame = forecasts().drop(columns=["ap_q10", "ap_q90"])
    issue = frame.issue_time.iloc[0]
    from models import dtm_snapshot
    fixed = dtm_snapshot(observations(), issue)
    assert forecast_dtm_snapshot(observations(), frame, issue, issue) == fixed
    varied = forecast_dtm_snapshot(observations(), frame, issue, issue+pd.Timedelta(hours=24))
    assert varied["kp"] == 5.
    assert varied["f107"] == fixed["f107"]
    assert varied["f107_mean"] == fixed["f107_mean"]


def test_all_three_variants_use_same_rows():
    frame = pd.DataFrame({"valid_time": pd.date_range("2025-01-01", periods=3, tz="UTC"),
        "truth": [1., 2., 3.], "jb2008": [1., 2., 3.], "dtm2020": [1., 2., 3.],
        "dtm2020_ap_forecast": [1., np.nan, 6.], "lead_hours": 24, "activity": "quiet", "window": "a"})
    paired = common_rows(frame)
    scored = metrics(paired)
    assert scored.n.eq(2).all()
    assert set(scored.model) == {"jb2008", "dtm2020", "dtm2020_ap_forecast"}


def test_forecast_refuses_training_overlap():
    with pytest.raises(ValueError, match="overlap"):
        forecast_ap({"metadata": {"max_label_available_at": "2025-01-01T00:00Z"}},
                     pd.DataFrame(), pd.DatetimeIndex(["2024-12-31T00:00Z"]))


def test_open_ap_bin_is_bridged_at_half_hour_issuance():
    frame = forecasts().drop(columns=["ap_q10", "ap_q90"])
    frame["issue_time"] += pd.Timedelta(minutes=90)
    frame["valid_time"] += pd.Timedelta(minutes=90)
    issue = frame.issue_time.iloc[0]
    # At 13:30 the most recent closed bin started 09:00: age 4.5h is valid.
    result = forecast_dtm_snapshot(observations(), frame, issue, issue+pd.Timedelta(hours=24))
    assert result["kp"] == 5.
