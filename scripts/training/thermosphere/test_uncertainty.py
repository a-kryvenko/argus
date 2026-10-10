from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from density_uncertainty import fit_density_calibration, apply_density_calibration, score_bounds, score_intervals
from ap_forecast import forecast_dtm_snapshot
from test_ap_forecast import observations, forecasts


def calibration():
    return pd.DataFrame({"valid_time": pd.date_range("2024-12-03", periods=200, freq="h", tz="UTC"),
        "lead_hours": 24, "rho_q50": 2e-12, "truth": np.linspace(1e-12, 4e-12, 200)})


def test_calibration_rejects_test_overlap():
    with pytest.raises(ValueError, match="precede"):
        fit_density_calibration(calibration(), "2024-12-05T00:00Z")


def test_log_calibration_is_multiplicative_and_ordered():
    cal = calibration()
    parameters = fit_density_calibration(cal, "2025-01-01T00:00Z")
    scaled = cal.copy();scaled[["truth", "rho_q50"]] *= 10
    scaled_parameters = fit_density_calibration(scaled, "2025-01-01T00:00Z")
    np.testing.assert_allclose(parameters.select_dtypes("number"), scaled_parameters.select_dtypes("number"))
    predictions = pd.DataFrame({"lead_hours": [24], "rho_q50": [1e-12]})
    result = apply_density_calibration(predictions, parameters)
    assert result.density_lower05.iloc[0] < result.density_upper90.iloc[0] < result.density_upper95.iloc[0]
    # Applying calibration needs no test truth.
    assert "truth" not in result


def test_unseen_horizon_is_rejected():
    params = fit_density_calibration(calibration(), "2025-01-01T00:00Z")
    with pytest.raises(ValueError, match="horizon"):
        apply_density_calibration(pd.DataFrame({"lead_hours": [48], "rho_q50": [1.]}), params)


def test_bounds_score_coverage_not_only_mean_error():
    frame = pd.DataFrame({"lead_hours": [24, 24], "activity": ["storm", "storm"],
        "truth": [1., 2.], "rho_q50": [1., 1.], "rho_q90": [1.5, 1.5], "rho_q95": [2., 2.],
        "density_lower05": [.5, .5], "density_upper90": [1.5, 1.5], "density_upper95": [2., 2.]})
    scores = score_bounds(frame).query('grouping == "lead_hours"').set_index("scenario")
    assert scores.loc["rho_q50", "underestimate_gt_30_pct"] == 50
    assert scores.loc["rho_q95", "coverage_pct"] == 100
    assert scores.loc["rho_q95", "overestimate_gt_50_pct"] == 50
    assert score_intervals(frame).coverage90_pct.eq(100).all()


def test_q95_ap_scenario_supported_without_fabricated_lower_quantiles():
    frame = forecasts().drop(columns=["ap_q10", "ap_q50"]).rename(columns={"ap_q90": "ap_q95"})
    issue = frame.issue_time.iloc[0]
    result = forecast_dtm_snapshot(observations(), frame, issue, issue+pd.Timedelta(hours=24), quantile="q95")
    assert result["kp"] == 6.
