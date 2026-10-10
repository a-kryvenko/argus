"""Numerical reference, production equivalence and temporal leakage checks."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import DTM2020, causal_interpolate, dtm_snapshot, jb_evaluate, jb_geometry
from prepare import OUT
from run import paired_metrics


def test_dtm_published_benchmark():
    if not (OUT / "libdtm2020.so").exists():
        pytest.skip("Run prepare.py to obtain licensed upstream research model")
    model = DTM2020(OUT)
    results = [model.evaluate(180, alt, 0, 0, 3.1415, flux, flux, 3, 3)
               for alt in (300, 800) for flux in (80, 180)]
    np.testing.assert_allclose(np.array(results)[:, 0],
                              np.array([.94719e-14, .29053e-13, .26600e-17, .39104e-16])*1000, rtol=5e-5)
    np.testing.assert_allclose(np.array(results)[:, 1], [843.243, 1312.266, 844.099, 1334.874], atol=.002)


def test_vectorized_jb_matches_unmodified_production_wrapper():
    from forecast_core.models.jb2008 import jb2008_density, JB2008Drivers
    frame = pd.DataFrame({"valid_time": pd.to_datetime(["2025-01-02T12:30Z", "2025-06-02T03:00Z", "2025-11-12T23:00Z"]),
                          "latitude_deg": [-70., 0., 85.], "longitude_deg": [-170., 40., 179.],
                          "altitude_km": [300., 475., 800.]})
    d = dict(f10_7=150., f10_7_81mean=140., s10=140., s10_81mean=130.,
             m10=150., m10_81mean=145., y10=160., y10_81mean=150., dtc=100.)
    production = JB2008Drivers(*d.values())
    geometry = jb_geometry(frame)
    for index, row in frame.iterrows():
        reference = jb2008_density(row.valid_time.to_pydatetime(), row.latitude_deg, row.longitude_deg, row.altitude_km, production)
        assert jb_evaluate(geometry[index], d) == pytest.approx(reference, rel=1e-10)


def test_dtm_excludes_unavailable_archive_values():
    issue = pd.Timestamp("2025-06-01T12:00Z")
    daily = pd.date_range(issue.floor("D")-pd.Timedelta(days=90), periods=90)+pd.Timedelta(hours=12)
    ap_times = pd.date_range(issue-pd.Timedelta(hours=36), issue, freq="3h")
    data = pd.concat([
        pd.DataFrame({"metric": "f10_7", "value": 100., "observed_at": daily, "available_at": daily+pd.Timedelta(hours=12)}),
        pd.DataFrame({"metric": "ap", "value": 4., "observed_at": ap_times, "available_at": ap_times+pd.Timedelta(hours=3)}),
    ], ignore_index=True)
    reference = dtm_snapshot(data, issue)
    data.loc[data.available_at > issue, "value"] = 400.
    extra = pd.DataFrame([dict(metric="f10_7", value=400., observed_at=issue, available_at=issue+pd.Timedelta(hours=12))])
    assert dtm_snapshot(pd.concat([data, extra], ignore_index=True), issue) == reference
    assert reference["kp"] == pytest.approx(1.)
    assert reference["f107_mean"] == 100.


def test_interpolation_does_not_extend_stale_inputs():
    times = pd.date_range("2025-01-01", periods=2, freq="3h", tz="UTC")
    with pytest.raises(ValueError, match="history"):
        causal_interpolate(pd.Series([4., 400.], index=times), [times[-1]+pd.Timedelta(hours=7)], pd.Timedelta(hours=6))


def test_metric_denominator_is_common_truth():
    frame = pd.DataFrame({"truth": [1., 2.], "jb2008": [1., 4.], "dtm2020": [.5, 2.],
                          "valid_time": pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)})
    jb, dtm = paired_metrics(frame)
    assert jb["mape_pct"] == 50
    assert dtm["mape_pct"] == 25
    assert dtm["underestimate_gt_30pct"] == .5
