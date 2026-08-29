import numpy as np
import pandas as pd
import pytest

from forecast.data_pipelines.solar_indices import (
    INDEX_FEATURE_COLUMNS,
    SolarIndexCalibration,
    build_daily_goes_features,
    extract_solar_indices,
    fit_solar_index_calibrations,
)


def _goes_frame(days: int = 8) -> pd.DataFrame:
    lyman_alpha = (5, 7, 6, 11, 8, 13, 10, 17)
    mgii = (0.250, 0.270, 0.255, 0.280, 0.260, 0.290, 0.265, 0.300)
    xray = (1, 3, 2, 6, 4, 9, 5, 8)
    rows = []
    for day in range(days):
        for minute, valid in ((0, True), (1, False)):
            value = float(day + 1)
            rows.append({
                "timestamp": pd.Timestamp("2026-01-01", tz="UTC")
                + pd.Timedelta(days=day, minutes=minute),
                "goes_au_factor": 2.0,
                "goes_euv_256": value,
                "goes_euv_284": value**2,
                "goes_euv_304": float(day % 3 + 1),
                "goes_euv_1175": value * 4,
                "goes_lya_1216": float(lyman_alpha[day]),
                "goes_euv_1335": value * 6,
                "goes_euv_1405": value * 7,
                "goes_mgii_index": mgii[day],
                "goes_xray_background": 1e-7 * xray[day],
                "goes_euvs_quality_valid": valid,
            })
    return pd.DataFrame(rows)


def test_build_daily_goes_features_uses_valid_records_and_adjusts_to_1au() -> None:
    daily = build_daily_goes_features(_goes_frame())

    assert len(daily) == 8
    assert daily.loc[0, "timestamp"] == pd.Timestamp("2026-01-01 12:00:00Z")
    assert daily.loc[0, "goes_euv_256_1au"] == 2.0
    assert daily.loc[0, "goes_lya_1216_1au"] == 10.0
    assert daily.loc[0, "goes_sample_count"] == 2
    assert daily.loc[0, "goes_valid_sample_count"] == 1
    assert daily.loc[0, "goes_valid_fraction"] == 0.5


def test_fit_and_extract_solar_indices_recovers_linear_targets() -> None:
    goes = _goes_frame()
    daily = build_daily_goes_features(goes)
    truth = daily[["timestamp"]].copy()
    truth["s10"] = (
        50
        + 2 * daily["goes_euv_256_1au"]
        + 3 * daily["goes_euv_284_1au"]
        - daily["goes_euv_304_1au"]
    )
    truth["m10"] = 20 + 400 * daily["goes_mgii_index"]
    truth["y10"] = (
        30
        + daily["goes_lya_1216_1au"]
        + 1e7 * daily["goes_xray_background"]
        + 10 * daily["goes_mgii_index"]
    )

    calibrations = fit_solar_index_calibrations(goes, truth)
    result = extract_solar_indices(goes, calibrations)

    np.testing.assert_allclose(result["s10"], truth["s10"], atol=1e-9)
    np.testing.assert_allclose(result["m10"], truth["m10"], atol=1e-9)
    np.testing.assert_allclose(result["y10"], truth["y10"], atol=1e-9)


def test_extract_solar_indices_requires_every_calibration() -> None:
    feature = INDEX_FEATURE_COLUMNS["s10"][0]
    calibration = SolarIndexCalibration(
        feature_columns=(feature,),
        feature_means=(0.0,),
        feature_scales=(1.0,),
        intercept=0.0,
        coefficients=(1.0,),
    )

    with pytest.raises(ValueError, match="Missing solar-index calibrations"):
        extract_solar_indices(_goes_frame(), {"s10": calibration})
