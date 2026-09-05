import numpy as np
import pandas as pd
import pytest

from forecast.data_pipelines.solar_indices import (
    INDEX_FEATURE_COLUMNS,
    SolarIndexCalibration,
    build_daily_goes_features,
    extract_solar_indices,
    extract_solar_index_observations,
    fit_solar_index_calibrations,
    load_solar_index_calibrations,
    save_solar_index_calibrations,
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


def _identity_calibrations():
    return {
        name: SolarIndexCalibration(
            feature_columns=features,
            feature_means=(0.0,) * len(features),
            feature_scales=(1.0,) * len(features),
            intercept=0.0,
            coefficients=(1.0,) + (0.0,) * (len(features) - 1),
        ) for name, features in INDEX_FEATURE_COLUMNS.items()
    }


def test_calibration_artifact_roundtrip_and_validation(tmp_path):
    import json

    path = tmp_path / "calibration.json"
    calibrations = _identity_calibrations()
    save_solar_index_calibrations(calibrations, path)
    loaded = load_solar_index_calibrations(path)
    assert loaded == calibrations
    pd.testing.assert_frame_equal(
        extract_solar_indices(_goes_frame(), loaded),
        extract_solar_indices(_goes_frame(), calibrations),
    )
    payload = json.loads(path.read_text())
    payload["calibrations"]["s10"]["feature_scales"][0] = 0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="positive scales"):
        load_solar_index_calibrations(path)


def test_hourly_snapshots_use_available_daily_data_without_lookahead():
    records = pd.concat([_goes_frame(1).iloc[[0]]] * 5, ignore_index=True)
    records["timestamp"] = pd.to_datetime([
        "2026-01-01T00:30Z", "2026-01-01T01:30Z", "2026-01-01T02:30Z",
        "2026-01-01T03:30Z", "2026-01-01T04:30Z",
    ])
    records["goes_euv_256"] = [10, 20, 999, 40, 9999]
    records.loc[2, "goes_euvs_quality_valid"] = False
    result = extract_solar_index_observations(
        records, _identity_calibrations(), pd.Timestamp("2026-01-01T04:00Z"),
    )
    assert list(result["observed_at"].dt.hour) == [1, 2, 4]
    # Irradiance correction is 2, followed by a median over the day so far.
    assert list(result["s10"]) == [20, 30, 40]


def test_midnight_closes_previous_day_and_next_hour_starts_new_day():
    records = pd.concat([_goes_frame(1).iloc[[0]]] * 2, ignore_index=True)
    records["timestamp"] = pd.to_datetime(["2026-01-01T23:30Z", "2026-01-02T00:30Z"])
    records["goes_euv_256"] = [10, 100]
    result = extract_solar_index_observations(
        records, _identity_calibrations(), pd.Timestamp("2026-01-02T01:00Z"),
    )
    assert list(result["s10"]) == [20, 200]


def test_no_valid_or_future_only_goes_yields_no_observations():
    records = _goes_frame(1)
    records["goes_euvs_quality_valid"] = False
    assert extract_solar_index_observations(
        records, _identity_calibrations(), pd.Timestamp("2026-01-01T02:00Z"),
    ).empty
    records["goes_euvs_quality_valid"] = True
    assert extract_solar_index_observations(
        records, _identity_calibrations(), pd.Timestamp("2025-12-31T23:00Z"),
    ).empty
