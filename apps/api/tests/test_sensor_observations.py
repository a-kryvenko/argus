from datetime import UTC, datetime, timedelta
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from app.db.models import Measurement, NormalizedObservation
from app.services.sensor_observations import (
    OBSERVATION_METRICS,
    REQUIRED_METRICS,
    SOLAR_INDEX_METRICS,
    normalize_measurements,
)
from app.services import sensor_observations as service
from forecast.data_pipelines.solar_indices import (
    INDEX_FEATURE_COLUMNS,
    REQUIRED_GOES_COLUMNS,
    SolarIndexCalibration,
    save_solar_index_calibrations,
)


def _measurement_rows(timestamp: datetime, offset: float = 0) -> list[dict]:
    return [
        {
            "observed_at": timestamp,
            "metric": metric,
            "value": index + offset,
        }
        for index, metric in enumerate(OBSERVATION_METRICS)
    ]


def test_normalize_measurements_builds_hourly_wide_rows() -> None:
    start = datetime(2026, 8, 30, tzinfo=UTC)
    measurements = pd.DataFrame([
        *_measurement_rows(start),
        *_measurement_rows(start + timedelta(hours=2), offset=2),
    ])

    result = normalize_measurements(measurements)

    assert list(result.columns) == ["observed_at", *OBSERVATION_METRICS]
    assert list(result["observed_at"]) == list(pd.date_range(start, periods=3, freq="1h"))
    middle = result.iloc[1]
    assert middle["bx"] == 1
    assert middle["f10_7"] == 10


def test_normalize_measurements_requires_every_metric() -> None:
    timestamp = datetime(2026, 8, 30, tzinfo=UTC)
    measurements = pd.DataFrame(_measurement_rows(timestamp))
    measurements = measurements[measurements["metric"] != "dst"]

    result = normalize_measurements(measurements)

    assert result.empty


def test_database_models_match_narrow_and_wide_storage_contracts() -> None:
    assert set(Measurement.__table__.columns.keys()) == {
        "id",
        "metric",
        "value",
        "observed_at",
    }
    assert set(NormalizedObservation.__table__.columns.keys()) == {
        "observed_at",
        *OBSERVATION_METRICS,
    }


def test_solar_indices_are_optional_and_never_gap_filled():
    start = datetime(2026, 9, 5, tzinfo=UTC)
    rows = [*_measurement_rows(start), *_measurement_rows(start + timedelta(hours=2))]
    frame = pd.DataFrame(rows)
    frame = frame.loc[
        frame["metric"].isin(REQUIRED_METRICS) | (frame["observed_at"] == start)
    ]
    result = normalize_measurements(frame)
    assert len(result) == 3
    assert result.loc[0, list(SOLAR_INDEX_METRICS)].notna().all()
    assert result.loc[1:, list(SOLAR_INDEX_METRICS)].isna().all().all()
    assert result.loc[:, list(REQUIRED_METRICS)].notna().all().all()


def test_solar_only_measurements_do_not_create_core_observations():
    frame = pd.DataFrame([{
        "observed_at": datetime(2026, 9, 5, tzinfo=UTC), "metric": "s10", "value": 100,
    }])
    assert normalize_measurements(frame).empty


@pytest.fixture
def calibrated_goes(monkeypatch, tmp_path):
    path = tmp_path / "calibration.json"
    save_solar_index_calibrations({
        name: SolarIndexCalibration(
            feature_columns=features, feature_means=(0.0,) * len(features),
            feature_scales=(1.0,) * len(features), intercept=100.0,
            coefficients=(1.0,) + (0.0,) * (len(features) - 1),
        ) for name, features in INDEX_FEATURE_COLUMNS.items()
    }, path)
    monkeypatch.setattr(service, "get_config", lambda: SimpleNamespace(
        workdir=tmp_path, models_registry={"models": {
            "solar_index_calibration": {"calibration_path": path.name},
        }},
    ))
    rows = []
    for hour in (0, 1, 2):
        row = dict.fromkeys(REQUIRED_GOES_COLUMNS, 1.0)
        row.update(timestamp=pd.Timestamp(f"2026-09-05T0{hour}:30Z"), goes_euvs_quality_valid=True)
        rows.append(row)
    fetch = Mock(return_value=pd.DataFrame(rows))
    monkeypatch.setattr(service, "fetch_goes", fetch)
    return fetch


def test_live_goes_is_calibrated_and_only_current_snapshot_is_ingested(calibrated_goes):
    now = datetime(2026, 9, 5, 2, 45, tzinfo=UTC)
    result = service._load_solar_index_measurements(now)
    calibrated_goes.assert_called_once_with(end=now)
    assert set(result["metric"]) == set(SOLAR_INDEX_METRICS)
    assert result["value"].tolist() == [101, 101, 101]
    assert result["observed_at"].tolist() == [pd.Timestamp("2026-09-05T02:00Z")] * 3


def test_missing_calibration_does_not_fetch_or_invent_indices(monkeypatch, tmp_path, caplog):
    monkeypatch.setattr(service, "get_config", lambda: SimpleNamespace(
        workdir=tmp_path, models_registry={"models": {
            "solar_index_calibration": {"calibration_path": "missing.json"},
        }},
    ))
    fetch = Mock()
    monkeypatch.setattr(service, "fetch_goes", fetch)
    assert service._load_solar_index_measurements(datetime(2026, 9, 5, tzinfo=UTC)).empty
    fetch.assert_not_called()
    assert len(caplog.records) == 1
    warning = caplog.records[0]
    assert warning.levelname == "WARNING"
    assert warning.exc_info is None
    assert str(tmp_path / "missing.json") in warning.message
    assert "models.solar_index_calibration.calibration_path" in warning.message


def test_goes_failure_leaves_optional_indices_absent(calibrated_goes):
    from requests import RequestException

    calibrated_goes.side_effect = RequestException("GOES unavailable")
    assert service._load_solar_index_measurements(datetime(2026, 9, 5, tzinfo=UTC)).empty


def test_refresh_integrates_solar_indices_and_serializes_nullable_values(monkeypatch, calibrated_goes):
    now = datetime(2026, 9, 5, 2, 45, tzinfo=UTC)
    live = pd.DataFrame(_measurement_rows(now.replace(minute=0)))
    live = live[live["metric"].isin(REQUIRED_METRICS)]
    monkeypatch.setattr(service.SWPC_Loader, "load_measurements", lambda: live)
    monkeypatch.setattr(service, "_database_is_empty", AsyncMock(return_value=False))
    ingested = []

    async def ingest(session, frame):
        ingested.append(frame)

    async def load(session, since):
        return pd.concat(ingested, ignore_index=True)

    session = SimpleNamespace(execute=AsyncMock(), commit=AsyncMock())

    async def store(session, frame):
        records = [NormalizedObservation(**row) for row in frame.to_dict("records")]
        session.execute.return_value = Mock()
        session.execute.return_value.scalars.return_value.all.return_value = records

    monkeypatch.setattr(service, "_upsert_measurements", ingest)
    monkeypatch.setattr(service, "_load_measurements", load)
    monkeypatch.setattr(service, "_upsert_normalized_observations", store)
    result = asyncio.run(service.refresh_normalized_observations(session, now))
    assert len(result.points) == 1
    assert result.points[0].s10 == 101.0
    assert result.points[0].m10 == 101.0
    assert result.points[0].y10 == 101.0
    session.commit.assert_awaited_once()


def test_normalized_upsert_writes_sql_null_for_missing_indices():
    frame = pd.DataFrame(_measurement_rows(datetime(2026, 9, 5, tzinfo=UTC)))
    frame = frame[frame["metric"].isin(REQUIRED_METRICS)]
    normalized = normalize_measurements(frame)
    session = SimpleNamespace(execute=AsyncMock())
    asyncio.run(service._upsert_normalized_observations(session, normalized))
    statement = session.execute.call_args.args[0]
    parameters = statement.compile().params
    assert all(parameters[f"{name}_m0"] is None for name in SOLAR_INDEX_METRICS)
