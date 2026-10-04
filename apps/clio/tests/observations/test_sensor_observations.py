from datetime import UTC, datetime, timedelta
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest


from clio.db.models import Measurement, NormalizedObservation
from clio.observations.schema import (
    OBSERVATION_METRICS,
    REQUIRED_METRICS,
    SOLAR_INDEX_METRICS,
)
from clio.observations import normalized as service
from clio.observations import derived as processing
from clio.observations.derived import normalize_measurements
from clio.observations.store import upsert_normalized_observations


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
        "source_product",
        "received_at",
        "interval_end", "quality", "spacecraft", "provider_quality", "station_count",
    }
    assert set(NormalizedObservation.__table__.columns.keys()) == {
        "observed_at",
        *OBSERVATION_METRICS,
    }


def test_legacy_calibrated_indices_are_not_published_as_observations():
    start = datetime(2026, 9, 5, tzinfo=UTC)
    rows = [*_measurement_rows(start), *_measurement_rows(start + timedelta(hours=2))]
    frame = pd.DataFrame(rows)
    frame = frame.loc[
        frame["metric"].isin(REQUIRED_METRICS) | (frame["observed_at"] == start)
    ]
    result = normalize_measurements(frame)
    assert len(result) == 3
    assert result.loc[0, list(SOLAR_INDEX_METRICS)].isna().all()
    assert result.loc[1:, list(SOLAR_INDEX_METRICS)].isna().all().all()
    assert result.loc[:, list(REQUIRED_METRICS)].notna().all().all()


def test_solar_only_measurements_do_not_create_core_observations():
    frame = pd.DataFrame([{
        "observed_at": datetime(2026, 9, 5, tzinfo=UTC), "metric": "s10", "value": 100,
    }])
    assert normalize_measurements(frame).empty


def test_refresh_reads_stored_values_without_fetching(monkeypatch):
    now = datetime(2026, 9, 5, 2, 45, tzinfo=UTC)
    stored = pd.DataFrame(_measurement_rows(now.replace(minute=0)))
    load = AsyncMock(return_value=stored)
    monkeypatch.setattr(service, "load_measurements", load)
    session = SimpleNamespace(execute=AsyncMock(), commit=AsyncMock())

    async def save(session, frame):
        records = [NormalizedObservation(**row) for row in frame.to_dict("records")]
        session.execute.return_value = Mock()
        session.execute.return_value.scalars.return_value.all.return_value = records

    monkeypatch.setattr(service, "upsert_normalized_observations", save)
    from clio.providers.swpc_loader import SWPC_Loader
    fetch = Mock(side_effect=AssertionError("refresh must not fetch"))
    monkeypatch.setattr(SWPC_Loader, 'load_measurements', fetch)
    result = asyncio.run(service.refresh_normalized_observations(session, now))
    load.assert_awaited_once_with(session, now - timedelta(days=60), until=now + timedelta(microseconds=1))
    assert len(result.points) == 1
    assert result.points[0].s10 is None
    fetch.assert_not_called()
    session.commit.assert_awaited_once()


def test_empty_refresh_fails_without_bootstrap_or_raw_writes(monkeypatch):
    monkeypatch.setattr(service, 'load_measurements', AsyncMock(return_value=pd.DataFrame(
        columns=['metric', 'value', 'observed_at'])))
    session = AsyncMock()
    with pytest.raises(RuntimeError, match='collect and backfill'):
        asyncio.run(service.refresh_normalized_observations(session))
    session.execute.assert_not_called()
    session.commit.assert_not_called()


def test_normalized_upsert_writes_sql_null_for_missing_indices():
    frame = pd.DataFrame(_measurement_rows(datetime(2026, 9, 5, tzinfo=UTC)))
    frame = frame[frame["metric"].isin(REQUIRED_METRICS)]
    normalized = normalize_measurements(frame)
    session = SimpleNamespace(execute=AsyncMock())
    asyncio.run(upsert_normalized_observations(session, normalized))
    statement = session.execute.call_args.args[0]
    parameters = statement.compile().params
    assert all(parameters[f"{name}_m0"] is None for name in SOLAR_INDEX_METRICS)
