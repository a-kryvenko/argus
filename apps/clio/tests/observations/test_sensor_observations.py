from datetime import UTC, datetime, timedelta
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest


from clio.db.models.measurement import Measurement
from clio.db.models.normalized_observation import NormalizedObservation
from clio.config import ClioObservations
from clio.db.models.measurement import (
    OBSERVATION_METRICS,
    REQUIRED_METRICS,
    SOLAR_INDEX_METRICS,
)
from clio.ingestion import normalize as service
from clio.ingestion import normalize as processing
from clio.ingestion.normalize import normalize_measurements
from clio.ingestion.normalize import upsert_normalized_observations


@pytest.fixture
def storage(monkeypatch):
    session = SimpleNamespace(execute=AsyncMock(), commit=AsyncMock())
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(service, "get_db_session", lambda: context)
    return session, context


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

    assert list(result.columns) == ["observed_at", *OBSERVATION_METRICS, "wind_time_basis"]
    assert list(result["observed_at"]) == [start, start + timedelta(hours=2)]
    assert result.bx.tolist() == [0, 2]


def test_normalize_measurements_requires_every_metric() -> None:
    timestamp = datetime(2026, 8, 30, tzinfo=UTC)
    measurements = pd.DataFrame(_measurement_rows(timestamp))
    measurements = measurements[measurements["metric"] != "dst"]

    result = normalize_measurements(measurements)

    assert result.empty


def test_model_preparation_uses_closed_mean_and_carries_only_previous_indices():
    start = datetime(2026, 8, 30, tzinfo=UTC)
    later = start + timedelta(hours=2)
    rows = _measurement_rows(start)
    rows += [
        {**row, 'value': 7.}
        for row in _measurement_rows(start + timedelta(hours=1))
        if row['metric'] not in ('kp', 'ap', 'f10_7')
    ]
    rows += [{'metric': 'v', 'observed_at': start + timedelta(minutes=30), 'value': 404.}]
    rows += _measurement_rows(later, offset=50.)
    result = normalize_measurements(pd.DataFrame(rows), now=later + timedelta(minutes=5))
    assert result.observed_at.tolist() == [start, start + timedelta(hours=1)]
    assert result.v.tolist() == [203.5, 7.]
    assert result.kp.tolist() == [6., 6.]


def test_daily_flux_prefers_configured_source_without_looking_ahead():
    start = datetime(2026, 8, 30, tzinfo=UTC)
    rows = [
        {**row, 'source_product': 'swpc.rtsw_mag'}
        for hour in (1, 3)
        for row in _measurement_rows(start + timedelta(hours=hour))
        if row['metric'] != 'f10_7'
    ]
    for source, hour, value in [('backup', 0, 120.), ('primary', 2, 130.),
                                ('backup', 3, 140.), ('disabled', 3, 150.)]:
        rows.append({'metric': 'f10_7', 'source_product': source, 'value': value,
                     'observed_at': start + timedelta(hours=hour)})
    result = normalize_measurements(pd.DataFrame(rows), priorities={'f10_7': ['primary', 'backup']})
    assert result.f10_7.tolist() == [120., 130.]


def test_database_models_match_narrow_and_wide_storage_contracts() -> None:
    assert set(Measurement.__table__.columns.keys()) == {
        "id",
        "metric",
        "value",
        "observed_at",
        "source_product",
        "received_at",
        "quality",
        "spacecraft",
    }
    assert set(NormalizedObservation.__table__.columns.keys()) == {
        "observed_at",
        *OBSERVATION_METRICS,
        "wind_time_basis",
    }


def test_legacy_calibrated_indices_are_not_published_as_observations():
    start = datetime(2026, 9, 5, tzinfo=UTC)
    rows = [*_measurement_rows(start), *_measurement_rows(start + timedelta(hours=2))]
    frame = pd.DataFrame(rows)
    frame = frame.loc[
        frame["metric"].isin(REQUIRED_METRICS) | (frame["observed_at"] == start)
    ]
    result = normalize_measurements(frame)
    assert len(result) == 2
    assert result.loc[0, list(SOLAR_INDEX_METRICS)].isna().all()
    assert result.loc[1:, list(SOLAR_INDEX_METRICS)].isna().all().all()
    assert result.loc[:, list(REQUIRED_METRICS)].notna().all().all()


def test_solar_only_measurements_do_not_create_core_observations():
    frame = pd.DataFrame([{
        "observed_at": datetime(2026, 9, 5, tzinfo=UTC), "metric": "s10", "value": 100,
    }])
    assert normalize_measurements(frame).empty


def test_refresh_reads_stored_values_without_fetching(monkeypatch, storage):
    now = datetime(2026, 9, 5, 2, 45, tzinfo=UTC)
    stored = pd.DataFrame(_measurement_rows(now.replace(minute=0) - timedelta(hours=1)))
    load = AsyncMock(return_value=stored)
    monkeypatch.setattr(service, "load_measurements", load)
    session, context = storage
    config = ClioObservations(observations={})

    async def save(session, frame):
        records = [NormalizedObservation(**row) for row in frame.to_dict("records")]
        session.execute.return_value = Mock()
        session.execute.return_value.scalars.return_value.all.return_value = records

    monkeypatch.setattr(service, "upsert_normalized_observations", save)
    fetch = Mock(side_effect=AssertionError("refresh must not fetch"))
    monkeypatch.setattr('requests.sessions.Session.request', fetch)
    result = asyncio.run(service.refresh_normalized_observations(config, now))
    load.assert_awaited_once_with(session, now - timedelta(days=62), until=now + timedelta(microseconds=1),
        source_products=['omni.hourly', 'soho.plasma_hourly', 'ace.plasma_hourly', 'swpc.rtsw_plasma',
                         'swpc.rtsw_mag', 'swpc.kp', 'swpc.dst', 'swpc.f107', 'gfz.f107',
                         'celestrak.f107', 'celestrak.f107_archive'], metrics=REQUIRED_METRICS)
    assert len(result.points) == 1
    assert result.points[0].s10 is None
    fetch.assert_not_called()
    session.commit.assert_awaited_once()
    context.__aenter__.assert_awaited_once()
    context.__aexit__.assert_awaited_once()


def test_empty_refresh_fails_without_bootstrap_or_raw_writes(monkeypatch, storage):
    monkeypatch.setattr(service, 'load_measurements', AsyncMock(return_value=pd.DataFrame(
        columns=['metric', 'value', 'observed_at'])))
    session, context = storage
    config = ClioObservations(observations={})
    with pytest.raises(RuntimeError, match='collect first'):
        asyncio.run(service.refresh_normalized_observations(config))
    session.execute.assert_not_called()
    session.commit.assert_not_called()
    context.__aexit__.assert_awaited_once()


def test_normalized_upsert_writes_sql_null_for_missing_indices():
    frame = pd.DataFrame(_measurement_rows(datetime(2026, 9, 5, tzinfo=UTC)))
    frame = frame[frame["metric"].isin(REQUIRED_METRICS)]
    normalized = normalize_measurements(frame)
    session = SimpleNamespace(execute=AsyncMock())
    asyncio.run(upsert_normalized_observations(session, normalized))
    statement = session.execute.call_args.args[0]
    parameters = statement.compile().params
    assert all(parameters[f"{name}_m0"] is None for name in SOLAR_INDEX_METRICS)


def test_model_preparation_uses_configured_retention(monkeypatch, storage):
    now = datetime(2026, 10, 27, tzinfo=UTC)
    config = ClioObservations(observations={}, retention={'normalized_days': 8})
    load = AsyncMock(return_value=pd.DataFrame(columns=['metric', 'value', 'observed_at']))
    monkeypatch.setattr(service, 'load_measurements', load)
    with pytest.raises(RuntimeError, match='collect first'):
        asyncio.run(service.refresh_normalized_observations(config, now))
    assert load.call_args.args[1] == now - timedelta(days=10)
    assert 'omni.hourly' in load.call_args.kwargs['source_products']


def test_normalized_reader_converts_indices_to_contract_integers():
    from clio.ingestion.normalize import load_normalized_observations
    expected = dict(kp=1.33, dst=-10.75, ap=5.666666666666667, f10_7=98.39285714285714)
    row = SimpleNamespace(observed_at=datetime(2026, 11, 3, tzinfo=UTC),
        bx=1., by=2., bz=-3., v=420., n=5., t=100000., **expected)
    session = AsyncMock()
    session.execute.return_value = Mock(scalars=lambda: Mock(all=lambda: [row]))
    result = asyncio.run(load_normalized_observations(session, since=row.observed_at, until=row.observed_at))
    assert {metric: getattr(result.points[0], metric) for metric in expected} == {metric: int(value) for metric, value in expected.items()}
