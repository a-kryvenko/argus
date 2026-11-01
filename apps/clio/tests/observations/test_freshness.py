"""Configured freshness applies to reads, diagnostics and ingestion."""
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from clio.config import load_observation_config
from clio.db.models.measurement import Measurement
from clio.ingestion.numeric_live import fetch_live
from clio.ingestion.numeric_backfill import missing_slots
from clio.observations import status as measurements
from clio.observations import solar_wind, geomagnetic

NOW = datetime(2026, 10, 3, 12, tzinfo=UTC)


@pytest.mark.parametrize('metric,age', [('bz', 15), ('dst', 90)])
@pytest.mark.parametrize('freshness,expected', [(10, 'delayed'), (120, 'fresh')])
def test_configured_threshold_controls_ingestion_and_reads(monkeypatch, metric, age, freshness, expected):
    config = load_observation_config().model_copy(deep=True)
    config.observations = {metric: config.observations[metric]}
    config.observations[metric].freshness_minutes = freshness
    interval = timedelta(hours=1) if metric == 'dst' else timedelta()
    observed = NOW - timedelta(minutes=age) - interval
    record = Measurement(metric=metric, value=-2, observed_at=observed, received_at=NOW)
    session = AsyncMock()
    session.scalars.return_value = [record]
    session.execute.return_value = Mock(scalars=Mock(return_value=Mock(first=Mock(return_value=record))))
    monkeypatch.setattr(measurements, 'datetime', Mock(now=Mock(return_value=NOW)))
    monkeypatch.setattr(measurements, 'load_observation_config', lambda: config)
    item = asyncio.run(measurements.observation_status(session))['measurements'][0]
    assert item['status'] == expected
    assert item['stale_after_seconds'] == freshness * 60
    assert item['age_seconds'] == age * 60

    reader = geomagnetic if metric == 'dst' else solar_wind
    monkeypatch.setattr(reader, 'load_observation_config', lambda: config)
    if metric == 'dst':
        monkeypatch.setattr(geomagnetic, 'SOURCES', {'dst': geomagnetic.SOURCES['dst']})
        result = asyncio.run(reader.latest(session, NOW))
    else:
        result = asyncio.run(reader.latest(session, [metric], NOW))
    assert result['series'][metric]['status'] == ('stale' if expected == 'delayed' else 'fresh')

    source = config.observations[metric].sources.live[0]
    frame = pd.DataFrame([dict(metric=metric, value=-2, observed_at=observed, received_at=NOW)])
    _, report = asyncio.run(fetch_live([metric], config.observations,
                                      {source: AsyncMock(return_value=frame)}, NOW))
    assert report['failed_metrics'] == ([metric] if expected == 'delayed' else [])


def test_mean_backfill_uses_configured_freshness():
    start, end = NOW - timedelta(days=14), NOW.replace(hour=0)
    frame = pd.DataFrame([dict(metric='s10_81mean', value=110, observed_at=end-timedelta(days=6))])
    assert not missing_slots(frame, 's10_81mean', start, end, freshness_minutes=7 * 1440)
    assert missing_slots(frame, 's10_81mean', start, end, freshness_minutes=5 * 1440)
