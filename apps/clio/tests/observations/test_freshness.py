"""Configured freshness applies to reads, diagnostics and ingestion."""
import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest

from clio.config import load_observation_config
from clio.db.models.measurement import Measurement
from clio.ingestion.collect_numeric import fetch_live
from clio.ingestion.backfill_numeric import missing_slots

NOW = datetime(2026, 10, 3, 12, tzinfo=UTC)


@pytest.mark.parametrize('metric,age', [('bz', 15), ('dst', 90)])
@pytest.mark.parametrize('freshness,expected', [(10, 'delayed'), (120, 'fresh')])
def test_configured_threshold_controls_ingestion_and_reads(monkeypatch, metric, age, freshness, expected):
    config = load_observation_config().model_copy(deep=True)
    config.observations = {metric: config.observations[metric]}
    config.observations[metric].freshness_minutes = freshness
    interval = timedelta(hours=1) if metric == 'dst' else timedelta()
    observed = NOW - timedelta(minutes=age) - interval
    source = config.observations[metric].sources.live[0]
    frame = pd.DataFrame([dict(metric=metric, value=-2, observed_at=observed, received_at=NOW)])
    _, report = asyncio.run(fetch_live([metric], config.observations,
                                      {source: AsyncMock(return_value=frame)}, NOW))
    assert report['failed_metrics'] == ([metric] if expected == 'delayed' else [])


def test_mean_backfill_uses_configured_freshness():
    start, end = NOW - timedelta(days=14), NOW.replace(hour=0)
    frame = pd.DataFrame([dict(metric='s10_81mean', value=110, observed_at=end-timedelta(days=6))])
    assert not missing_slots(frame, 's10_81mean', start, end, freshness_minutes=7 * 1440, resolution_minutes=load_observation_config().observations['s10_81mean'].resolution_minutes)
    assert missing_slots(frame, 's10_81mean', start, end, freshness_minutes=5 * 1440, resolution_minutes=load_observation_config().observations['s10_81mean'].resolution_minutes)
