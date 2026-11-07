from datetime import UTC, datetime, timedelta
import asyncio
import json
from unittest.mock import Mock, AsyncMock
import pandas as pd
import pytest
from clio.ingestion.extras import goes


NOW = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)


def test_goes_stores_source_samples_directly_in_s3(storage, monkeypatch):
    frame = pd.DataFrame([dict(timestamp=pd.Timestamp(NOW-timedelta(minutes=10)), goes_euv_256=2.5)])
    monkeypatch.setattr(goes, 'source_frames', Mock(return_value=[frame]))
    row = goes.download_snapshots(NOW-timedelta(hours=1), NOW, False, resolution_minutes=1440 if False else 60)[0]
    content = next(iter(storage.objects.values()))[0]
    assert json.loads(content)[0]['goes_euv_256'] == 2.5
    assert not {'raw_path', 'features'}.intersection(row)
    assert row['file_id']


def test_goes_retains_registered_days_without_downloading(storage, monkeypatch):
    slot = NOW.replace(hour=0, minute=0)
    saved = dict(slot_at=slot, observed_at=slot, source_product='goes.archive', file_id='existing')
    monkeypatch.setattr(goes, 'source_frames', Mock(side_effect=AssertionError('No source downloads')))
    assert goes.download_snapshots(slot, slot+timedelta(days=1), True, {(slot, 'goes.archive'): saved}, resolution_minutes=1440 if True else 60) == [saved]
    assert not storage.objects


@pytest.mark.parametrize('freshness,expected', [(5, 'partial'), (30, 'complete')])
def test_live_goes_respects_freshness(storage, monkeypatch, freshness, expected):
    from clio.config import load_observation_config
    config = load_observation_config().model_copy(deep=True)
    config.observations['goes'].freshness_minutes = freshness
    frame = pd.DataFrame([dict(timestamp=pd.Timestamp(NOW-timedelta(minutes=15)), goes_euv_256=2.5)])
    monkeypatch.setattr(goes, 'source_frames', Mock(return_value=[frame]))
    assert asyncio.run(goes.collect(config, mode='live', now=NOW))['status'] == expected
