import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call
from uuid import uuid4

import pandas as pd
import pytest

from common.schemas.forecast_release import (
    FAST_BT_ARTIFACT, FAST_BT_COLUMNS, FAST_BT_PRODUCT, FAST_L1_ARTIFACT, FAST_L1_PRODUCT,
    ForecastArtifact, ForecastRelease,
)
from common import config
from prophet.services.generation import fast_l1


def test_fast_l1_publishes_a_joint_exact_time_release_at_most_once(monkeypatch):
    issue = datetime(2026, 1, 2, 1, 5, 20, tzinfo=UTC)
    valid = pd.Timestamp(issue).ceil('3h')
    trigger = issue - timedelta(minutes=5)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return issue.astimezone(tz)

    monkeypatch.setattr(fast_l1, 'datetime', Clock)
    frame = pd.DataFrame({'valid_time': [valid], 'p_kp_ge_4': [.7], 'p_kp_ge_5': [.4], 'p_kp_ge_6': [.2],
                          'ap_q10': [9], 'ap_q50': [32], 'ap_q90': [80]})
    source = fast_l1.PendingUpdate(FAST_L1_PRODUCT, issue, trigger, frame, {'as_of': issue.isoformat()},
                                   {'model': 'fast'})
    writer = object()

    @asynccontextmanager
    async def lock(_):
        yield writer

    recorder = SimpleNamespace(snapshot_payload=AsyncMock(), store=AsyncMock(),
                               finish=AsyncMock(return_value=SimpleNamespace(status='published', release_id=uuid4())))
    begin = AsyncMock(return_value=recorder)
    monkeypatch.setattr(fast_l1, 'writer_lock', lock)
    monkeypatch.setattr(fast_l1, 'published_trigger', AsyncMock(side_effect=[None, trigger]))
    monkeypatch.setattr(fast_l1.RunRecorder, 'begin', begin)

    async def check():
        runtime = object()
        await fast_l1.FastL1Update().publish(runtime, source)
        await fast_l1.FastL1Update().publish(runtime, source)
        begin.assert_awaited_once_with(FAST_L1_PRODUCT, 'scheduled', runtime, writer=writer)
        name, content, model_info, rows, columns = recorder.store.await_args.args
        assert name == FAST_L1_ARTIFACT
        artifact = ForecastArtifact(name=name, sha256=sha256(content).hexdigest(), row_count=rows,
                                    columns=columns, csv_text=content.decode(), model_info=model_info)
        release = ForecastRelease(release_id=uuid4(), run_id=uuid4(), product=FAST_L1_PRODUCT,
                                  published_at=issue, issue_time=issue, artifacts=[artifact])
        assert release.issue_time == issue
        assert 'lead_hours' not in columns
        recorder.finish.assert_awaited_once_with()

    asyncio.run(check())


def test_fast_l1_release_rejects_wrong_clock_or_unordered_distribution():
    issue = datetime(2026, 1, 2, 1, 5, tzinfo=UTC)
    columns = ['issue_time', 'valid_time', 'p_kp_ge_4', 'p_kp_ge_5', 'p_kp_ge_6',
               'ap_q10', 'ap_q50', 'ap_q90']
    for end, p5 in ((datetime(2026, 1, 2, 4, tzinfo=UTC), '.4'),
                    (datetime(2026, 1, 2, 3, tzinfo=UTC), '.9')):
        content = f"{','.join(columns)}\n{issue.isoformat()},{end.isoformat()},.7,{p5},.2,9,32,80\n"
        with pytest.raises(ValueError, match='Invalid fast L1'):
            ForecastArtifact(name=FAST_L1_ARTIFACT, sha256=sha256(content.encode()).hexdigest(), row_count=1,
                             columns=columns, csv_text=content, model_info={})


def test_fast_l1_retries_the_original_issue_when_the_writer_is_busy(monkeypatch):
    runtime = SimpleNamespace(models_registry={'models': {'geomagnetic_fast_l1': {}}})
    monkeypatch.setattr(config, 'get_config', lambda: runtime)
    runner = fast_l1.FastL1Update()
    update = object()
    runner.calculate = Mock(return_value=[update])
    runner.publish = AsyncMock(side_effect=[fast_l1.GenerationBusy(), None])

    async def check():
        await runner.poll()
        assert runner.pending == [update]
        await runner.poll()
        assert runner.pending == []
        runner.calculate.assert_called_once()
        assert runner.publish.await_args_list == [call(runtime, update), call(runtime, update)]

    asyncio.run(check())


def test_bt_fast_l1_release_accepts_multiple_hourly_targets():
    issue = datetime(2026, 1, 2, 1, 5, tzinfo=UTC)
    columns = ['issue_time', 'valid_time', *FAST_BT_COLUMNS]
    first, last = (issue.replace(minute=0) + timedelta(hours=hours) for hours in (1, 5))
    rows = [f'{issue.isoformat()},{time.isoformat()},.9,.6,.35' for time in (first, last)]
    content = '\n'.join([','.join(columns), *rows, ''])
    artifact = ForecastArtifact(name=FAST_BT_ARTIFACT, sha256=sha256(content.encode()).hexdigest(),
                                row_count=2, columns=columns, csv_text=content, model_info={})
    release = ForecastRelease(release_id=uuid4(), run_id=uuid4(), product=FAST_BT_PRODUCT,
                              published_at=issue, issue_time=issue, artifacts=[artifact])
    assert release.artifacts[0].row_count == 2
    late = content.replace(last.isoformat(), (last + timedelta(hours=1)).isoformat())
    with pytest.raises(ValueError, match='Invalid fast L1 target interval'):
        ForecastArtifact(name=FAST_BT_ARTIFACT, sha256=sha256(late.encode()).hexdigest(),
                         row_count=2, columns=columns, csv_text=late, model_info={})


def test_bt_fast_calculates_from_shared_wind_even_if_geomagnetic_history_is_unavailable(monkeypatch, tmp_path):
    issue = datetime(2026, 1, 2, 1, 5, tzinfo=UTC)
    trigger = SimpleNamespace(trigger_time=issue - timedelta(minutes=5), estimated_arrival=issue + timedelta(minutes=20))
    frame = pd.DataFrame({'valid_time': [pd.Timestamp(issue).ceil('h')], 'p_bt_ge_5': [.9],
                          'p_bt_ge_10': [.6], 'p_bt_ge_15': [.35]})
    wind = Mock(return_value=pd.DataFrame({'bt': [12]}))
    index_history = Mock(side_effect=RuntimeError('indices unavailable'))
    bt = SimpleNamespace(forecast_frame=Mock(return_value=frame))
    monkeypatch.setenv('OBSERVATIONS_URL', 'https://example.test')
    monkeypatch.setenv('OBSERVATIONS_SERVICE_TOKEN', 'test-token')
    monkeypatch.setattr(fast_l1, 'read_l1_wind_history', wind)
    monkeypatch.setattr(fast_l1, 'read_l1_geomagnetic_indices', index_history)
    detect = Mock(return_value=trigger)
    monkeypatch.setattr(fast_l1, 'find_trigger', detect)
    monkeypatch.setattr(fast_l1, 'load_model', lambda model, **_: (
        bt if model is fast_l1.FastL1BtForecaster else object(), {'model': model.registry_name},
    ))
    runtime = SimpleNamespace(workdir=tmp_path, models_registry={
        'models': {'geomagnetic_fast_l1': {}, 'hmf_fast_l1_bt': {}},
    })
    updates = fast_l1.FastL1Update().calculate(runtime, issue)
    assert [item.product for item in updates] == [FAST_BT_PRODUCT]
    assert updates[0].frame is frame
    wind.assert_called_once()
    detect.assert_called_once()
    index_history.assert_called_once()
