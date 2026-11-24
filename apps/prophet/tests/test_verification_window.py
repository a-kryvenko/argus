from contextlib import asynccontextmanager, nullcontext
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from prophet.config import load_config
from prophet.services import verification


@pytest.mark.anyio
@pytest.mark.parametrize('days', [14, 30])
async def test_verification_and_summaries_use_configured_window(monkeypatch, days):
    from common import config

    monkeypatch.setattr(config, 'get_config', lambda: SimpleNamespace(
        project_config={'prophet': {'verification_days': days}}))
    assert load_config().verification_days == days
    rows = Mock()
    rows.all.return_value = []
    rows.mappings.return_value.all.return_value = []
    session = Mock(execute=AsyncMock(return_value=rows))
    session.begin.return_value = nullcontext()
    monkeypatch.setattr(verification, 'get_db_session', lambda: nullcontext(session))
    now = datetime(2026, 10, 27, tzinfo=UTC)
    start = now - timedelta(days=days)

    result = await verification.verify('dst', writer=None, now=now)
    assert result['from'] == start.isoformat()
    assert session.execute.call_args.args[1]['start'] == start

    report = await verification.verification_report('dst')
    assert report['days'] == days

    summary = await verification.monthly_accuracy('dst', now=now)
    assert summary.start == start
    assert session.execute.call_args.args[1]['start'] == start - timedelta(hours=96)


@pytest.mark.anyio
async def test_verification_budget_includes_file_lock_and_releases_writer_locks(monkeypatch, tmp_path):
    from common import config
    from prophet.commands import verification as command
    from prophet.db import locks
    from prophet.services.heavy_task import heavy_task

    acquired = []
    released = []

    @asynccontextmanager
    async def writer_lock(key):
        acquired.append(key)
        try:
            yield object()
        finally:
            released.append(key)

    verify = AsyncMock()
    monkeypatch.setattr(verification, 'verify', verify)
    monkeypatch.setattr(locks, 'writer_lock', writer_lock)
    monkeypatch.setattr(config, 'get_config', lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr('prophet.config.load_config', lambda: SimpleNamespace(
        shutdown_grace_seconds=1, verification_timeout_seconds=0.05,
    ))
    with heavy_task(tmp_path):
        with pytest.raises(TimeoutError):
            await command.run(SimpleNamespace(product='dst'))

    assert acquired == [locks.GENERATION_LOCK, locks.VERIFICATION_LOCK]
    assert released == acquired[::-1]
    verify.assert_not_awaited()
