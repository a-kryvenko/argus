from contextlib import nullcontext
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
