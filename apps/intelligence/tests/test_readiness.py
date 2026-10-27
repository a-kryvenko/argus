import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
import sys

from fastapi import HTTPException
import pytest

from intelligence import main
from intelligence.db import session as db


@pytest.mark.parametrize('available', [True, False])
def test_readiness_checks_owned_storage(monkeypatch, available):
    for key in ('INTELLIGENCE_SERVICE_TOKEN', 'FORECASTS_URL', 'FORECASTS_SERVICE_TOKEN'):
        monkeypatch.setenv(key, 'configured')
    names = ('assess_density', 'describe_mode', 'summarize_monitor', 'prioritize_monitors', 'prepare_operator_chart',
             'evaluate_drag_events', 'assess_density_exposure', 'summarize_density_advisory', 'evaluate_advisory_events', 'reuse_density_advisory')
    monkeypatch.setitem(sys.modules, 'intelligence_core.api', SimpleNamespace(**dict.fromkeys(names, lambda: None)))
    connection = AsyncMock()
    connection.__aenter__.return_value = connection
    if not available:
        connection.execute.side_effect = RuntimeError('database password must not leak')
    monkeypatch.setattr(main, 'get_db_session', lambda: connection)
    if available:
        assert asyncio.run(main.ready())['status'] == 'ok'
        assert any('intelligence.drag_profile' in str(call.args[0]) for call in connection.execute.call_args_list)
    else:
        with pytest.raises(HTTPException) as result:
            asyncio.run(main.ready())
        assert result.value.status_code == 503
        assert 'password' not in result.value.detail
