import asyncio
from unittest.mock import AsyncMock, MagicMock

from fastapi import HTTPException
import pytest

from api.routers import healthcheck


@pytest.mark.parametrize('available', [True, False])
def test_readiness_checks_storage_and_returns_503_on_failure(monkeypatch, available):
    db = MagicMock()
    db.execute = AsyncMock(side_effect=None if available else RuntimeError('database password must not leak'))
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=db)
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(healthcheck, 'get_session_factory', lambda: lambda: context)
    if available:
        assert asyncio.run(healthcheck.ready()) == {'service': 'api', 'status': 'ok'}
    else:
        with pytest.raises(HTTPException) as result:
            asyncio.run(healthcheck.ready())
        assert result.value.status_code == 503
        assert 'password' not in result.value.detail
    assert 'dashboard_user' in str(db.execute.call_args.args[0])
