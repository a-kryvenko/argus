from datetime import UTC, datetime
import pytest
from argus_clio.scheduler import slot_for
from argus_clio.db.session import get_database_url


def test_calendar_slots_are_aligned():
    now = datetime(2026, 9, 12, 12, 19, 45, tzinfo=UTC)
    assert slot_for('aggregate', now) == now.replace(minute=15, second=0)
    assert slot_for('refresh', now) == now.replace(minute=0, second=0)
    assert slot_for('aia', now) == now.replace(minute=0, second=0)


def test_clio_uses_only_its_own_database_settings(monkeypatch):
    monkeypatch.setenv('DB_USER', 'postgres')
    monkeypatch.setenv('DB_PASSWORD', 'admin-password')
    monkeypatch.delenv('CLIO_DB_PASSWORD', raising=False)
    with pytest.raises(RuntimeError, match='CLIO_DB_PASSWORD'):
        get_database_url()
    monkeypatch.setenv('CLIO_DB_NAME', 'argus_clio')
    monkeypatch.setenv('CLIO_DB_USER', 'argus_clio')
    monkeypatch.setenv('CLIO_DB_PASSWORD', 'raw@password:/%')
    assert get_database_url().username == 'argus_clio'
    assert get_database_url().database == 'argus_clio'
    assert get_database_url().password == 'raw@password:/%'
