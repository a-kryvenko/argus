from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
from argus_prophet import worker


def test_hourly_boundary_uses_utc_and_rejects_naive_time():
    before = datetime(2026, 9, 12, 10, 9, tzinfo=UTC)
    assert worker.due_slot(before) == before.replace(hour=9, minute=0)
    assert worker.due_slot(before.replace(minute=10)) == before.replace(minute=0)
    assert worker.due_slot(before.astimezone(timezone(timedelta(hours=2)))) == worker.due_slot(before)
    with pytest.raises(ValueError, match='timezone'):
        worker.due_slot(before.replace(tzinfo=None))


def fake_storage(monkeypatch, rows):
    conn = Mock()
    conn.execute.side_effect = [Mock(fetchone=Mock(return_value=row)) for row in rows]
    @contextmanager
    def connect(**_):
        yield conn
    monkeypatch.setattr(worker, 'connect', connect)
    monkeypatch.setattr(worker, 'generation_lock', nullcontext)
    return conn


def test_completed_slot_skips_generation_including_clock_rollback(monkeypatch):
    now = datetime(2026, 9, 12, 10, 10, tzinfo=UTC)
    for completed in (worker.due_slot(now), worker.due_slot(now) + timedelta(hours=1)):
        fake_storage(monkeypatch, [(completed,)])
        generate = Mock()
        assert not worker.run_due(generate, now)
        generate.assert_not_called()


def test_latest_slot_is_passed_to_generation_without_replaying_missed_hours(monkeypatch):
    now = datetime(2026, 9, 12, 10, 10, tzinfo=UTC)
    fake_storage(monkeypatch, [(now - timedelta(days=3),), ('succeeded',)])
    generate = Mock()
    assert worker.run_due(generate, now)
    generate.assert_called_once_with(worker.due_slot(now))


def test_callback_cannot_claim_success_without_atomic_slot_completion(monkeypatch):
    fake_storage(monkeypatch, [(None,), ('running',)])
    with pytest.raises(RuntimeError, match='did not complete'):
        worker.run_due(Mock(), datetime.now(UTC))






def test_worker_retries_exports_even_when_generation_slot_completed(monkeypatch):
    class StopAfterThree:
        count = 0
        def is_set(self):
            return self.count == 3
        def set(self):
            self.count = 3
        def wait(self, _):
            self.count += 1
    monkeypatch.setattr(worker.threading, 'Event', StopAfterThree)
    monkeypatch.setattr(worker.signal, 'signal', lambda *_: None)
    monkeypatch.setattr(worker, 'generation_lock', nullcontext)
    run_due = Mock(return_value=False)
    monkeypatch.setattr(worker, 'run_due', run_due)
    generate = Mock()
    export = Mock(side_effect=[OSError('disk full'), None, None])
    worker.work(generate, export=export)
    generate.assert_not_called()
    assert run_due.call_count == 3 and export.call_count == 3
