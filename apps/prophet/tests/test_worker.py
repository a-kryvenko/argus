from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from argus_prophet.worker import due_slot, generation_lock, run_due


def test_hourly_boundary_and_restart(tmp_path):
    generate = Mock()
    before = datetime(2026, 9, 12, 10, 9, tzinfo=UTC)
    after = before.replace(minute=10)
    assert due_slot(before) == '2026-09-12T09:00:00+00:00'
    assert run_due(generate, tmp_path, before)
    assert not run_due(generate, tmp_path, before)
    assert run_due(generate, tmp_path, after)
    assert not run_due(generate, tmp_path, after)
    assert generate.call_count == 2


def test_failed_generation_retries_without_advancing_marker(tmp_path):
    now = datetime(2026, 9, 12, 10, 10, tzinfo=UTC)
    generate = Mock(side_effect=[RuntimeError('unavailable'), None])
    with pytest.raises(RuntimeError):
        run_due(generate, tmp_path, now)
    assert not (tmp_path / 'last-completed-slot').exists()
    assert run_due(generate, tmp_path, now)


def test_manual_and_scheduled_generations_cannot_overlap(tmp_path):
    generate = Mock()
    with generation_lock(tmp_path):
        with pytest.raises(RuntimeError, match='Another Prophet'):
            run_due(generate, tmp_path, datetime.now(UTC))
    generate.assert_not_called()
    assert run_due(generate, tmp_path, datetime.now(UTC))


def test_worker_retries_exports_even_when_generation_slot_completed(tmp_path, monkeypatch):
    from argus_prophet import worker
    # Three polls of one completed slot: an export failure must not recalculate.
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
    monkeypatch.setattr(worker, 'state_directory', lambda: tmp_path)
    monkeypatch.setattr(worker, 'due_slot', lambda _: 'fixed-slot')
    generate = Mock()
    export = Mock(side_effect=[OSError('disk full'), None, None])
    worker.work(generate, export=export)
    generate.assert_called_once()
    assert export.call_count == 3
