from contextlib import nullcontext
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest

from argus_prophet.config import VerificationSchedule
from argus_prophet.scheduling import verification


def test_disabled_schedule_never_acquires_lock(monkeypatch):
    lock = Mock(side_effect=AssertionError('Disabled job acquired lock'))
    monkeypatch.setattr(verification, 'verification_lock', lock)
    assert not verification.run_due(VerificationSchedule())


def test_verification_checkpoint_only_follows_complete_scoring(monkeypatch):
    now = datetime(2026, 9, 29, 13, tzinfo=UTC)
    conn = Mock()
    conn.execute.return_value.fetchone.return_value = None
    monkeypatch.setattr(verification, 'connect', lambda **_: nullcontext(conn))
    monkeypatch.setattr(verification, 'verification_lock', nullcontext)
    config = VerificationSchedule(enabled=True)
    score = Mock(side_effect=ValueError('Clio unavailable'))
    with pytest.raises(ValueError):
        verification.run_due(config, now=now, verify=score)
    assert conn.execute.call_count == 1
    conn.reset_mock()
    score.side_effect = None
    assert verification.run_due(config, now=now, verify=score)
    assert score.call_args.args == ('all',)
    assert score.call_args.kwargs['days'] == 7 and score.call_args.kwargs['now'] == now
    assert score.call_args.kwargs['deadline'] > 0
    assert conn.execute.call_args.args[1][0] == now.replace(hour=12)
    conn.execute.return_value.fetchone.return_value = (now.replace(hour=12),)
    score.reset_mock()
    assert not verification.run_due(config, now=now, verify=score)
    score.assert_not_called()
