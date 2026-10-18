from datetime import UTC, datetime
from unittest.mock import Mock

from clio.providers import f107


def test_live_f107_preserves_observed_values_and_source_timestamps(monkeypatch):
    first = datetime(2026, 8, 30, 0, 7, tzinfo=UTC)
    second = datetime(2026, 8, 30, 0, 52, tzinfo=UTC)
    response = Mock()
    response.json.return_value = [dict(time_tag=first.isoformat(), flux=110.5),
                                 dict(time_tag=second.isoformat(), flux=120.25)]
    get = Mock(return_value=response)
    monkeypatch.setattr(f107.requests, 'get', get)
    result = f107.fetch_f107()
    assert result.issue_time.tolist() == [first, second]
    assert result.f10_7.tolist() == [110.5, 120.25]
    get.assert_called_once_with(f107.F10_7_FLUX_URL, timeout=60)
    response.raise_for_status.assert_called_once()
