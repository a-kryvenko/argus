from datetime import UTC, datetime, timedelta
import json

from argus_prophet.services import proswin_cache


def test_missing_corrupt_and_future_cache_leave_dlinear_available(tmp_path, monkeypatch):
    monkeypatch.setattr(proswin_cache, 'cache_root', lambda: tmp_path)
    issue = datetime(2026, 10, 8, 12, tzinfo=UTC)
    assert proswin_cache.read_predictions(issue) == []
    root = tmp_path / 'predictions'
    root.mkdir()
    for lead in [1, 2, 3]:
        valid = issue + timedelta(hours=lead)
        record = dict(valid_time=valid.isoformat(), image_slot=(valid-timedelta(hours=96)).isoformat(),
                      available_at=(issue+timedelta(minutes=lead-2)).isoformat(), value=500)
        (root / f'{valid:%Y%m%dT%H}.json').write_text(json.dumps(record) if lead != 1 else '{')
    records = proswin_cache.read_predictions(issue)
    assert len(records) == 1
    assert records[0].valid_time == issue + timedelta(hours=2)
