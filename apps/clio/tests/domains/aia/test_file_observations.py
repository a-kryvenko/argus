import asyncio
from datetime import UTC, datetime, timedelta
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from clio.config import load_observation_config, ClioObservations
from clio.domains.aia import collection as service
from clio.providers.aia import snapshot_paths

NOW = datetime(2026, 9, 28, 12, 20, tzinfo=UTC)
SLOT = NOW.replace(minute=0) - timedelta(hours=1)


def record(slot, root):
    raw, cache, _ = snapshot_paths(slot, root)
    return dict(slot_at=slot.isoformat(), observed_at=slot.isoformat(), available_at=NOW.isoformat(),
                sha256='a' * 64, raw_path=str(raw), cache_path=str(cache),
                b0_deg=1., valid_fraction=1., carrington_lon=20.)


@pytest.fixture
def database(monkeypatch):
    session = AsyncMock()
    existing = []
    session.execute.return_value = Mock(scalars=lambda: Mock(all=lambda: existing))
    context = AsyncMock()
    context.__aenter__.return_value = session
    monkeypatch.setattr(service, 'get_session_factory', lambda: lambda: context)
    return session, existing


def test_live_checks_recent_slots_while_backfill_uses_configured_closed_range():
    slots = service.observation_slots('live', NOW, 40)
    assert slots == [NOW.replace(minute=0) - timedelta(hours=h) for h in range(4)]
    slots = service.observation_slots('backfill', NOW, 40)
    assert len(slots) == 40 * 24
    assert slots[-1] == SLOT
    assert service.observation_slots('backfill', NOW, 40, start=SLOT, end=SLOT + timedelta(hours=1)) == [SLOT]


def test_file_config_rejects_numeric_sources_and_unbounded_history():
    raw = {'sources': {'live': ['aia.nrt_193'], 'historical': ['aia.synoptic_193']},
           'schedules': {'live': {'every': '1h'}, 'backfill': {'every': '6h'}}, 'backfill': {'days': 61}}
    with pytest.raises(ValidationError, match='60 days'):
        ClioObservations.model_validate({'observations': {'aia193': raw}})
    raw['backfill']['days'] = 40
    raw['sources']['live'] = ['omni.hourly']
    with pytest.raises(ValidationError, match='incompatible'):
        ClioObservations.model_validate({'observations': {'aia193': raw}})


def test_new_snapshot_insert_is_conflict_safe_and_records_product(database, tmp_path):
    session, _ = database
    adapter = Mock(fetch=lambda slot, root, expected: record(slot, root))
    result = asyncio.run(service.collect_slots([SLOT], ['aia.synoptic_193'], root=tmp_path,
                                               adapters={'aia.synoptic_193': adapter}))
    assert result['received'] == 1 and not result['failed']
    compiled = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert 'ON CONFLICT (slot_at) DO NOTHING' in str(compiled)
    assert compiled.params['source_product_m0'] == 'aia.synoptic_193'
    assert compiled.params['available_at_m0'] == NOW
    assert not compiled.params['raw_path_m0'].startswith('/')


def test_known_snapshot_restoration_never_updates_database_record(database, tmp_path):
    session, existing = database
    raw = record(SLOT, tmp_path)
    expected = service.snapshot_values(raw, tmp_path)
    existing.append(SimpleNamespace(**expected))
    fetch = Mock(return_value=raw)
    result = asyncio.run(service.collect_slots([SLOT], ['aia.synoptic_193'], root=tmp_path,
                                               adapters={'aia.synoptic_193': SimpleNamespace(fetch=fetch)}))
    assert result['restored'] == 1
    assert fetch.call_args.args[2]['available_at'] == NOW
    assert fetch.call_args.args[2]['sha256'] == raw['sha256']
    session.execute.assert_awaited_once()  # coverage SELECT only; immutable row
    session.commit.assert_not_called()


def test_fallback_only_fills_absent_slot_and_keeps_attempt_errors(database, tmp_path):
    failing = Mock(side_effect=OSError('offline'))
    fallback = Mock(side_effect=lambda slot, root, expected: record(slot, root))
    result = asyncio.run(service.collect_slots([SLOT], ['primary', 'backup'], root=tmp_path, adapters={
        'primary': SimpleNamespace(fetch=failing), 'backup': SimpleNamespace(fetch=fallback)}))
    assert result['received'] == 1 and result['failed'] == 0
    assert result['source_errors'][0]['error'] == 'offline'
    assert result['source_attempts'][1]['available'] == 1


def test_qc_rejection_is_terminal_and_not_replaced_by_fallback(database, tmp_path):
    def reject(slot, root, expected):
        raw, _, receipt = snapshot_paths(slot, root)
        raw.parent.mkdir(parents=True)
        raw.write_bytes(b'rejected original')
        receipt.write_text(json.dumps({'rejected': True}))

    backup = Mock()
    result = asyncio.run(service.collect_slots([SLOT], ['primary', 'backup'], root=tmp_path, adapters={
        'primary': SimpleNamespace(fetch=reject), 'backup': SimpleNamespace(fetch=backup)}))
    assert result['rejected'] == 1 and not result['missing']
    backup.assert_not_called()
    database[0].commit.assert_not_called()


@pytest.mark.parametrize('mode, failed, expected', [('live', 0, ['aia193']), ('backfill', 0, []), ('backfill', 1, ['aia193'])])
def test_file_failures_distinguish_archive_gaps_from_live_freshness(monkeypatch, mode, failed, expected):
    collect = AsyncMock(return_value=dict(received=0, restored=0, retained=0, rejected=0, missing=1,
                                         failed=failed, latest_observed_at=None))
    monkeypatch.setattr(service, 'collect_slots', collect)
    result = asyncio.run(service.collect_file_observations(load_observation_config(), ['aia193'], mode=mode, now=NOW))
    assert result['failed_metrics'] == expected
    assert result['status'] == 'partial'


@pytest.mark.parametrize('mode, product', [
    ('live', 'aia.synoptic_193'), ('historical', 'aia.nrt_193'),
])
def test_file_sources_reject_wrong_mode(mode, product):
    raw = {'sources': {'live': ['aia.nrt_193'], 'historical': ['aia.synoptic_193']},
           'schedules': {'live': {'every': '1h'}, 'backfill': {'every': '6h'}},
           'backfill': {'days': 40}}
    raw['sources'][mode] = [product]
    with pytest.raises(ValidationError, match='incompatible'):
        ClioObservations.model_validate({'observations': {'aia193': raw}})
