from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from clio.config import ClioObservations, load_observation_config
from clio.ingestion.extras import sdo


def policy(metric):
    return load_observation_config().observations[metric].model_dump()


@pytest.mark.parametrize('mode', ['live', 'historical'])
def test_numeric_observations_allow_one_mode(mode):
    settings = policy('v')
    settings['sources']['historical' if mode == 'live' else 'live'] = []
    if mode == 'live':
        settings['backfill_days'] = 0
    config = ClioObservations(observations={'v': settings})
    assert getattr(config.observations['v'].sources, mode)


@pytest.mark.parametrize('field', ['resolution_minutes', 'live_resolution_minutes', 'poll_minutes', 'backfill_days'])
def test_required_policy_settings_have_no_implicit_defaults(field):
    settings = policy('v')
    settings.pop(field)
    with pytest.raises(ValidationError, match=field):
        ClioObservations(observations={'v': settings})


def test_file_retention_is_never_inferred_from_backfill():
    settings = policy('gong')
    settings.pop('retention_days')
    with pytest.raises(ValidationError, match='retention_days is required'):
        ClioObservations(observations={'gong': settings})


@pytest.mark.parametrize('metric', ['gong', 'goes', 'aia_171_nrt'])
def test_file_horizon_has_no_arbitrary_legacy_limit(metric):
    settings = policy(metric)
    settings.update(retention_days=180, backfill_days=180)
    config = ClioObservations(observations={metric: settings})
    assert config.observations[metric].backfill_days == 180


def test_nrt_sampling_uses_configured_interval(storage, monkeypatch):
    settings = policy('aia_171_nrt')
    settings.update(retention_days=1, backfill_days=1, resolution_minutes=120)
    config = ClioObservations(observations={'aia_171_nrt': settings})
    collect = Mock(return_value='existing')
    monkeypatch.setattr(sdo, 'collect_one', collect)
    now = datetime(2026, 11, 2, 10, 30, tzinfo=UTC)
    report = sdo.collect(config.observations['aia_171_nrt'], channel='aia171',
        source='aia.nrt_171', mode='backfill', now=now)
    assert report['existing'] == 12
    slots = [call.args[0] for call in collect.call_args_list]
    assert all((left-right).total_seconds() == 7200 for left, right in zip(slots, slots[1:]))


def test_resolution_is_checked_against_source_capabilities():
    settings = policy('aia_171_nrt')
    settings['resolution_minutes'] = 30
    with pytest.raises(ValidationError, match='multiple of 60 minutes'):
        ClioObservations(observations={'aia_171_nrt': settings})


def test_scheduler_supports_seven_minute_intervals():
    from clio.scheduler.schedule import collecting_schedules
    from clio.scheduler.worker import _init_collect_job
    settings = policy('v')
    settings['poll_minutes'] = 7
    config = ClioObservations(observations={'v': settings})
    assert collecting_schedules(config) == [(['v'], 7)]
    scheduler = Mock(timezone=UTC)
    scheduler.get_job.return_value = None
    _init_collect_job(scheduler, ['v'], 7)
    assert scheduler.add_job.call_args.kwargs['trigger'].interval.total_seconds() == 420
