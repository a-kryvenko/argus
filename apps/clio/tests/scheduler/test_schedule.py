"""Scheduler coverage and polling cadence must match configured observations."""
from clio.config import load_observation_config
from clio.scheduler.schedule import collecting_schedules

COLLECTING_SCHEDULES = collecting_schedules(load_observation_config())


def test_every_configured_observation_is_scheduled_once():
    metrics = [metric for group, _ in COLLECTING_SCHEDULES for metric in group]
    assert len(metrics) == len(set(metrics))
    assert set(load_observation_config().observations) <= set(metrics)
    assert {'donki', 'enlil', 'aia_171_nrt', 'aia_094_browse'} <= set(metrics)


def test_monitored_source_intervals_match_scheduler():
    from clio.ingestion.collect_numeric import SOURCE_IDS
    from clio.scheduler.schedule import poll_seconds

    config = load_observation_config()
    checked = set()
    for metrics, minute in COLLECTING_SCHEDULES:
        for metric in metrics:
            policy = config.observations.get(metric)
            for product in policy.sources.live if policy else []:
                source = SOURCE_IDS.get(product)
                if source is None:
                    continue
                seconds = poll_seconds(metric)
                assert minute * 60 == seconds
                checked.add(source)
    assert checked == set(SOURCE_IDS.values())


def test_sdo_polling_follows_product_policy():
    config = load_observation_config().model_copy(deep=True)
    config.observations['aia_171_nrt'].poll_minutes = 10
    schedules = collecting_schedules(config)
    assert [minute for metrics, minute in schedules if 'aia_171_nrt' in metrics] == [10]
    config.observations.pop('aia_171_nrt')
    assert all('aia_171_nrt' not in metrics for metrics, minute in collecting_schedules(config))
