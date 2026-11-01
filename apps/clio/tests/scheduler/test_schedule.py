"""Scheduler coverage and polling cadence must match configured observations."""
from clio.config import load_observation_config
from clio.scheduler.worker import COLLECTING_SCHEDULES


def test_every_configured_observation_is_scheduled_once():
    metrics = [metric for group, _ in COLLECTING_SCHEDULES for metric in group]
    assert len(metrics) == len(set(metrics))
    assert set(load_observation_config().observations) <= set(metrics)
    assert {'donki', 'enlil', 'sdo', 'sdo-browse'} <= set(metrics)


def test_monitored_source_intervals_match_scheduler():
    from clio.ingestion.numeric_live import SOURCE_IDS
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
                assert minute == ('*' if seconds == 60 else f'*/{int(seconds // 60)}')
                checked.add(source)
    assert checked == set(SOURCE_IDS.values())
