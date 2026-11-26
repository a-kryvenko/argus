"""Batch metrics that share a provider and polling interval."""

from clio.ingestion.utils.products import OBSERVATIONS


def collecting_schedules(config):
    groups = {}
    for metric, policy in config.observations.items():
        if not policy.sources.live:
            continue
        definition = OBSERVATIONS[metric]
        lane = definition.file_kind if definition.kind == 'file' else policy.sources.live[0]
        groups.setdefault((definition.kind, lane, policy.poll_minutes), []).append(metric)
    return [(metrics, minutes) for (_, _, minutes), metrics in groups.items()]


def poll_seconds(metric: str) -> int:
    from clio.config import load_observation_config
    config = load_observation_config()
    if metric == 'donki' and config.solar_wind_events.enabled:
        return config.solar_wind_events.poll_minutes * 60
    for metrics, minutes in collecting_schedules(config):
        if metric in metrics:
            return minutes * 60
    raise ValueError(f'No collection schedule for {metric}')
