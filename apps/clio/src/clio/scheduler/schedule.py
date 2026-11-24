"""Group configured observations by polling interval."""


def collecting_schedules(config):
    groups = {}
    for metric, policy in config.observations.items():
        if policy.sources.live:
            groups.setdefault(policy.poll_minutes, []).append(metric)
    return [(metrics, minutes) for minutes, metrics in groups.items()]


def poll_seconds(metric: str) -> int:
    from clio.config import load_observation_config
    config = load_observation_config()
    if metric == 'donki' and config.solar_wind_events.enabled:
        return config.solar_wind_events.poll_minutes * 60
    for metrics, minutes in collecting_schedules(config):
        if metric in metrics:
            return minutes * 60
    raise ValueError(f'No collection schedule for {metric}')
