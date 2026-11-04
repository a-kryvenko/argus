"""Group configured observations by polling interval."""


def collecting_schedules(config):
    groups = {}
    for metric, policy in config.observations.items():
        if policy.sources.live:
            groups.setdefault(policy.poll_minutes, []).append(metric)
    if config.solar_wind_events.enabled:
        groups.setdefault(config.solar_wind_events.poll_minutes, []).extend(['donki', 'enlil'])
    return [(metrics, minutes) for minutes, metrics in groups.items()]


def poll_seconds(metric: str) -> int:
    from clio.config import load_observation_config
    for metrics, minutes in collecting_schedules(load_observation_config()):
        if metric in metrics:
            return minutes * 60
    raise ValueError(f'No collection schedule for {metric}')
