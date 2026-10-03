"""Scheduling and freshness contracts shared by status and collector probes."""
from clio.providers.solar_wind_loader import SOURCES as WIND_SOURCES
from clio.providers.geomagnetic_loader import SOURCES as INDEX_SOURCES, INTERVAL_SECONDS, POLL_SECONDS

WIND_POLL_SECONDS = 60
WIND_STALE_AFTER_SECONDS = 600
ATTEMPT_TIMEOUT_SECONDS = 120
SOURCE_SPECS = {
    'solar_wind_mag': {'label': 'Solar wind magnetic field', 'collector': 'solar-wind',
                       'poll_seconds': WIND_POLL_SECONDS, 'stale_after_seconds': WIND_STALE_AFTER_SECONDS, 'freshness_basis': 'observed_at',
                       'source_url': WIND_SOURCES['mag']},
    'solar_wind_plasma': {'label': 'Solar wind plasma', 'collector': 'solar-wind',
                          'poll_seconds': WIND_POLL_SECONDS, 'stale_after_seconds': WIND_STALE_AFTER_SECONDS, 'freshness_basis': 'observed_at',
                          'source_url': WIND_SOURCES['plasma']},
    **{metric: {'label': 'Estimated Kp' if metric == 'kp' else 'Real-time Dst', 'collector': 'geomagnetic',
                'poll_seconds': POLL_SECONDS[metric], 'stale_after_seconds': INTERVAL_SECONDS[metric] + 3600,
                'freshness_basis': 'interval_end', 'source_url': INDEX_SOURCES[metric]}
       for metric in INDEX_SOURCES},
}


def overdue_after(source_id: str) -> int:
    return 2 * poll_seconds(source_id) + 60


def configured_geomagnetic_polls() -> dict[str, float]:
    from clio.config import load_observation_config
    config = load_observation_config()
    polls = {}
    for source in INDEX_SOURCES:
        intervals = [policy.schedules.live.every.total_seconds() for policy in config.observations.values()
                     if f'swpc.{source}' in policy.sources.live]
        if intervals:
            polls[source] = min(intervals)
    return polls


def configured_wind_polls() -> dict[str, float]:
    from clio.config import load_observation_config
    config = load_observation_config()
    polls = {}
    for kind in ('mag', 'plasma'):
        intervals = [policy.schedules.live.every.total_seconds() for policy in config.observations.values()
                     if f'swpc.rtsw_{kind}' in policy.sources.live]
        if intervals:
            polls[f'solar_wind_{kind}'] = min(intervals)
    return polls


def poll_seconds(source_id: str) -> float:
    if source_id in INDEX_SOURCES:
        return configured_geomagnetic_polls().get(source_id, POLL_SECONDS[source_id])
    return configured_wind_polls().get(source_id, SOURCE_SPECS[source_id]['poll_seconds'])


def collector_sources(collector: str) -> list[str]:
    if collector == 'geomagnetic':
        return list(configured_geomagnetic_polls())
    return list(configured_wind_polls()) if collector == 'solar-wind' else []
