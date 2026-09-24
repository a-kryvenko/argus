"""Scheduling and freshness contracts shared by status and collector probes."""
from clio.dataloaders.solar_wind_loader import SOURCES as WIND_SOURCES
from clio.dataloaders.geomagnetic_loader import SOURCES as INDEX_SOURCES, INTERVAL_SECONDS, POLL_SECONDS

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
    return 2 * SOURCE_SPECS[source_id]['poll_seconds'] + 60


def collector_sources(collector: str) -> list[str]:
    return [source_id for source_id, spec in SOURCE_SPECS.items() if spec['collector'] == collector]
