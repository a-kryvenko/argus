"""Live product adapters, with one shared fetch for products using one feed."""
import asyncio
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import select

from clio.providers.swpc_loader import SWPC_Loader
from clio.observations.schema import wide_to_measurements
from clio.db.models import GeomagneticObservation
from clio.db.session import get_session_factory
from clio.domains.geomagnetic import ingest_source
from clio.ingestion.adapters import clean_records


class WideLiveAdapter:
    def __init__(self, loader, metrics):
        self.loader, self.metrics = loader, metrics

    async def fetch(self, start, now):
        wide = await asyncio.to_thread(self.loader)
        received = datetime.now(UTC)
        frame = clean_records(wide_to_measurements(wide), start, pd.Timestamp(now) + pd.Timedelta(microseconds=1), self.metrics)
        return frame.assign(received_at=received)


class GeomagneticLiveAdapter:
    def __init__(self, metric):
        self.metric = metric

    async def fetch(self, start, now):
        records = await ingest_source(self.metric)
        if records is None:
            # Another collector owns the source lock. Use its persisted data;
            # the orchestrator will independently check observation freshness.
            async with get_session_factory()() as session:
                stored = (await session.scalars(select(GeomagneticObservation).where(
                    GeomagneticObservation.metric == self.metric,
                    GeomagneticObservation.interval_start >= start,
                    GeomagneticObservation.interval_start <= now,
                ))).all()
            records = [dict(interval_start=r.interval_start, value=r.value, quality=r.quality,
                            received_at=r.received_at, raw=r.raw) for r in stored]
        rows = []
        for record in records:
            if record['quality'] == 'flagged':
                continue
            values = {self.metric: record['value']}
            if self.metric == 'kp':
                values['ap'] = record['raw'].get('a_running')
            for metric, value in values.items():
                if isinstance(value, bool):
                    continue
                rows.append(dict(metric=metric, value=value, observed_at=record['interval_start'],
                                 received_at=record['received_at']))
        frame = pd.DataFrame(rows, columns=['metric', 'value', 'observed_at', 'received_at'])
        return clean_records(frame, start, pd.Timestamp(now) + pd.Timedelta(microseconds=1),
                             ('kp', 'ap') if self.metric == 'kp' else ('dst',))


class SolarLiveAdapter:
    async def fetch(self, start, now):
        from clio.observations.derived import load_solar_index_measurements
        # Two short live requests must not hold up the minute-index schedule.
        frame = await asyncio.to_thread(load_solar_index_measurements, now, timeout=25)
        return frame.assign(received_at=datetime.now(UTC))


def live_adapters():
    wind = WideLiveAdapter(SWPC_Loader._fetch_live_sensors, ('bx', 'by', 'bz', 'v', 'n', 't'))
    return {
        'swpc.propagated_magnetic': wind,
        'swpc.propagated_plasma': wind,
        'swpc.kp': GeomagneticLiveAdapter('kp'),
        'swpc.dst': GeomagneticLiveAdapter('dst'),
        'swpc.f107': WideLiveAdapter(SWPC_Loader._fetch_f10_7_flux, ('f10_7',)),
        'goes.calibrated_live': SolarLiveAdapter(),
    }
