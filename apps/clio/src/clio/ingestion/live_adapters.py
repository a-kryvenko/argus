"""Live product adapters, with one shared fetch for products using one feed."""
import asyncio
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import select

from clio.providers.swpc_loader import SWPC_Loader
from clio.observations.schema import wide_to_measurements
from clio.db.models import Measurement
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
        await ingest_source(self.metric)
        return await stored_native_frame(('kp', 'ap') if self.metric == 'kp' else ('dst',), start, now, f'swpc.{self.metric}')


async def stored_native_frame(metrics, start, now, product):
    async with get_session_factory()() as session:
        statement = select(Measurement.metric, Measurement.value, Measurement.observed_at, Measurement.received_at).where(
            Measurement.metric.in_(metrics), Measurement.source_product == product, Measurement.observed_at >= start, Measurement.observed_at <= now,
            Measurement.quality.not_in(['flagged', 'missing']), Measurement.value.is_not(None))
        rows = (await session.execute(statement)).all()
    return pd.DataFrame(rows, columns=['metric', 'value', 'observed_at', 'received_at'])


class SolarWindLiveAdapter:
    def __init__(self, kind):
        self.kind = kind

    async def fetch(self, start, now):
        from clio.domains.solar_wind.observations import ingest_source as ingest_wind
        from clio.providers.solar_wind_loader import FIELDS
        await ingest_wind(self.kind)
        return await stored_native_frame(tuple(FIELDS[self.kind]), start, now, f'swpc.rtsw_{self.kind}')


def live_adapters():
    return {
        'swpc.rtsw_mag': SolarWindLiveAdapter('mag'),
        'swpc.rtsw_plasma': SolarWindLiveAdapter('plasma'),
        'swpc.kp': GeomagneticLiveAdapter('kp'),
        'swpc.dst': GeomagneticLiveAdapter('dst'),
        'swpc.f107': WideLiveAdapter(SWPC_Loader._fetch_f10_7_flux, ('f10_7',)),
    }
