import asyncio
from datetime import datetime, timedelta

from sqlalchemy import delete, or_, and_

from clio.db.models.measurement import Measurement
from clio.db.models.normalized_observation import NormalizedObservation
from clio.db.session import get_db_session
from clio.ingestion.products import PRODUCTS


async def cleanup(config, now: datetime):
    products = [name for name, product in PRODUCTS.items() if product.kind == 'numeric' and 'live' in product.modes]
    
    async with get_db_session() as session:
        # Published means can describe an older window while still being the
        # latest release. Retain them for their configured collection horizon.
        means = {m: p.backfill_days for m, p in config.observations.items() if m.endswith('_81mean')}
        expired = or_(and_(Measurement.metric.not_in(list(means)),
                           Measurement.observed_at < now - timedelta(days=config.retention.live_days)),
                      *(and_(Measurement.metric == m,
                             Measurement.observed_at < now - timedelta(days=max(days, config.retention.live_days)))
                        for m, days in means.items()))
        live = await session.execute(delete(Measurement).where(
            Measurement.source_product.in_(products),
            expired))
        normalized = await session.execute(delete(NormalizedObservation).where(
            NormalizedObservation.observed_at < now - timedelta(days=config.retention.normalized_days)))
        await session.commit()
    
    result = {'live_removed': live.rowcount, 'normalized_removed': normalized.rowcount}
    
    if config.sdo_browse_enabled:
        from clio.observations.sdo_browse import cleanup
        result['sdo_browse_removed'] = await asyncio.to_thread(cleanup)
    
    if config.sdo_images.enabled:
        from clio.ingestion.sdo_images import cycle
        from clio.observations.sdo_archive import archive_root
        result['sdo'] = await asyncio.to_thread(cycle, archive_root(), 'cleanup', clock=lambda: now)

    return result
