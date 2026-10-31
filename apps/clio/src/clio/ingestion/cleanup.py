import asyncio
from datetime import datetime, timedelta

from sqlalchemy import delete

from clio.db.models.measurement import Measurement
from clio.db.models.normalized_observation import NormalizedObservation
from clio.db.session import get_db_session
from clio.ingestion.products import PRODUCTS


async def cleanup(config, now: datetime):
    products = [name for name, product in PRODUCTS.items() if product.kind == 'numeric' and 'live' in product.modes]
    
    async with get_db_session() as session:
        live = await session.execute(delete(Measurement).where(
            Measurement.source_product.in_(products),
            Measurement.observed_at < now - timedelta(days=config.retention.live_days)))
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
