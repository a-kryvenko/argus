"""Read calibrated measurements from the latest recorded model inputs."""
from datetime import datetime, timedelta

from sqlalchemy import text

from common.schemas.solar_indices import SolarIndices
from prophet.db.session import get_db_session


async def read_solar_indices():
    async with get_db_session() as session:
        await session.execute(text('SET LOCAL statement_timeout=3000'))
        row = (await session.execute(text("""SELECT
            input_snapshot->>'as_of' AS as_of,
            input_snapshot->'measurements' AS measurements
            FROM forecast_run WHERE input_snapshot IS NOT NULL
            AND product<>'solar-wind-events'
            ORDER BY started_at DESC, id DESC LIMIT 1"""))).mappings().first()
    if row is None:
        return None
    as_of = datetime.fromisoformat(row['as_of'])
    cutoff = as_of-timedelta(days=14)
    return SolarIndices(as_of=as_of, items=[point for point in row['measurements']
        if point['metric'] in ('f10_7', 's10', 'm10', 'y10')
        and cutoff <= datetime.fromisoformat(point['observed_at']) <= as_of])
