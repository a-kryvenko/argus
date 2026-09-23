"""Read causal AIA features from Clio's archived snapshots."""
import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import select

from common.schemas.forecast_inputs import AIAFeatureFrame
from argus_clio.db.models.aia_snapshot import AIASnapshot
from argus_clio.services.aia_archive import archive_root
from clio.aia_features import sectors, weighted_mean, aligned_change, ROTATION_HOURS

logger = logging.getLogger(__name__)


def feature_frames(records, as_of: datetime) -> list[AIAFeatureFrame]:
    # Use the identical six-hour subset for current areas and temporal pairs.
    records = sorted(
        (row for row in records
         if row.slot_at.astimezone(UTC).hour % 6 == 0
         and row.slot_at.minute == 0
         and row.available_at <= as_of
         and row.observed_at <= as_of),
        key=lambda row: row.observed_at,
    )
    if not records:
        return []

    masks = []
    usable = []
    for row in records:
        try:
            cache = Path(row.cache_path)
            if not cache.is_absolute():
                cache = archive_root() / cache
            with np.load(cache, allow_pickle=False) as saved:
                masks.append(saved['dark'])
                usable.append(row)
        except (OSError, ValueError) as exc:
            logger.warning('Unavailable AIA cache for %s: %s', row.slot_at, exc)

    records = usable
    times = pd.DatetimeIndex([row.observed_at for row in records])
    result = []
    regions = list(sectors())
    for i, row in enumerate(records):
        if row.observed_at < as_of - timedelta(days=10):
            continue
        current = masks[i]
        values = dict(aia_valid_fraction=row.valid_fraction, aia_b0_deg=row.b0_deg)
        available = row.available_at
        for name, mask in regions:
            values['aia_area_' + name] = weighted_mean(current, mask)

        for label, hours in [('24h', 24.), ('rotation', ROTATION_HOURS)]:
            target = times[i] - pd.Timedelta(hours=hours)
            insertion = int(times.searchsorted(target))
            candidates = [j for j in (insertion - 1, insertion) if 0 <= j < i]
            previous = min(candidates, key=lambda j: abs(times[j] - target)) if candidates else None
            change = np.full(current.shape, np.nan)
            values[f'aia_{label}_separation_h'] = None
            if previous is not None and abs(times[previous] - target) <= pd.Timedelta(hours=12):
                change = aligned_change(
                    current, masks[previous], row.carrington_lon, records[previous].carrington_lon,
                )
                values[f'aia_{label}_separation_h'] = (
                    times[i] - times[previous]
                ).total_seconds() / 3600
                available = max(available, records[previous].available_at)
            for name, mask in regions:
                values[f'aia_delta_{label}_{name}'] = weighted_mean(change, mask)
                values[f'aia_overlap_{label}_{name}'] = weighted_mean(
                    np.isfinite(change).astype(float), mask,
                )

        values = {
            key: float(value) if value is not None and np.isfinite(value) else None
            for key, value in values.items()
        }
        result.append(AIAFeatureFrame(
            slot_at=row.slot_at, observed_at=row.observed_at, available_at=available,
            sha256=row.sha256, features=values,
        ))
    return result


async def load_aia_features(session, as_of: datetime) -> list[AIAFeatureFrame]:
    records = (await session.execute(
        select(AIASnapshot).where(
            AIASnapshot.observed_at >= as_of - timedelta(days=40),
            AIASnapshot.observed_at <= as_of,
            AIASnapshot.available_at <= as_of,
        ).order_by(AIASnapshot.observed_at).limit(1001)
    )).scalars().all()
    if len(records) > 1000:
        raise ValueError('AIA read exceeds bounded 40-day hourly archive')
    return await asyncio.to_thread(feature_frames, records, as_of)
