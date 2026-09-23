"""Clio owns raw AIA storage and causal feature reads; no Prophet filesystem sharing."""
import asyncio,logging,os
from datetime import UTC,datetime,timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from common.config import get_config
from common.schemas.forecast_inputs import AIAFeatureFrame
from argus_clio.db.models.aia_snapshot import AIASnapshot
from argus_clio.db.session import get_session_factory
from clio.dataloaders.aia import fetch_snapshot,hourly_slots
from clio.aia_features import sectors,weighted_mean,aligned_change,ROTATION_HOURS
logger=logging.getLogger(__name__)


def archive_root():
    return Path(os.getenv('ARGUS_AIA_ARCHIVE',str(get_config().data_root/'observations/aia193')))


async def collect_aia(*,now=None,history_days=40):
    now=now or datetime.now(UTC);slots=hourly_slots(now,history_days)
    async with get_session_factory()() as session:
        records=(await session.execute(select(AIASnapshot).where(AIASnapshot.slot_at>=slots[0],AIASnapshot.slot_at<=slots[-1]))).scalars().all()
    root=archive_root()
    existing={r.slot_at for r in records if (root/Path(r.raw_path)).is_file() and (root/Path(r.cache_path)).is_file()}
    pending=[slot for slot in slots if slot not in existing]
    # Recent images first: the forecast need not wait for historical warmup.
    pending.reverse();received=missing=failed=0
    loop=asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for begin in range(0,len(pending),4):
            outcomes=await asyncio.gather(*(loop.run_in_executor(pool,fetch_snapshot,slot,root) for slot in pending[begin:begin+4]),return_exceptions=True)
            async with get_session_factory()() as session:
                for result in outcomes:
                    if isinstance(result,BaseException):
                        failed+=1;logger.error('AIA collection failed: %s',result);continue
                    if result is None:missing+=1;continue
                    row=dict(result)
                    for key in ['raw_path','cache_path']:row[key]=str(Path(row[key]).relative_to(root.resolve()))
                    for key in ['slot_at','observed_at','available_at']:row[key]=datetime.fromisoformat(row[key])
                    await session.execute(insert(AIASnapshot).values(**row).on_conflict_do_nothing(index_elements=['slot_at']))
                    received+=1
                await session.commit()
    logger.info('AIA hourly archive: received=%s unavailable=%s failed=%s retained=%s',received,missing,failed,len(existing))
    if failed:raise RuntimeError(f'{failed} AIA snapshots failed; retry next schedule')
    return dict(received=received,missing=missing,retained=len(existing))


def feature_frames(records,as_of):
    # Use the identical six-hour subset for BOTH current areas and temporal pairs.
    records=sorted([r for r in records if r.slot_at.astimezone(UTC).hour%6==0 and r.slot_at.minute==0 and r.available_at<=as_of and r.observed_at<=as_of],key=lambda r:r.observed_at)
    if not records:return []
    masks=[];usable=[]
    for row in records:
        try:
            cache=Path(row.cache_path)
            if not cache.is_absolute():cache=archive_root()/cache
            with np.load(cache,allow_pickle=False) as saved:
                masks.append(saved['dark']);usable.append(row)
        except (OSError,ValueError) as exc:
            logger.warning('Unavailable AIA cache for %s: %s',row.slot_at,exc)
    records=usable
    times=pd.DatetimeIndex([r.observed_at for r in records])
    result=[];regions=list(sectors())
    for i,row in enumerate(records):
        if row.observed_at<as_of-timedelta(days=10):continue
        current=masks[i];values=dict(aia_valid_fraction=row.valid_fraction,aia_b0_deg=row.b0_deg);available=row.available_at
        for name,mask in regions:values['aia_area_'+name]=weighted_mean(current,mask)
        for label,hours in [('24h',24.),('rotation',ROTATION_HOURS)]:
            target=times[i]-pd.Timedelta(hours=hours);k=int(times.searchsorted(target));candidates=[j for j in [k-1,k] if 0<=j<i]
            j=min(candidates,key=lambda j:abs(times[j]-target)) if candidates else None
            change=np.full(current.shape,np.nan);values[f'aia_{label}_separation_h']=None
            if j is not None and abs(times[j]-target)<=pd.Timedelta(hours=12):
                change=aligned_change(current,masks[j],row.carrington_lon,records[j].carrington_lon)
                values[f'aia_{label}_separation_h']=(times[i]-times[j]).total_seconds()/3600
                available=max(available,records[j].available_at)
            for name,mask in regions:
                values[f'aia_delta_{label}_{name}']=weighted_mean(change,mask)
                values[f'aia_overlap_{label}_{name}']=weighted_mean(np.isfinite(change).astype(float),mask)
        values={k:(float(v) if v is not None and np.isfinite(v) else None) for k,v in values.items()}
        result.append(AIAFeatureFrame(slot_at=row.slot_at,observed_at=row.observed_at,available_at=available,sha256=row.sha256,features=values))
    return result


async def load_aia_features(session,as_of):
    records=(await session.execute(select(AIASnapshot).where(AIASnapshot.observed_at>=as_of-timedelta(days=40),AIASnapshot.observed_at<=as_of,AIASnapshot.available_at<=as_of).order_by(AIASnapshot.observed_at).limit(1001))).scalars().all()
    if len(records)>1000:raise ValueError('AIA read exceeds bounded40day hourly archive')
    return await asyncio.to_thread(feature_frames,records,as_of)
