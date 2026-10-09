"""Prepare optional PROSWIN records out of band; restart resumes completed slots."""
import argparse
import hashlib
from datetime import UTC,datetime
import io
import json
import logging
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import fcntl
import httpx
import numpy as np
import pandas as pd
from common.config import get_config
from common.schemas.forecast_inputs import ObservationInputs, ProswinPrediction
from argus_prophet.services.source_cache import atomic_write
from argus_prophet.services.proswin_cache import cache_root

logger=logging.getLogger(__name__)


def get(client,url,**kwargs):
    for attempt in range(4):
        try:
            r=client.get(url,**kwargs);r.raise_for_status();return r
        except httpx.HTTPError:
            if attempt==3: raise
            time.sleep(2**attempt)


def physical_features(inputs,slot,sunspots):
    from astropy.time import Time
    from sunpy.coordinates.sun import B0
    from proswin.data_management.physical_data_loader import create_solar_cycle_feature
    valid=slot+pd.Timedelta(hours=96)
    # Clio's hourly timestamps label hour starts. These hours are 22–24 days old.
    history=pd.Series({pd.Timestamp(p.issue_time):p.v for p in inputs.speed_observations})
    window=pd.date_range(valid-pd.Timedelta(hours=672),valid-pd.Timedelta(hours=624),freq='h')
    wind=history.reindex(window).to_numpy(float)
    if not np.isfinite(wind).all() or (wind<=0).any(): raise ValueError('Missing required hourly speed')
    months=pd.to_datetime(dict(year=sunspots[0],month=sunspots[1],day=1),utc=True)
    s=pd.Series(sunspots[3].to_numpy(float),index=months)
    selected=s[(s.index+pd.offsets.MonthBegin(1)+pd.Timedelta(days=7)<=slot)&s.ge(0)].tail(12)
    if len(selected)!=12 or not selected.index.equals(pd.date_range(selected.index[0],periods=12,freq='MS')):
        raise ValueError('Missing consecutive sunspot months')
    cycle=create_solar_cycle_feature().loc[valid.tz_localize(None)]
    return [*wind,*selected.to_numpy(),float(B0(Time(slot.to_pydatetime())).value),float(cycle)]


def cycle(runtime,client,url,token,inputs=None):
    now=pd.Timestamp(inputs.as_of) if inputs is not None else pd.Timestamp.now(tz='UTC')
    hour=now.floor('h');root=cache_root()
    headers={'Authorization':f'Bearer {token}'}
    if inputs is None:
        response=get(client,url+'/internal/v1/observations/forecast-inputs',params={'as_of':now.isoformat()},headers=headers)
        inputs=ObservationInputs.model_validate(response.json())
        if pd.Timestamp(inputs.as_of)!=now: raise ValueError('Clio returned wrong input time')
    # Preserve first-seen timing of the snapshot actually used. It is never backdated.
    ssn=root/'sunspots.csv';meta=root/'sunspots.json'
    try:
        snapshot = json.loads(meta.read_text())
        fresh = (snapshot['sha256'] == hashlib.sha256(ssn.read_bytes()).hexdigest()
                 and now.timestamp()-ssn.stat().st_mtime < 86400)
    except (OSError, ValueError, KeyError):
        fresh = False
    if not fresh:
        response=get(client,'https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv')
        pd.read_csv(io.BytesIO(response.content),sep=';',header=None)
        atomic_write(ssn,response.content)
        atomic_write(meta,json.dumps({'retrieved_at':pd.Timestamp.now(tz='UTC').isoformat(), 'sha256': hashlib.sha256(response.content).hexdigest()}).encode())
    snapshot = json.loads(meta.read_text())
    if pd.Timestamp(snapshot['retrieved_at']) > now:
        logger.info('SILSO snapshot arrived after input cutoff; available for the next request')
        return
    sunspots=pd.read_csv(ssn,sep=';',header=None)
    groups={}
    for channel in ['aia171','aia211']:
        response=get(client,url+'/internal/v1/observations/sdo-images',headers=headers,params={
            'start':(hour-pd.Timedelta(hours=95)).isoformat(),'end':(hour+pd.Timedelta(hours=1)).isoformat(),
            'as_of':now.isoformat(),'channel':channel})
        for item in response.json()['data']['items']:
            m=item['metadata'];slot=pd.Timestamp(m['slot_at'])
            if pd.Timestamp(m['available_at'])<=now and pd.Timestamp(m['observed_at'])<=now:
                groups.setdefault(slot,{})[channel[3:]]=m
    completed=0
    from forecast.inference.proswin_runtime import digest
    for slot,images in sorted(groups.items(),reverse=True):
        valid=slot+pd.Timedelta(hours=96);dest=root/'predictions'/f'{valid:%Y%m%dT%H}.json'
        if set(images)!={'171','211'}: continue
        if dest.exists():
            try:
                saved = ProswinPrediction.model_validate_json(dest.read_text())
                if saved.valid_time == valid and saved.image_slot == slot and slot <= saved.available_at <= now:
                    continue
            except (OSError, ValueError):
                logger.warning('Rebuilding invalid PROSWIN record: %s', dest)
        try:
            features=physical_features(inputs,slot,sunspots)
            with TemporaryDirectory(prefix='proswin-') as temporary:
                crops={}
                for wl,m in images.items():
                    response=get(client,url+'/internal/v1/observations/files/aia/'+m['sha256'],headers=headers,
                        params={'slot_at':slot.isoformat(),'channel':'aia'+wl})
                    if len(response.content)>128*1024*1024: raise ValueError('Oversized AIA original')
                    path=Path(temporary)/f'{wl}.fits';path.write_bytes(response.content)
                    if digest(path)!=m['sha256']: raise ValueError('AIA checksum mismatch')
                    crops[wl]=runtime.crop(path,wl,slot)
                value=runtime.predict(crops,features)
            record={'valid_time':valid.isoformat(),'image_slot':slot.isoformat(),
                'available_at':pd.Timestamp.now(tz='UTC').isoformat(),'value':value,'model_version':'proswin-fold1-nrt-v1',
                'source_cutoff':now.isoformat(), 'inputs_read_at':inputs.read_at.isoformat(),'image_receipts':images,'features':features,
                'ssn_sha256':digest(ssn),'ssn_snapshot':json.loads(meta.read_text()),
                'ssn_month_selection':'period_end+7d<=image_slot; revised values known at generation'}
            atomic_write(dest,json.dumps(record,allow_nan=False).encode());completed+=1
            logger.info('PROSWIN ready: %s %.2f km/s',valid,value)
        except (OSError,ValueError,KeyError,RuntimeError,httpx.HTTPError):
            logger.exception('PROSWIN unavailable for %s; DLinear remains active',valid)
    logger.info('PROSWIN cycle: %d new predictions, %d image slots',completed,len(groups))
    # Forecast cache is reproducible; retain 45 days for audit, never remove live horizons.
    for path in (root/'predictions').glob('*.json'):
        if path.stat().st_mtime<now.timestamp()-45*86400: path.unlink()


def main():
    import signal
    from datetime import datetime, UTC
    from argus_prophet.services.heavy_task import heavy_task
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--job', type=Path, required=True)
    args=p.parse_args()
    job=args.job
    request=json.loads((job/'request.json').read_text())
    remaining=(datetime.fromisoformat(request['deadline'])-datetime.now(UTC)).total_seconds()
    if remaining <= 0 or (job/'cancelled').exists():
        raise SystemExit(1)
    # Also bounds an orphaned child if the queue consumer is killed.
    signal.setitimer(signal.ITIMER_REAL, remaining)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    for name in ("XDG_CACHE_HOME", "TORCHINDUCTOR_CACHE_DIR", "SUNPY_CONFIGDIR", "SUNPY_DOWNLOADDIR", "MPLCONFIGDIR"):
        if os.environ.get(name):
            Path(os.environ[name]).mkdir(parents=True, exist_ok=True)
    inputs=ObservationInputs.model_validate_json((job/'inputs.json').read_text())
    url=os.environ['OBSERVATIONS_URL'].rstrip('/');token=os.environ['OBSERVATIONS_SERVICE_TOKEN']
    with heavy_task(get_config().data_root):
        from forecast.inference.proswin_runtime import ProswinRuntime
        runtime=ProswinRuntime(Path(os.getenv('PROPHET_PROSWIN_ASSETS',str(get_config().data_root/'models/proswin-fold1-nrt-v1'))))
        with httpx.Client(timeout=httpx.Timeout(60,connect=10),follow_redirects=False) as client:
            cycle(runtime,client,url,token,inputs=inputs)
        from argus_prophet.services.proswin_cache import read_predictions
        records=read_predictions(inputs.as_of, ready_at=datetime.now(UTC))
        atomic_write(job/'predictions.json', json.dumps([r.model_dump(mode='json') for r in records]).encode())

if __name__=='__main__':main()
