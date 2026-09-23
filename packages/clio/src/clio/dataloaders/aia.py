"""Hourly AIA193 original FITS, validated and atomically stored by Clio."""
from datetime import UTC,datetime,timedelta
from pathlib import Path
import hashlib,json,os,tempfile
import numpy as np
import requests
from clio.aia_features import extract_frame

BASE_URL='https://jsoc1.stanford.edu/data/aia/synoptic'
MAX_BYTES=32*1024*1024


def hourly_slots(now,history_days=40):
    if now.tzinfo is None or not 1<=history_days<=60:raise ValueError('Aware time and1–60 history days required')
    end=now.astimezone(UTC).replace(minute=0,second=0,microsecond=0)
    start=end-timedelta(days=history_days)
    return [start+timedelta(hours=h) for h in range(history_days*24+1)]


def fetch_snapshot(slot,root):
    if slot.tzinfo is None or slot.minute or slot.second or slot.microsecond:raise ValueError('Expected aware hourly slot')
    slot=slot.astimezone(UTC);root=Path(root)
    name=f'AIA{slot:%Y%m%d_%H%M}_0193.fits';directory=root/f'{slot:%Y/%m/%d}';directory.mkdir(parents=True,exist_ok=True)
    raw=directory/name;cache=directory/(name+'.npz');receipt=directory/(name+'.json')
    if raw.exists() and receipt.exists():
        record=json.loads(receipt.read_text())
        if record.get('rejected'):
            if hashlib.sha256(raw.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed rejected AIA FITS')
            return None
    if raw.exists() and cache.exists() and receipt.exists():
        record=json.loads(receipt.read_text())
        if hashlib.sha256(raw.read_bytes()).hexdigest()!=record['sha256']:raise ValueError('Changed archived AIA FITS')
        return {**record,'raw_path':str(raw.resolve()),'cache_path':str(cache.resolve())}
    fd,temp=tempfile.mkstemp(prefix=name+'.',suffix='.part',dir=directory);os.close(fd);temporary=Path(temp)
    try:
        url=f'{BASE_URL}/{slot:%Y/%m/%d}/H{slot:%H}00/{name}'
        with requests.get(url,stream=True,timeout=(10,60)) as response:
            if response.status_code==404:return None
            response.raise_for_status();size=0
            with temporary.open('wb') as f:
                for block in response.iter_content(1024*1024):
                    size+=len(block)
                    if size>MAX_BYTES:raise ValueError('Oversized AIA FITS')
                    f.write(block)
        received=datetime.now(UTC)
        try:
            dark,ratio,meta=extract_frame(temporary)
        except ValueError as exc:
            # Retain original but do not repeatedly process permanently failed QC.
            rejected=dict(rejected=True,slot_at=slot.isoformat(),available_at=received.isoformat(),sha256=hashlib.sha256(temporary.read_bytes()).hexdigest(),reason=str(exc))
            temporary.replace(raw)
            receipt_temp=receipt.with_suffix(receipt.suffix+'.part');receipt_temp.write_text(json.dumps(rejected,indent=2));receipt_temp.replace(receipt)
            return None
        observed=datetime.fromisoformat(meta['observed_at']).astimezone(UTC)
        if abs((observed-slot).total_seconds())>600 or observed>received:raise ValueError('AIA timestamp does not match requested slot')
        # Original FITS bytes are never resized or transformed.
        temporary.replace(raw);meta['path']=str(raw.resolve())
        cache_temp=cache.with_suffix(cache.suffix+'.part')
        with cache_temp.open('wb') as f:np.savez_compressed(f,dark=dark,ratio=ratio,metadata=json.dumps(meta))
        cache_temp.replace(cache)
        record=dict(slot_at=slot.isoformat(),observed_at=observed.isoformat(),available_at=received.isoformat(),sha256=meta['sha256'],raw_path=str(raw.resolve()),cache_path=str(cache.resolve()),b0_deg=meta['b0_deg'],valid_fraction=meta['valid_fraction'],carrington_lon=meta['carrington_lon'])
        receipt_temp=receipt.with_suffix(receipt.suffix+'.part');receipt_temp.write_text(json.dumps(record,indent=2));receipt_temp.replace(receipt)
        return record
    finally:
        temporary.unlink(missing_ok=True)
