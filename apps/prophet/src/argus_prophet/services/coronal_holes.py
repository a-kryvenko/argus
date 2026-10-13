"""Source and cache adapter for private coronal-hole feature extraction."""
import hashlib
import json
import logging
from datetime import timedelta
from pathlib import Path
from importlib.metadata import version
import httpx
from common.schemas.solar_wind_events import CoronalHoleFrame
from forecast_core.api import CORONAL_HOLE_CHANNELS as CHANNELS, extract_coronal_holes as extract
logger = logging.getLogger(__name__)

def prepare(inputs, client, url, token):
    """Catalog authorizes source selection; read only the canonical shared archive path."""
    from common.sdo_images import archive_root, archive_lock, read_image, utc
    from argus_prophet.services.source_cache import atomic_write
    from common.config import get_config
    cutoff = inputs.as_of
    hour = cutoff.replace(minute=0, second=0, microsecond=0)
    groups = {}
    for channel in (*CHANNELS, 'hmi_m'):
        try:
            response = client.get(url.rstrip('/')+'/internal/v1/observations/sdo-images',
                params={'start': (hour-timedelta(hours=6)).isoformat(), 'end': (hour+timedelta(hours=1)).isoformat(),
                        'as_of': cutoff.isoformat(), 'channel': channel},
                headers={'Authorization': f'Bearer {token}'})
            response.raise_for_status()
        except httpx.HTTPError:
            if channel != 'hmi_m':
                raise
            logger.warning('HMI catalog unavailable; preserving EUV-only candidates')
            continue
        for item in response.json()['data']['items']:
            metadata = item['metadata']
            if metadata['channel'] != channel:
                raise ValueError('SDO catalog returned a different channel')
            slot = utc(metadata['slot_at'])
            if (hour-timedelta(hours=6) <= slot <= hour and utc(metadata['available_at']) <= cutoff and
                utc(metadata['observed_at']) <= cutoff):
                groups.setdefault(slot, {})[channel] = metadata
    slots = sorted((s for s, group in groups.items() if set(CHANNELS).issubset(group)), reverse=True)
    if not slots:
        inputs.hss_input_status['coronal_holes'] = {'state': 'unavailable', 'reason': 'no_complete_recent_AIA_triplet'}
        return inputs
    slot = slots[0]
    selected = groups[slot]
    root = archive_root()
    images, receipts = {}, {}
    with archive_lock(root):
        for channel, reference in selected.items():
            image, metadata = read_image(root, slot, channel, as_of=cutoff)
            if metadata['sha256'] != reference['sha256'] or metadata['available_at'] != reference['available_at']:
                raise ValueError('SDO image differs from selected source receipt')
            images[channel], receipts[channel] = image, metadata
    cache_identity = hashlib.sha256(json.dumps({
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'libraries': {name: version(name) for name in ('numpy', 'scipy', 'sunpy', 'astropy')},
        'receipts': receipts}, sort_keys=True).encode()).hexdigest()
    cache = get_config().data_root/'prophet/hss-features'/f'{cache_identity}.json'
    if cache.exists():
        frame = CoronalHoleFrame.model_validate_json(cache.read_bytes())
    else:
        try:
            frame = extract(images, receipts)
        except ValueError:
            if 'hmi_m' not in images:
                raise
            logger.warning('HMI support rejected; preserving EUV-only candidates', exc_info=True)
            frame = extract({c: i for c, i in images.items() if c != 'hmi_m'},
                            {c: r for c, r in receipts.items() if c != 'hmi_m'})
            frame.magnetic_state = 'rejected'
        atomic_write(cache, frame.model_dump_json().encode())
    inputs.coronal_holes = frame
    inputs.hss_input_status['coronal_holes'] = dict(state='available', observed_at=frame.observed_at.isoformat(),
        magnetic_state=frame.magnetic_state, region_count=len(frame.regions), feature_version=frame.feature_version)
    if frame.magnetic_state == 'rejected':
        inputs.hss_input_status['coronal_holes']['rejected_hmi_sha256'] = receipts['hmi_m']['sha256']
        inputs.hss_input_status['coronal_holes']['rejected_hmi_quality'] = receipts['hmi_m']['header'].get('QUALITY')
    return inputs
