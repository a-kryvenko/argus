"""Native GOES primary X-ray irradiance and integral proton flux from NOAA SWPC."""
from datetime import UTC, datetime

import pandas as pd
import requests

from clio.ingestion.utils.products import PRODUCTS
from clio.providers._contract import sample, samples_frame


BASE_URL = 'https://services.swpc.noaa.gov/json/goes/primary'
FEEDS = {
    'goes.xrs': ('xrays', {'0.05-0.4nm': 'xrs_a', '0.1-0.8nm': 'xrs_b'}),
    'goes.protons': ('integral-protons', {'>=10 MeV': 'proton_10mev', '>=100 MeV': 'proton_100mev'}),
}


def parse_records(product: str, payload: object, received_at: datetime) -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f'{product}: expected a nonempty JSON record list')
    _, bands = FEEDS[product]
    slot_minutes = PRODUCTS[product].slot_minutes
    rows = []
    for row in payload:
        if not isinstance(row, dict) or not {'time_tag', 'energy', 'flux'} <= row.keys():
            raise ValueError(f'{product}: unexpected record schema')
        energy = row['energy']
        if not isinstance(energy, str) or energy not in bands:
            continue
        try:
            observed = datetime.fromisoformat(row['time_tag'])
        except (TypeError, ValueError):
            continue
        if observed.tzinfo is None or observed > received_at:
            continue
        observed = observed.astimezone(UTC)
        if observed.minute % slot_minutes or observed.second or observed.microsecond:
            continue
        # SWPC XRS flux already includes the electron correction. The source
        # contamination marker describes that correction and is not a fill flag.
        satellite = row.get('satellite')
        spacecraft = f'GOES-{satellite}' if isinstance(satellite, int) and not isinstance(satellite, bool) else None
        rows.append(sample(bands[energy], observed, row['flux'], spacecraft=spacecraft))
    return samples_frame(rows)


def fetch_records(product: str, *, historical: bool = False) -> pd.DataFrame:
    feed, _ = FEEDS[product]
    window = '7-day' if historical else '6-hour'
    response = requests.get(f'{BASE_URL}/{feed}-{window}.json', timeout=(5, 25))
    response.raise_for_status()
    return parse_records(product, response.json(), datetime.now(UTC))
