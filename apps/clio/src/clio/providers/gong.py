"""Bounded GONG transfers; observation times come from provider filenames."""
import gzip
import io
import re
from datetime import UTC, datetime, timedelta
from urllib.parse import urljoin, urlparse

ARCHIVE_URL = 'https://gong.nso.edu/archive/oQR/zqs/'
LIVE_URL = 'https://services.swpc.noaa.gov/products/gong/zqs/'

MAX_FITS_BYTES = 32 * 1024 * 1024


def observation_time(name):
    match = re.search(r'[a-z]+(\d{6})t(\d{4})[^/]*\.fits\.gz$', name, re.I)
    if match is None:
        return None
    return datetime.strptime('20' + ''.join(match.groups()), '%Y%m%d%H%M').replace(tzinfo=UTC)


def candidates(start, end, *, historical=False):
    import requests
    urls = [LIVE_URL]
    if historical:
        day = start.replace(hour=0, minute=0, second=0, microsecond=0)
        urls = []
        while day < end:
            urls.append(f'{ARCHIVE_URL}{day:%Y%m}/mrzqs{day:%y%m%d}/')
            day += timedelta(days=1)
    found = {}
    for index_url in urls:
        response = requests.get(index_url, timeout=(10, 60))
        if response.status_code == 404 and historical:
            continue
        response.raise_for_status()
        for name in re.findall(r'href=[\"\']([^\"\']+\.fits\.gz)[\"\']', response.text, re.I):
            url = urljoin(index_url, name)
            if urlparse(url).netloc != urlparse(index_url).netloc:
                continue
            observed = observation_time(urlparse(url).path)
            if observed is not None and start <= observed < end:
                # Stable choice when several stations advertise one timestamp.
                found[observed] = min(url, found.get(observed, url))
    return sorted(found.items(), reverse=True)


def download(url):
    import requests
    compressed = bytearray()
    with requests.get(url, stream=True, timeout=(10, 60)) as response:
        response.raise_for_status()
        for chunk in response.iter_content(65536):
            compressed.extend(chunk)
            if len(compressed) > MAX_FITS_BYTES:
                raise ValueError('GONG download exceeds 32 MiB')
    with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as source:
        content = source.read(MAX_FITS_BYTES + 1)
    if len(content) > MAX_FITS_BYTES:
        raise ValueError('GONG original exceeds 32 MiB')
    return content
