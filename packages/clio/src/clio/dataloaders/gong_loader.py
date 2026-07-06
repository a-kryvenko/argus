from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
from pathlib import Path
from urllib.request import urlopen
import gzip
import shutil
import re

ARCHIVE_URL = "https://gong.nso.edu/archive/oQR/zqs/"

def fetch_gong(start: datetime, end: datetime, out_dir: Path):
    out_dir.mkdir(exist_ok=True)

    d = start
    while d <= end:
        _load_daily_gong(d, out_dir)
        d += timedelta(days=1)

def _load_daily_gong(day: datetime, out_dir: Path):
    y = day.strftime("%Y")
    y_s = y[2:]
    m = day.strftime("%m")
    d = day.strftime("%d")
    url = f"{ARCHIVE_URL}{y}{m}/mrzqs{y_s}{m}{d}/"

    r = requests.get(url, timeout=60)
    r.raise_for_status()

    if r.status_code != 200:
        print("missing:", url)
        return None
    
    soup = BeautifulSoup(r.text, "html.parser")
    
    for a in soup.select("table tr td a"):
        href = a.get("href")
        if href and href[-8:] == ".fits.gz":
            _load_fits_file(url + href, out_dir)
            

def _load_fits_file(url: str, out_dir: Path):
    gz_path = out_dir / Path(url).name

    m = re.search(r"([a-z]+)(\d{2})(\d{2})(\d{2})t(\d{2})(\d{2})", gz_path.name)
    prefix, yy, mm, dd, hour, minute = m.groups()

    dt = datetime(
        year=2000 + int(yy),
        month=int(mm),
        day=int(dd),
        hour=int(hour),
        minute=int(minute),
    )

    rounded_hour = dt.replace(minute=0, second=0, microsecond=0)

    new_name = f"{prefix}_{rounded_hour:%Y%m%d_%H_00}.fits"
    out_path = out_dir / new_name

    with urlopen(url) as response:
        with gzip.GzipFile(fileobj=response) as gz:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(gz, f_out)
