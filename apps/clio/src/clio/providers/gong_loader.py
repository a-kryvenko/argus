import gzip
import re
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urljoin
from urllib.request import urlopen

import requests
from bs4 import BeautifulSoup

ARCHIVE_URL = "https://gong.nso.edu/archive/oQR/zqs/"
LIVE_URL = "https://services.swpc.noaa.gov/products/gong/zqs/"

class GONG_Loader:
    def load_live(output_file: Path):
        r = requests.get(LIVE_URL)
        r.raise_for_status()

        matches = re.findall(
            r'href="([^"]+\.fits\.gz)"',
            r.text,
            flags=re.IGNORECASE,
        )

        if not matches:
            raise RuntimeError("No .fits.gz files found")

        latest = matches[-1]
        url = urljoin(LIVE_URL, latest)

        with TemporaryDirectory() as tmpdir:
            gz_path = Path(tmpdir) / Path(url).name

            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(gz_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)

            with gzip.open(gz_path, "rb") as src, open(output_file, "wb") as dst:
                shutil.copyfileobj(src, dst)

    def load_historical(start_date: datetime, end_date: datetime, output_dir: Path) -> Path:
        output_dir.mkdir(exist_ok=True)
        
        d = start_date
        while d <= end_date:
            GONG_Loader._load_daily_gong(d, output_dir)
            d += timedelta(days=1)

    def _load_daily_gong(date: datetime, output_dir: Path) -> bool:
        y = date.strftime("%Y")
        y_s = y[2:]
        m = date.strftime("%m")
        d = date.strftime("%d")
        url = f"{ARCHIVE_URL}{y}{m}/mrzqs{y_s}{m}{d}/"

        r = requests.get(url, timeout=60)
        r.raise_for_status()

        if r.status_code != 200:
            print("missing:", url)
            return False
        
        soup = BeautifulSoup(r.text, "html.parser")
        
        for a in soup.select("table tr td a"):
            href = a.get("href")
            if href and href[-8:] == ".fits.gz":
                GONG_Loader._load_fits_gz_file(url + href, output_dir)

        return True


    def _load_fits_gz_file(url: str, output_dir: Path):
        gz_path = output_dir / Path(url).name

        m = re.search(r"([a-z]+)(\d{2})(\d{2})(\d{2})t(\d{2})(\d{2})", gz_path.name)
        prefix, yy, mm, dd, hour, minute = m.groups()

        issue_time = datetime(
            year=2000 + int(yy),
            month=int(mm),
            day=int(dd),
            hour=int(hour),
            minute=int(minute),
            tzinfo=UTC
        )

        new_name = f"{prefix}_{issue_time:%Y%m%d_%H}_00.fits"
        out_path = output_dir / new_name

        with (
            urlopen(url) as response,
            gzip.GzipFile(fileobj=response) as gz,
            open(out_path, "wb") as f_out
        ):
            shutil.copyfileobj(gz, f_out)
