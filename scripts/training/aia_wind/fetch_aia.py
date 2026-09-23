"""Resume bounded AIA 193/211 level-1.5 downloads from the JSOC archive."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import threading

import pandas as pd
import requests
from astropy.io import fits
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://jsoc1.stanford.edu/data/aia/synoptic"


def download_archive(start, end, output, *, cadence_hours=24, workers=2, channel=193):
    """UTC [start, end), exact archive slots, missing frames are logged, not filled."""
    start, end = pd.to_datetime(start, utc=True), pd.to_datetime(end, utc=True)
    if channel not in (193, 211):
        raise ValueError("Supported AIA channels: 193, 211")
    if end <= start or cadence_hours < 1 or cadence_hours > 24 or 24 % cadence_hours:
        raise ValueError("Use an increasing interval and a cadence dividing 24 hours")
    if workers not in range(1, 5):
        raise ValueError("Use 1–4 download workers")
    if start != start.floor("h") or end != end.floor("h"):
        raise ValueError("Archive interval must use full UTC hours")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    times = pd.date_range(start, end, freq=f"{cadence_hours}h", inclusive="left")
    print(f"AIA: {len(times)} slots; about {len(times)*1.3/1000:.1f} GB before caching", flush=True)
    local = threading.local()

    def session_for_thread():
        if not hasattr(local, "session"):
            local.session = requests.Session()
            local.session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                status_forcelist=[429, 500, 502, 503, 504])))
        return local.session

    def fetch(t):
        name = f"AIA{t:%Y%m%d_%H%M}_{channel:04d}.fits"
        url = f"{BASE_URL}/{t:%Y/%m/%d}/H{t:%H}00/{name}"
        path = output/f"{t:%Y/%m/%d}"/name
        row = dict(slot=t.isoformat(), url=url, path=str(path.resolve()))
        try:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                with session_for_thread().get(url, timeout=(15, 90), stream=True) as response:
                    if response.status_code == 404:
                        return {**row, "status": "missing"}
                    response.raise_for_status()
                    temporary = path.with_suffix(".part")
                    with temporary.open("wb") as stream:
                        for chunk in response.iter_content(1024*1024):
                            stream.write(chunk)
                # An HTML error page must never be cached as a FITS observation.
                with fits.open(temporary) as hdus:
                    if not any(h.data is not None and h.data.ndim == 2 for h in hdus):
                        raise ValueError("No FITS image")
                temporary.replace(path)
            return {**row, "status": "ok", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size}
        except (requests.RequestException, OSError, ValueError) as exc:
            return {**row, "status": "error", "reason": str(exc)}

    # Append every outcome immediately; interrupted downloads resume by file existence.
    rows = []
    with (output/"downloads.jsonl").open("a") as log, ThreadPoolExecutor(max_workers=workers) as pool:
        for i, row in enumerate(pool.map(fetch, times), 1):
            rows.append(row)
            log.write(json.dumps(row)+"\n"); log.flush()
            if i % 50 == 0 or i == len(times):
                print(f"AIA downloads: {i}/{len(times)}", flush=True)
    report = pd.DataFrame(rows)
    print(report.status.value_counts().to_string())
    errors = report[report.status == "error"]
    if len(errors):
        raise RuntimeError(f"{len(errors)} downloads failed; see {output/'downloads.jsonl'} and rerun")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="Exclusive UTC boundary")
    parser.add_argument("--output", type=Path, default=Path("data/raw/aia193_daily"))
    parser.add_argument("--cadence-hours", type=int, default=24)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--channel", type=int, choices=[193, 211], default=193)
    args = parser.parse_args()
    download_archive(args.start, args.end, args.output, cadence_hours=args.cadence_hours, workers=args.workers, channel=args.channel)
