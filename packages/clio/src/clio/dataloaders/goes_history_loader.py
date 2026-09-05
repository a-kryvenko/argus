"""Science-quality NOAA GOES-R EUVS/XRS archive, normalized like fetch_goes.

Uses annual minute EUVS files and mission-length daily XRS-B background files.
See the NOAA GOES-R EUVS and XRS L2 product guides in the archive docs directory.
"""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
import logging
import re
import tempfile
from typing import Iterator, Sequence
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from clio.dataloaders.goes_loader import EUVS_LINE_COLUMNS


ARCHIVE_URL = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/"
EUVS_PRODUCT = "euvs-l2-avg1m"
XRS_PRODUCT = "xrsf-l2-bkd1d"
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoesArchiveFile:
    satellite: int
    product: str
    url: str
    year: int | None = None


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(value for name, value in attrs if name == "href" and value)


def archive_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def discover_goes_history(
    years: Sequence[int],
    satellites: Sequence[int],
    session: requests.Session,
    *,
    archive_url: str = ARCHIVE_URL,
) -> list[GoesArchiveFile]:
    """Discover the newest published file versions without hardcoding versions.

    Missing annual files/404 directories mean unavailable coverage, not a reason
    to substitute operational data. Other HTTP failures propagate.
    """
    files = []
    for satellite in satellites:
        for product in (EUVS_PRODUCT, XRS_PRODUCT):
            directory = f"{archive_url.rstrip('/')}/goes{satellite}/l2/data/{product}_science/"
            response = session.get(directory, timeout=(20, 120))
            if response.status_code == 404:
                logger.warning("No science-quality archive at %s", directory)
                continue
            response.raise_for_status()
            parser = _Links()
            parser.feed(response.text)
            span = r"y(?P<year>\d{4})" if product == EUVS_PRODUCT else r"s(?P<start>\d{8})_e(?P<end>\d{8})"
            pattern = re.compile(rf"sci_{product}_g{satellite}_{span}_v(?P<version>\d+(?:-\d+)*)\.nc")
            candidates = {}
            for link in parser.links:
                # Only accept bare archive filenames, not paths or external links.
                match = pattern.fullmatch(link)
                if not match:
                    continue
                year = int(match["year"]) if product == EUVS_PRODUCT else None
                if year is not None and year not in years:
                    continue
                version = tuple(int(part) for part in match["version"].split("-"))
                rank = (version, match["end"] if year is None else "")
                if year not in candidates or rank > candidates[year][0]:
                    candidates[year] = (rank, link)
            files.extend(
                GoesArchiveFile(satellite, product, urljoin(directory, link), year)
                for year, (_, link) in candidates.items()
            )
    return files


def cache_archive_file(
    source: GoesArchiveFile,
    cache_dir: Path,
    session: requests.Session,
    *,
    refresh: bool = False,
) -> Path:
    """Cache complete NetCDF downloads atomically; reuse them after interruption."""
    import xarray as xr

    path = cache_dir / Path(urlparse(source.url).path).name
    cache_dir.mkdir(parents=True, exist_ok=True)
    if path.is_file() and not refresh:
        with xr.open_dataset(path, engine="h5netcdf"):
            pass
        return path
    temporary = None
    try:
        with session.get(source.url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(dir=cache_dir, suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    stream.write(chunk)
        # Never promote an interrupted download or an HTML error page to cache.
        with xr.open_dataset(temporary, engine="h5netcdf"):
            pass
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def read_euvs_archive(path: Path, satellite: int) -> pd.DataFrame:
    """Read measured line irradiances and standardized Mg II, preserving quality."""
    import xarray as xr

    mapping = {f"irr_{line}": column for line, column in EUVS_LINE_COLUMNS.items()
               if line != "mgii_index"}
    flags = [f"{name}_flag" for name in mapping] + ["MgII_flag"]
    with xr.open_dataset(path, engine="h5netcdf") as data:
        required = [*mapping, *flags, "MgII_standard", "au_factor", "geocorona_flag"]
        missing = set(required).difference(data.variables)
        if missing:
            raise ValueError(f"{path.name}: missing EUVS variables {sorted(missing)}")
        if data.attrs.get("platform") != f"g{satellite}":
            raise ValueError(f"{path.name}: unexpected satellite platform")
        frame = data[required].to_dataframe().reset_index().rename(columns={
            "time": "timestamp", "MgII_standard": "goes_mgii_index",
            "au_factor": "goes_au_factor", **mapping,
        })
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    numeric = [*mapping.values(), "goes_mgii_index", "goes_au_factor"]
    frame[numeric] = frame[numeric].astype(float)
    combined_flags = np.bitwise_or.reduce(
        frame[flags].fillna(4).to_numpy(dtype=np.int64), axis=1,
    )
    frame["goes_eclipse"] = (combined_flags & 1) != 0
    frame["goes_lunar_transit"] = (combined_flags & 2) != 0
    frame["goes_geocorona"] = frame["geocorona_flag"].ne(0)
    frame["goes_euvs_quality_valid"] = (
        frame[flags].eq(0).all(axis=1) & ~frame["goes_geocorona"]
        & np.isfinite(frame[numeric]).all(axis=1) & frame["goes_au_factor"].gt(0)
    )
    frame["goes_euvs_satellite"] = np.int16(satellite)
    frame["goes_euvs_file"] = pd.Series(path.name, index=frame.index, dtype="string")
    return frame.drop(columns=[*flags, "geocorona_flag"]).sort_values("timestamp").reset_index(drop=True)


def read_xrs_archive(path: Path, satellite: int) -> pd.DataFrame:
    """Read daily XRS-B background (not the daily mean), retaining its UTC day."""
    import xarray as xr

    with xr.open_dataset(path, engine="h5netcdf") as data:
        if data.attrs.get("platform") != f"g{satellite}":
            raise ValueError(f"{path.name}: unexpected satellite platform")
        frame = data[["bkd1d_xrsb_flux", "bkd1d_xrsb_flag"]].to_dataframe().reset_index()
    frame = frame.loc[frame.bkd1d_xrsb_flag.eq(0) & np.isfinite(frame.bkd1d_xrsb_flux)].copy()
    frame = frame.rename(columns={"time": "goes_xray_background_timestamp", "bkd1d_xrsb_flux": "goes_xray_background"})
    frame["goes_xray_background_timestamp"] = pd.to_datetime(frame["goes_xray_background_timestamp"], utc=True)
    frame["goes_xray_background"] = frame["goes_xray_background"].astype(float)
    frame["goes_xray_satellite"] = satellite
    frame["goes_xray_file"] = path.name
    return frame.drop(columns="bkd1d_xrsb_flag").drop_duplicates("goes_xray_background_timestamp", keep="last")


def join_archive_background(euvs: pd.DataFrame, xrs: pd.DataFrame) -> pd.DataFrame:
    """Use the preceding UTC day's background, which requires a completed day.

    Archive time labels the START of the background's measurement day. Attaching
    that day's final background to morning EUVS would leak later measurements.
    No background is propagated across missing days. Science-product publication
    latency itself is not simulated by this historical observation reconstruction.
    """
    frame = euvs.copy()
    frame["goes_xray_background_timestamp"] = frame.timestamp.dt.floor("D") - pd.Timedelta(days=1)
    frame = frame.merge(xrs, on="goes_xray_background_timestamp", how="left", validate="many_to_one")
    frame["goes_xray_satellite"] = frame["goes_xray_satellite"].astype("Int16")
    frame["goes_xray_file"] = frame["goes_xray_file"].astype("string")
    return frame


def select_satellite_records(frame: pd.DataFrame, satellites: Sequence[int]) -> pd.DataFrame:
    """One row per minute: complete quality-valid records, then configured priority."""
    frame = frame.copy()
    frame["_usable"] = frame.goes_euvs_quality_valid & np.isfinite(frame.goes_xray_background)
    frame["_priority"] = frame.goes_euvs_satellite.map({sat: i for i, sat in enumerate(satellites)})
    return frame.sort_values(
        ["timestamp", "_usable", "_priority"], ascending=[True, False, True],
    ).drop_duplicates("timestamp").drop(columns=["_usable", "_priority"]).reset_index(drop=True)


def iter_goes_history(
    files: Sequence[GoesArchiveFile],
    days: pd.DatetimeIndex,
    satellites: Sequence[int],
    cache_dir: Path,
    session: requests.Session,
    *,
    refresh: bool = False,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Download and normalize one year at a time to bound memory use."""
    xrs = {}
    for source in files:
        if source.product == XRS_PRODUCT:
            path = cache_archive_file(source, cache_dir, session, refresh=refresh)
            xrs[source.satellite] = read_xrs_archive(path, source.satellite)
    for year in sorted(set(days.year)):
        frames = []
        for source in files:
            if source.product != EUVS_PRODUCT or source.year != year:
                continue
            if source.satellite not in xrs:
                raise ValueError(f"No XRS background archive for GOES-{source.satellite}")
            path = cache_archive_file(source, cache_dir, session, refresh=refresh)
            euvs = read_euvs_archive(path, source.satellite)
            euvs = euvs.loc[euvs.timestamp.dt.floor("D").isin(days) & euvs.timestamp.dt.year.eq(year)]
            frames.append(join_archive_background(euvs, xrs[source.satellite]))
        if frames:
            yield year, select_satellite_records(pd.concat(frames, ignore_index=True), satellites)
