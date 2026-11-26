"""Unpropagated, usable one-minute NOAA RTSW observations at L1.
https://www.spaceweather.gov/products/solar-wind
"""
from datetime import UTC, datetime

import pandas as pd
import requests

from clio.providers._contract import sample, samples_frame

SOURCES = {
    "mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    "plasma": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
}
FIELDS = {
    "mag": {"bx": "bx_gsm", "by": "by_gsm", "bz": "bz_gsm", "bt": "bt"},
    "plasma": {"v": "proton_speed", "n": "proton_density", "t": "proton_temperature"},
}


def parse_records(kind: str, payload: object, received_at: datetime) -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"NOAA {kind}: expected a nonempty record list")
    records = {}
    for row in payload:
        if not isinstance(row, dict) or not {"time_tag", "source", "active", *FIELDS[kind].values()} <= row.keys():
            raise ValueError(f"NOAA {kind}: unexpected record schema")
        try:
            observed_at = datetime.fromisoformat(row["time_tag"].replace("Z", "+00:00"))
            observed_at = observed_at.replace(tzinfo=UTC) if observed_at.tzinfo is None else observed_at.astimezone(UTC)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"NOAA {kind}: invalid time_tag") from exc
        if not isinstance(row["active"], bool) or not isinstance(row["source"], str) or not row["source"]:
            raise ValueError(f"NOAA {kind}: invalid source selection")
        if observed_at > received_at:
            continue
        # Last revision wins per spacecraft, including revisions that invalidate it.
        records[(observed_at, row['source'])] = row
    selected = {}
    for (timestamp, source), row in records.items():
        if not row['active']:
            continue
        flagged = row.get('overall_quality') not in (None, 0)
        if timestamp not in selected or selected[timestamp][0] and not flagged:
            selected[timestamp] = (flagged, source, row)
    result = []
    for timestamp, (flagged, spacecraft, row) in selected.items():
        for metric, field in FIELDS[kind].items():
            try:
                fill = float(row[field]) <= -9999
            except (TypeError, ValueError, OverflowError):
                fill = False
            result.append(sample(metric, timestamp, row[field], flagged=flagged or fill, spacecraft=spacecraft))
    return samples_frame(result)


def fetch_records(kind: str) -> pd.DataFrame:
    response = requests.get(SOURCES[kind], timeout=(5, 25))
    response.raise_for_status()
    return parse_records(kind, response.json(), datetime.now(UTC))
