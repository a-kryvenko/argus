"""Unpropagated, one-minute NOAA RTSW observations at L1.

Source records and provider quality flags are retained without interpolation.
https://www.spaceweather.gov/products/solar-wind
"""
from datetime import UTC, datetime
import math

import requests

SOURCES = {
    "mag": "https://services.swpc.noaa.gov/json/rtsw/rtsw_mag_1m.json",
    "plasma": "https://services.swpc.noaa.gov/json/rtsw/rtsw_wind_1m.json",
}
FIELDS = {
    "mag": {"bx": "bx_gsm", "by": "by_gsm", "bz": "bz_gsm", "bt": "bt"},
    "plasma": {"v": "proton_speed", "n": "proton_density", "t": "proton_temperature"},
}


def parse_records(kind: str, payload: object, received_at: datetime) -> list[dict]:
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
        values = {}
        for metric, field in FIELDS[kind].items():
            try:
                value = float(row[field]) if row[field] is not None else None
            except (ValueError, TypeError):
                value = None
            if value is not None and (not math.isfinite(value) or value <= -9999 or (metric in {"bt", "v", "n", "t"} and value < 0)):
                value = None
            values[metric] = value
        record = {
            "kind": kind, "observed_at": observed_at, "spacecraft": row["source"],
            "active": row["active"], "received_at": received_at,
            "values": values, "raw": row,
        }
        records[(observed_at, row["source"])] = record
    if not records:
        raise ValueError(f"NOAA {kind}: no usable timestamps")
    return list(records.values())


def fetch_records(kind: str) -> list[dict]:
    response = requests.get(SOURCES[kind], timeout=(5, 25))
    response.raise_for_status()
    return parse_records(kind, response.json(), datetime.now(UTC))
