import pandas as pd
import pytest

from clio.dataloaders.goes_loader import (
    EUVS_LINE_COLUMNS,
    _parse_euvs_json,
    _parse_xray_background_json,
    fetch_goes,
)


def _euvs_observation(time_tag: str, satellite: int = 19):
    return [
        {
            "time_tag": time_tag,
            "satellite": satellite,
            "line": line,
            "value": float(index),
            "au_factor": 1.02,
            "flags": {
                "eclipse": False,
                "lunar_transit": False,
                "geocorona": False,
            },
        }
        for index, line in enumerate(EUVS_LINE_COLUMNS, start=1)
    ]


def test_parse_euvs_json_returns_complete_wide_records() -> None:
    payload = _euvs_observation("2026-08-27T12:00:00Z")
    payload.extend(_euvs_observation("2026-08-27T12:01:00Z")[:-1])

    frame = _parse_euvs_json(payload)

    assert len(frame) == 1
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2026-08-27T12:00:00Z")
    assert frame.loc[0, "goes_euvs_satellite"] == 19
    assert frame.loc[0, "goes_lya_1216"] == 5.0
    assert frame.loc[0, "goes_mgii_index"] == 8.0
    assert bool(frame.loc[0, "goes_euvs_quality_valid"]) is True


def test_parse_euvs_json_rejects_incomplete_response() -> None:
    payload = _euvs_observation("2026-08-27T12:00:00Z")[:-1]

    with pytest.raises(RuntimeError, match="no complete observation"):
        _parse_euvs_json(payload)


def test_parse_xray_background_json_sorts_and_deduplicates() -> None:
    payload = [
        {
            "time_tag": "2026-08-27T00:00:00Z",
            "satellite": 18,
            "background": 6.2e-7,
        },
        {
            "time_tag": "2026-08-26T00:00:00Z",
            "satellite": 18,
            "background": 7.1e-7,
        },
    ]

    frame = _parse_xray_background_json(payload)

    assert list(frame["goes_xray_background"]) == [7.1e-7, 6.2e-7]


def test_fetch_goes_merges_background_and_filters(monkeypatch) -> None:
    euvs_payload = []
    euvs_payload.extend(_euvs_observation("2026-08-27T12:00:00Z"))
    euvs_payload.extend(_euvs_observation("2026-08-27T12:01:00Z"))
    responses = [
        euvs_payload,
        [
            {
                "time_tag": "2026-08-27T00:00:00Z",
                "satellite": 18,
                "background": 6.2e-7,
            }
        ],
    ]
    monkeypatch.setattr(
        "clio.dataloaders.goes_loader._fetch_json",
        lambda _url: responses.pop(0),
    )

    frame = fetch_goes(
        start="2026-08-27T12:01:00Z",
        end="2026-08-27T12:01:00Z",
    )

    assert len(frame) == 1
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2026-08-27T12:01:00Z")
    assert frame.loc[0, "goes_xray_background"] == 6.2e-7
    assert frame.loc[0, "goes_xray_satellite"] == 18
