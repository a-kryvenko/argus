import pandas as pd
import pytest

from clio.providers.goes_loader import (
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

    assert len(frame) == 2
    assert pd.isna(frame.loc[1, "goes_mgii_index"])
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2026-08-27T12:00:00Z")
    assert frame.loc[0, "goes_euvs_satellite"] == 19
    assert frame.loc[0, "goes_lya_1216"] == 5.0
    assert frame.loc[0, "goes_mgii_index"] == 8.0
    assert not any("quality" in column for column in frame)


def test_parse_euvs_json_keeps_usable_values_in_incomplete_response() -> None:
    payload = _euvs_observation("2026-08-27T12:00:00Z")[:-1]

    frame = _parse_euvs_json(payload)
    assert len(frame) == 1 and frame.goes_mgii_index.isna().all()
    assert frame.goes_euv_256.tolist() == [1.]


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
        "clio.providers.goes_loader._fetch_json",
        lambda _url, **kwargs: responses.pop(0),
    )

    frame = fetch_goes(
        start="2026-08-27T12:01:00Z",
        end="2026-08-27T12:02:00Z",
    )

    assert len(frame) == 1
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2026-08-27T12:01:00Z")
    assert frame.loc[0, "goes_xray_background"] == 6.2e-7
    assert frame.loc[0, "goes_xray_satellite"] == 18


def test_flagged_euvs_observations_are_rejected():
    payload = _euvs_observation('2026-08-27T12:00:00Z')
    payload[0]['flags']['eclipse'] = True
    frame = _parse_euvs_json(payload)
    assert frame.goes_euv_256.isna().all()
    assert frame.goes_euv_284.tolist() == [2.]
    assert not any('quality' in name or 'flag' in name for name in frame)


def test_flagged_revision_does_not_restore_older_valid_value():
    payload = _euvs_observation('2026-08-27T12:00:00Z')
    payload.append({**payload[0], 'flags': {'eclipse': True}})
    frame = _parse_euvs_json(payload)
    assert frame.goes_euv_256.isna().all()
    assert frame.goes_euv_284.tolist() == [2.]
