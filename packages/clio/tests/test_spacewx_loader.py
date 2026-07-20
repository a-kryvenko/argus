import pandas as pd
import pytest

from clio.dataloaders.spacewx_loader import _parse_spacewx_text


SAMPLE = """# F10, S10, M10, Y10 test data
2024 001 2460310.0 150.0 145.0 140.0 135.0 130.0 125.0 120.0 115.0 1
2024 002 2460311.0 151.0 146.0 141.0 136.0 131.0 126.0 121.0 116.0 1
"""


def test_parse_spacewx_text() -> None:
    frame = _parse_spacewx_text(SAMPLE)

    assert list(frame["f10"]) == [150.0, 151.0]
    assert list(frame["s10_81c"]) == [135.0, 136.0]
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2024-01-01 12:00:00+00:00")
    assert frame.loc[1, "day_of_year"] == 2


def test_parse_spacewx_text_rejects_empty_response() -> None:
    with pytest.raises(RuntimeError, match="No SOLFSMY records parsed"):
        _parse_spacewx_text("# header only")
