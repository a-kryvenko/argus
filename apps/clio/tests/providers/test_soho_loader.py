from io import BytesIO
from zipfile import ZipFile
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import requests

from clio.providers.soho_loader import SOHOLoader, parse_soho, TEMPERATURE_FACTOR


def row(minute, v=400, n=5, thermal=20, hour=9):
    return f'26 Aug 04 216:{hour:02}:{minute:02}:00 {v} {n} {thermal} 0 400 230 0 0 150 0 0 2315'


def test_temperature_converted_before_averaging_and_independent_validity():
    text = '\n'.join(row(i*5, n=-1, thermal=20 if i<3 else 40) for i in range(6))
    frame = parse_soho(text, 2026)
    assert frame.issue_time.iloc[0] == pd.Timestamp('2026-08-04T09:00Z')
    assert frame.v.iloc[0] == 400
    assert np.isnan(frame.n.iloc[0])
    assert frame.t.iloc[0] == pytest.approx(TEMPERATURE_FACTOR * 1000)


def test_sparse_hours_and_nonfinite_values_are_not_filled():
    text = '\n'.join([row(i*5, v='nan') for i in range(6)] + [row(0, hour=11)])
    frame = parse_soho(text, 2026)
    assert frame.v.isna().all()
    assert frame.n.iloc[0] == 5
    assert frame.iloc[1:][['v','n','t']].isna().all().all()


@pytest.mark.parametrize('text', [row(0)+'\n'+row(0), row(0).replace('216:', '217:'), '<html>error</html>'])
def test_bad_schema_and_duplicate_timestamps_fail(text):
    with pytest.raises(ValueError):
        parse_soho(text, 2026)


def test_year_download_reused_only_within_loader(monkeypatch):
    stream = BytesIO()
    with ZipFile(stream, 'w') as archive:
        archive.writestr('2026_CELIAS_Proton_Monitor_5min.txt', '\n'.join(row(i*5) for i in range(6)))
    response = Mock(status_code=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content = lambda *_: iter([stream.getvalue()])
    get = Mock(return_value=response)
    monkeypatch.setattr('clio.providers.soho_loader.requests.get', get)
    loader = SOHOLoader()
    start, end = pd.Timestamp('2026-08-04T09:00Z'), pd.Timestamp('2026-08-04T10:00Z')
    assert len(loader.load(start, end)) == 1
    assert len(loader.load(start, end)) == 1
    assert get.call_count == 1
    SOHOLoader().load(start, end)
    assert get.call_count == 2


def test_http_failure_not_treated_as_missing(monkeypatch):
    response = Mock(status_code=503)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.raise_for_status.side_effect = requests.HTTPError('503')
    monkeypatch.setattr('clio.providers.soho_loader.requests.get', Mock(return_value=response))
    with pytest.raises(requests.HTTPError):
        SOHOLoader()._year(2026)


def test_non_zip_response_is_reported_as_adapter_error(monkeypatch):
    response = Mock(status_code=200)
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.iter_content = lambda *_: iter([b'<html>temporary service error</html>'])
    monkeypatch.setattr('clio.providers.soho_loader.requests.get', Mock(return_value=response))
    with pytest.raises(ValueError, match='Invalid SOHO ZIP'):
        SOHOLoader()._year(2026)
