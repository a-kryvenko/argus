from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest

from clio.ingestion.collect_numeric import LIVE_LOADERS
from clio.ingestion.utils.adapters import historical_adapters
from clio.providers import goes_flux


NOW = datetime(2026, 10, 27, 8, 30, tzinfo=UTC)


def record(energy, flux, at=NOW, **extra):
    return {'time_tag': at.isoformat(), 'satellite': 18, 'energy': energy, 'flux': flux, **extra}


def test_xrs_uses_corrected_flux_and_clips_to_native_valid_data():
    frame = goes_flux.parse_records('goes.xrs', [
        record('0.05-0.4nm', 2e-8, observed_flux=4e-8, electron_contaminaton=True),
        record('0.1-0.8nm', 3e-7),
        record('0.05-0.4nm', -99999., NOW - timedelta(minutes=2)),
        record('0.1-0.8nm', None, NOW - timedelta(minutes=1)),
        record('0.1-0.8nm', 1.e-6, NOW + timedelta(minutes=1)),
    ], NOW)
    good = frame.loc[frame.quality.isna()]
    assert good.set_index('metric').value.to_dict() == {'xrs_a': 2.e-8, 'xrs_b': 3.e-7}
    assert good.observed_at.eq(NOW).all() and good.spacecraft.eq('GOES-18').all()
    assert frame.quality.dropna().tolist() == ['flagged', 'missing']


def test_proton_feeds_keep_only_10_and_100_mev_at_five_minute_slots():
    frame = goes_flux.parse_records('goes.protons', [
        record('>=10 MeV', .5), record('>=100 MeV', 0.), record('>=50 MeV', .2),
        record('>=10 MeV', 3., NOW - timedelta(minutes=1)),
        record('>=100 MeV', float('inf'), NOW - timedelta(minutes=5)),
        record('>=10 MeV', True, NOW - timedelta(minutes=5)),
    ], NOW)
    good = frame.loc[frame.quality.isna()]
    assert good.set_index('metric').value.to_dict() == {'proton_10mev': .5, 'proton_100mev': 0.}
    assert good.observed_at.eq(NOW).all()
    assert frame.quality.dropna().tolist() == ['flagged', 'flagged']


@pytest.mark.parametrize(('product', 'feed', 'energy', 'metric'), [
    ('goes.xrs', 'xrays', '0.1-0.8nm', 'xrs_b'),
    ('goes.protons', 'integral-protons', '>=10 MeV', 'proton_10mev'),
])
def test_live_and_backfill_use_their_own_feed_windows(monkeypatch, product, feed, energy, metric):
    first = NOW - timedelta(days=4)
    monkeypatch.setattr(goes_flux, 'datetime', Mock(wraps=datetime, now=Mock(return_value=NOW)))
    response = Mock()
    response.json.return_value = [record(energy, 1., first), record(energy, 2.)]
    get = Mock(return_value=response)
    monkeypatch.setattr(goes_flux.requests, 'get', get)
    live = LIVE_LOADERS[product](NOW - timedelta(hours=6), NOW)
    assert live[['metric', 'value', 'source_product']].to_dict('records') == [
        {'metric': metric, 'value': 2., 'source_product': product},
    ]
    get.assert_called_once_with(f'{goes_flux.BASE_URL}/{feed}-6-hour.json', timeout=(5, 25))

    get.reset_mock()
    adapter = historical_adapters()[product]
    assert adapter.fetch(first, first + timedelta(days=1)).value.tolist() == [1.]
    assert adapter.fetch(NOW, NOW + timedelta(minutes=1)).value.tolist() == [2.]
    get.assert_called_once_with(f'{goes_flux.BASE_URL}/{feed}-7-day.json', timeout=(5, 25))
