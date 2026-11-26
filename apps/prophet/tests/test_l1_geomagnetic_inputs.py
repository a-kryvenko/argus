from datetime import UTC, datetime, timedelta

import httpx
import pytest

from prophet.services.inputs.l1_geomagnetic import (
    WIND_PRODUCTS, read_l1_geomagnetic_indices, read_l1_wind_history,
)


AS_OF = datetime(2026, 11, 26, 12, 0, 20, tzinfo=UTC)
HOUR = timedelta(hours=1)
MINUTE = timedelta(minutes=1)


def series(basis, seconds, points):
    return dict(time_basis=basis, resolution_seconds=seconds, aggregation='source', points=points)


def point(at, value, *, receipt=AS_OF, **details):
    return dict(observed_at=at.isoformat(), received_at=receipt.isoformat(), value=value, **details)


def test_live_l1_solar_wind_uses_only_complete_causal_spacecraft_vectors():
    end = AS_OF.replace(second=0)
    usable = end - 2 * MINUTE
    late = end - MINUTE
    mixed = end - 3 * MINUTE
    archive = end - 4 * MINUTE
    flagged = end - 5 * MINUTE

    def response(request):
        assert request.url.path == '/internal/v1/observations/series'
        assert request.headers['Authorization'] == 'Bearer token'
        assert request.url.params['metrics'] == ','.join(WIND_PRODUCTS)
        assert request.url.params['interval'] == '1m'
        assert request.url.params['time_basis'] == 'l1'
        assert request.url.params['from'] == (end - 4 * HOUR).isoformat()
        assert request.url.params['to'] == end.isoformat()
        values = dict(v=481.5, n=4.5, bx=-2, by=2, bz=-6, bt=7)
        data = {}
        for metric, source in WIND_PRODUCTS.items():
            def sample(at, *, receipt=AS_OF, spacecraft='DSCOVR', product=source, quality=None):
                return point(at, values[metric], receipt=receipt, source_product=product,
                             spacecraft=spacecraft, quality=quality)
            data[metric] = series('l1', 60, [
                sample(usable, receipt=AS_OF if metric == 'bz' else end),
                sample(late, receipt=AS_OF + MINUTE if metric == 'bz' else AS_OF),
                sample(mixed, spacecraft='ACE' if metric == 'v' else 'DSCOVR'),
                sample(archive, product='omni.hourly' if metric == 'bx' else source),
                sample(flagged, quality='flagged' if metric == 'n' else None),
                sample(end),
            ])
        return httpx.Response(200, json={'success': True, 'data': data})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        wind = read_l1_wind_history(client, 'http://clio', 'token', AS_OF)

    assert len(wind) == 240 and wind.index.max() == late
    complete = wind.dropna()
    assert list(complete.index) == [usable]
    assert complete.loc[usable].to_dict() == dict(v=481.5, n=4.5, bx=-2., by=2., bz=-6., bt=7.,
                                                 received_at=AS_OF, spacecraft='dscovr',
                                                 plasma_speed=481.5, plasma_received_at=end)


def test_standalone_plasma_speed_keeps_its_own_receipt_clock():
    end = AS_OF.replace(second=0)
    earlier = end - 3 * MINUTE
    late = end - 2 * MINUTE
    archive = end - MINUTE
    receipt = end - MINUTE
    data = {metric: series('l1', 60, []) for metric in WIND_PRODUCTS}
    data['v']['points'] = [
        point(earlier, 460., receipt=receipt, spacecraft='DSCOVR', source_product=WIND_PRODUCTS['v']),
        point(late, 420., receipt=AS_OF + MINUTE, spacecraft='DSCOVR', source_product=WIND_PRODUCTS['v']),
        point(archive, 410., spacecraft='DSCOVR', source_product='omni.hourly'),
        point(end, 515., spacecraft='DSCOVR', source_product=WIND_PRODUCTS['v']),
    ]
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={'success': True, 'data': data}))
    with httpx.Client(transport=transport) as client:
        wind = read_l1_wind_history(client, 'http://clio', 'token', AS_OF)

    assert len(wind) == 240
    assert wind[list(WIND_PRODUCTS) + ['received_at', 'spacecraft']].isna().all(axis=None)
    assert wind.plasma_speed.dropna().to_dict() == {earlier: 460.}
    assert wind.plasma_received_at.dropna().to_dict() == {earlier: receipt}


def test_fast_geomagnetic_indices_use_native_clocks_and_both_receipt_delays():
    end = AS_OF.replace(second=0)
    kp_stamp = end - 6 * HOUR
    dst_stamp = end - 3 * HOUR

    def response(request):
        assert request.url.path == '/internal/v1/observations/series'
        assert request.url.params['metrics'] == 'kp,dst'
        assert request.url.params['to'] == AS_OF.isoformat()
        return httpx.Response(200, json={'success': True, 'data': {
            'kp': series('source_interval_start', 10800, [
                point(kp_stamp - 3 * HOUR, 4.67, receipt=AS_OF + MINUTE),
                point(kp_stamp, 3.67),
                point(kp_stamp + MINUTE, 5),
                point(kp_stamp + 3 * HOUR, 7),
            ]),
            'dst': series('source_interval_start', 3600, [
                point(dst_stamp - HOUR, -10.5, quality='flagged'),
                point(dst_stamp, -12.5),
                point(dst_stamp + MINUTE, -43),
                point(dst_stamp + HOUR, -54, receipt=AS_OF + MINUTE),
                point(dst_stamp + 2 * HOUR, -90),
            ]),
        }})

    with httpx.Client(transport=httpx.MockTransport(response)) as client:
        kp, dst = read_l1_geomagnetic_indices(client, 'http://clio', 'token', AS_OF)

    assert kp.to_dict() == {kp_stamp: 3.67}
    assert dst.to_dict() == {dst_stamp: -12.5}


def test_missing_observations_return_the_regular_empty_wind_clock():
    data = {metric: series('l1', 60, []) for metric in WIND_PRODUCTS}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={'success': True, 'data': data}))
    with httpx.Client(transport=transport) as client:
        wind = read_l1_wind_history(client, 'http://clio', 'token', AS_OF)
    assert len(wind) == 240
    assert wind.isna().all(axis=None)


def test_fast_wind_reader_rejects_incompatible_resolution():
    data = {metric: series('l1', 3600, []) for metric in WIND_PRODUCTS}
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={'success': True, 'data': data}))
    with httpx.Client(transport=transport) as client, pytest.raises(ValueError, match='Expected native Clio series'):
        read_l1_wind_history(client, 'http://clio', 'token', AS_OF)
