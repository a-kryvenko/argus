import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from clio.config import ClioObservations
from clio.ingestion.adapters import PlasmaHistoryAdapter, WideHistoryAdapter, historical_adapters
from clio.ingestion.products import PRODUCTS
from clio.observations import backfill as service
from clio.observations.store import upsert_measurements
from clio.providers.spdf_loader import SPDF_Loader

START = datetime(2026, 8, 31, tzinfo=UTC)
END = START + timedelta(hours=3)


def config(metrics=('v', 'n', 't')):
    def sources(metric):
        live = [name for name, product in PRODUCTS.items() if metric in product.metrics and 'live' in product.modes]
        historical = (['gfz.f107', 'omni.hourly'] if metric == 'f10_7' else
                      ['omni.hourly', 'ace.plasma_hourly'] if metric in ('v', 'n', 't') else ['omni.hourly'])
        return {'live': live, 'historical': historical}
    return ClioObservations.model_validate({'observations': {
        metric: {'sources': sources(metric),
                 'schedules': {'live': {'every': '1h'}, 'backfill': {'every': '6h'}},
                 'backfill': {'days': 2}}
        for metric in metrics}})


def frame(*rows):
    return pd.DataFrame(rows, columns=['metric', 'value', 'observed_at'])


@pytest.mark.parametrize('change', [
    lambda p: p['sources'].update(historical=['typo']),
    lambda p: p['sources'].update(historical=['swpc.rtsw_plasma']),
    lambda p: p['sources'].update(historical=['omni.hourly', 'omni.hourly']),
    lambda p: p['sources'].update(historical=[]),
    lambda p: p['schedules']['live'].update(every='0s'),
    lambda p: p['schedules']['live'].update(every='whenever'),
    lambda p: p['backfill'].update(days=0),
    lambda p: p['backfill'].update(days=True),
    lambda p: p.update(unknown=True),
])
def test_invalid_policy_rejected(change):
    policy = {'sources': {'live': ['swpc.rtsw_plasma'], 'historical': ['omni.hourly']},
              'schedules': {'live': {'every': '60s'}, 'backfill': {'every': '1d'}},
              'backfill': {'days': 60}}
    change(policy)
    with pytest.raises(ValidationError):
        ClioObservations.model_validate({'observations': {'v': policy}})


def test_fallback_is_per_metric_and_hour_and_requests_are_batched():
    policies = config().observations
    pending = {metric: set(pd.date_range(START, END, freq='h', inclusive='left')) for metric in policies}
    primary = Mock()
    primary.fetch.return_value = frame(('v', 400., START), ('v', float('nan'), START + timedelta(hours=1)),
                                       ('n', 5., START), ('t', -1e31, START))
    fallback = Mock()
    fallback.fetch.return_value = frame(*((metric, 999., hour) for metric, hours in pending.items() for hour in hours))
    result, attempts = service.fetch_missing(pending, policies, {'omni.hourly': primary, 'ace.plasma_hourly': fallback})
    assert len(result) == 9
    assert result[(result.metric == 'v') & (result.observed_at == START)].iloc[0].value == 400.
    assert result[(result.metric == 't') & (result.observed_at == START)].iloc[0].source_product == 'ace.plasma_hourly'
    primary.fetch.assert_called_once()
    fallback.fetch.assert_called_once()
    assert attempts[0]['accepted_measurements'] == 2
    assert attempts[1]['accepted_measurements'] == 7
    assert result.received_at.min() > END


def test_complete_primary_skips_fallback_and_clips_extra_values():
    primary, fallback = Mock(), Mock()
    primary.fetch.return_value = frame(('v', 1., START - timedelta(hours=1)), ('v', 2., START), ('n', 3., START))
    result, _ = service.fetch_missing({'v': {pd.Timestamp(START)}}, config(('v',)).observations,
                                     {'omni.hourly': primary, 'ace.plasma_hourly': fallback})
    assert result.value.tolist() == [2.]
    fallback.fetch.assert_not_called()


def test_failure_uses_fallback_and_reports_failure():
    primary, fallback = Mock(), Mock()
    primary.fetch.side_effect = OSError('offline')
    fallback.fetch.return_value = frame(('v', 300., START))
    result, attempts = service.fetch_missing({'v': {pd.Timestamp(START)}}, config(('v',)).observations,
                                            {'omni.hourly': primary, 'ace.plasma_hourly': fallback})
    assert result.value.tolist() == [300.]
    assert attempts[0]['error'] == 'offline'


def test_calendar_requests_are_bounded_and_skip_complete_days():
    hours = pd.date_range(START, periods=33 * 24, freq='h')
    ranges = list(service.request_ranges(hours))
    assert ranges == [(START, START + timedelta(days=31)),
                      (START + timedelta(days=31), START + timedelta(days=33))]
    assert list(service.request_ranges([pd.Timestamp(START), pd.Timestamp(START + timedelta(days=2))])) == [
        (START, START + timedelta(days=1)), (START + timedelta(days=2), START + timedelta(days=3))]


def test_adapter_clips_provider_days_and_rejects_invalid_plasma():
    loader = Mock(return_value=pd.DataFrame({'issue_time': [START, END, START + timedelta(hours=1)],
                                           'v': [400., 500., -1e31], 'by': [1., 2., 3.]}))
    result = PlasmaHistoryAdapter(loader).fetch(START, END)
    assert result.metric.tolist() == ['v']
    assert result.value.tolist() == [400.]


def test_ace_plasma_does_not_request_magnetic_and_skips_fill_values(monkeypatch):
    load = Mock(return_value=object())
    monkeypatch.setattr(SPDF_Loader, '_load_cdf_from_url', load)
    monkeypatch.setattr(SPDF_Loader, '_swepam_dataframe', lambda _: pd.DataFrame({
        'issue_time': [START, START + timedelta(minutes=5)],
        'v': [-1e31, 410.], 'n': [5., 6.], 't': [100., 200.]}))
    result = SPDF_Loader.load_plasma(START, END)
    assert result.v.tolist() == [410.]
    assert result.issue_time.tolist() == [START]
    assert '/swepam/' in load.call_args.args[0]
    load.assert_called_once()


def test_complete_history_is_noop_without_provider_requests(monkeypatch):
    cfg = config(('v',))
    stored = frame(*(('v', 400., ts) for ts in pd.date_range(START, periods=48, freq='h')))
    monkeypatch.setattr(service, 'load_measurements', AsyncMock(return_value=stored))
    adapters = {'omni.hourly': Mock(), 'ace.plasma_hourly': Mock()}
    monkeypatch.setattr(service, 'historical_adapters', lambda: adapters)
    session = AsyncMock()
    result = asyncio.run(service.backfill_selected(session, cfg, ['v'], now=START + timedelta(days=2)))
    assert result['status'] == 'complete'
    assert result['downloaded_measurements'] == 0
    for adapter in adapters.values():
        adapter.fetch.assert_not_called()
    session.execute.assert_not_called()


def test_backfill_preserves_concurrent_values_and_uses_metric_depth(monkeypatch):
    cfg = config(('v', 'n'))
    cfg.observations['n'].backfill.days = 1
    reads = AsyncMock(side_effect=[frame(), frame(('v', 450., START))])
    monkeypatch.setattr(service, 'load_measurements', reads)
    fetch = Mock(return_value=(frame(('v', 400., START)), []))
    monkeypatch.setattr(service, 'fetch_missing', fetch)
    insert = AsyncMock()
    monkeypatch.setattr(service, 'upsert_measurements', insert)
    from clio.observations import derived
    normalize = Mock(return_value=pd.DataFrame())
    monkeypatch.setattr(derived, 'normalize_measurements', normalize)
    monkeypatch.setattr(service, 'upsert_normalized_observations', AsyncMock())
    session = AsyncMock()
    result = asyncio.run(service.backfill_selected(session, cfg, ['v', 'n'], now=START + timedelta(days=2)))
    pending = fetch.call_args.args[0]
    assert len(pending['v']) == 48 and len(pending['n']) == 24
    assert normalize.call_args.args[0].value.tolist() == [450.]
    assert insert.call_args.kwargs == {'replace_existing': False}
    assert result['missing_observed_hours'] == {'v': 47, 'n': 24}
    session.commit.assert_awaited_once()


def test_provenance_and_insert_only_conflict_do_not_update_existing_rows():
    session = AsyncMock()
    data = frame(('v', 400., START)).assign(source_product='omni.hourly', received_at=END)
    asyncio.run(upsert_measurements(session, data, replace_existing=False))
    compiled = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert 'DO NOTHING' in str(compiled)
    assert compiled.params['source_product_m0'] == 'omni.hourly'
    assert compiled.params['received_at_m0'] == END


def test_legacy_repoll_does_not_erase_provenance_for_unchanged_values():
    session = AsyncMock()
    asyncio.run(upsert_measurements(session, frame(('v', 400., START))))
    sql = str(session.execute.call_args.args[0].compile(dialect=postgresql.dialect()))
    assert sql.rsplit(' WHERE ', 1)[-1].endswith('measurement.value IS DISTINCT FROM excluded.value')


def test_existing_observation_closes_hour_even_when_flagged_or_off_grid():
    stored = frame(('v', float('nan'), START + timedelta(minutes=5)))
    assert service.missing_hours(stored, 'v', START, START + timedelta(hours=1)) == set()


def test_all_sources_failing_does_not_commit(monkeypatch):
    monkeypatch.setattr(service, 'load_measurements', AsyncMock(return_value=frame()))
    adapters = {'omni.hourly': Mock(), 'ace.plasma_hourly': Mock()}
    for adapter in adapters.values():
        adapter.fetch.side_effect = OSError('offline')
    monkeypatch.setattr(service, 'historical_adapters', lambda: adapters)
    session = AsyncMock()
    with pytest.raises(RuntimeError, match='No gaps filled'):
        asyncio.run(service.backfill_selected(session, config(('v',)), ['v'], start=START, end=END))
    session.commit.assert_not_awaited()
    session.execute.assert_not_called()


def test_coverage_uses_three_hour_and_daily_slots_and_only_closed_intervals():
    end = START + timedelta(days=1, hours=2)
    stored = frame(('kp', 3., START), ('kp', 2., START + timedelta(hours=3)),
                   ('f10_7', 150., START + timedelta(hours=12)))
    assert len(service.missing_slots(stored, 'kp', START, end)) == 6
    assert not service.missing_slots(stored, 'f10_7', START, end)
    assert len(service.missing_slots(stored, 'ap', START, end)) == 8
    assert not service.missing_slots(stored, 'f10_7', START + timedelta(hours=1), end)


def test_omni_adapter_retains_signed_fields_and_filters_invalid_indices():
    loader = Mock(return_value=pd.DataFrame({
        'issue_time': [START, START + timedelta(hours=1)],
        'bx': [-5., 0.], 'by': [-2., 1.], 'bz': [-3., float('inf')],
        'dst': [-80., -20.], 'kp': [9., 10.], 'ap': [0., -1.]}))
    result = WideHistoryAdapter(loader, PRODUCTS['omni.hourly'].metrics).fetch(START, END)
    assert result[result.metric == 'dst'].value.tolist() == [-80., -20.]
    assert result[result.metric == 'bz'].value.tolist() == [-3.]
    assert result[result.metric == 'kp'].value.tolist() == [9.]
    assert result[result.metric == 'ap'].value.tolist() == [0.]


def test_daily_fallback_preserves_original_timestamp_and_never_overwrites_primary():
    day2 = START + timedelta(days=1)
    gfz, omni = Mock(), Mock()
    gfz.fetch.return_value = frame(('f10_7', 150., START + timedelta(hours=12)))
    omni.fetch.return_value = frame(('f10_7', 199., START), ('f10_7', 160., day2),
                                   ('f10_7', 161., day2 + timedelta(hours=1)))
    result, _ = service.fetch_missing({'f10_7': {pd.Timestamp(START), pd.Timestamp(day2)}},
                                     config(('f10_7',)).observations, {'gfz.f107': gfz, 'omni.hourly': omni})
    assert result.value.tolist() == [150., 160.]
    assert result.observed_at.tolist() == [START + timedelta(hours=12), day2]
    assert omni.fetch.call_args.args == (day2, day2 + timedelta(days=1))


def test_three_hour_coverage_stores_one_original_per_missing_slot():
    omni = Mock()
    omni.fetch.return_value = frame(*(('kp', 3.33, START + timedelta(hours=h)) for h in range(6)))
    result, _ = service.fetch_missing({'kp': {pd.Timestamp(START), pd.Timestamp(START + timedelta(hours=3))}},
                                     config(('kp',)).observations, {'omni.hourly': omni})
    assert result.value.tolist() == [3.33, 3.33]
    assert result.observed_at.tolist() == [START, START + timedelta(hours=3)]


def test_numeric_ingestion_does_not_register_calibration_products():
    from clio.ingestion.live_adapters import live_adapters
    assert not any('calibrated' in name for name in historical_adapters())
    assert not any('calibrated' in name for name in live_adapters())


def test_partial_explicit_day_does_not_fetch_daily_observation(monkeypatch):
    monkeypatch.setattr(service, 'load_measurements', AsyncMock(return_value=frame()))
    adapters = {'gfz.f107': Mock(), 'omni.hourly': Mock()}
    monkeypatch.setattr(service, 'historical_adapters', lambda: adapters)
    result = asyncio.run(service.backfill_selected(AsyncMock(), config(('f10_7',)), ['f10_7'], start=START, end=END))
    assert result['status'] == 'complete'
    for adapter in adapters.values():
        adapter.fetch.assert_not_called()


def test_magnetic_config_cannot_use_ace_gse_as_gsm_fallback():
    policy = {'sources': {'live': ['swpc.rtsw_mag'], 'historical': ['ace.plasma_hourly']},
              'schedules': {'live': {'every': '1h'}, 'backfill': {'every': '1d'}}, 'backfill': {'days': 60}}
    with pytest.raises(ValidationError, match='incompatible'):
        ClioObservations.model_validate({'observations': {'bz': policy}})
