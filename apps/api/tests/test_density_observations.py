from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.dialects import postgresql

from app.services.density_observations import SOLAR_LAGS_DAYS, observed_driver_frame, load_density_drivers
from app.services.forecast_products import ArtifactNotReadyError
from forecast.inference.jb2008_drivers import causal_dtc, quiet_dtc, DriverDataUnavailable

ISSUE = datetime(2026, 9, 6, 12, tzinfo=UTC)


def observations():
    rows = [dict(metric=name, value=100., observed_at=day)
            for name in SOLAR_LAGS_DAYS
            for day in pd.date_range(ISSUE-timedelta(days=87), ISSUE, freq='D')]
    rows += [dict(metric=name, value=value, observed_at=hour)
             for name, value in [('dst', -10.), ('ap', 5.)]
             for hour in pd.date_range(ISSUE-timedelta(days=8), ISSUE, freq='h')]
    return pd.DataFrame(rows)


def test_internal_snapshot_persistence_uses_only_raw_indices():
    records = observations()
    # Values after the lagged cutoff and issue time cannot affect the snapshot.
    records = pd.concat([records, pd.DataFrame([
        dict(metric='s10', value=900., observed_at=ISSUE),
        dict(metric='dst', value=-900., observed_at=ISSUE + timedelta(hours=1)),
        dict(metric='dtc', value=900., observed_at=ISSUE),
        dict(metric='s10_81c', value=900., observed_at=ISSUE),
    ])], ignore_index=True)
    result = observed_driver_frame(records, ISSUE)
    assert len(result) == 49
    assert result.s10.eq(100).all()
    assert result.s10_81mean.eq(100).all()
    assert result.dtc.iloc[0] == pytest.approx(37.9679953964)
    assert result.dtc.nunique() == 1
    assert result.valid_time.iloc[-1] == ISSUE + timedelta(hours=48)
    assert result.observed_at.eq(ISSUE - timedelta(days=5)).all()
    assert result.background_method.eq('trailing_81_daily_values').all()
    assert result.dtc_method.eq('causal_dst_ap_v1').all()


def test_daily_average_has_equal_day_weights_and_respects_lag():
    records = observations()
    cutoff = ISSUE - timedelta(days=1)
    mask = records.metric.eq('s10') & records.observed_at.between(cutoff-timedelta(days=80), cutoff)
    records.loc[mask, 'value'] = np.arange(1, 82)
    # Extra early sample on the final day must not overweight that day.
    records = pd.concat([records, pd.DataFrame([
        dict(metric='s10', value=1000., observed_at=cutoff-timedelta(hours=1))
    ])], ignore_index=True)
    result = observed_driver_frame(records, ISSUE)
    assert result.s10.iloc[0] == 81
    assert result.s10_81mean.iloc[0] == 41


@pytest.mark.parametrize('metric', ['dst', 'ap', 'f10_7', 's10', 'm10', 'y10'])
def test_missing_inputs_are_rejected(metric):
    records = observations()
    with pytest.raises(ArtifactNotReadyError):
        observed_driver_frame(records.loc[records.metric != metric], ISSUE)


def test_short_or_gapped_solar_history_is_not_called_81_day_mean():
    records = observations()
    for filtered in [records.loc[records.observed_at >= ISSUE-timedelta(days=30)],
                     records.loc[~(records.metric.eq('s10') & records.observed_at.between(ISSUE-timedelta(days=42), ISSUE-timedelta(days=40)))]]:
        with pytest.raises(ArtifactNotReadyError, match='81 consecutive'):
            observed_driver_frame(filtered, ISSUE)


def test_stale_and_invalid_observations_rejected():
    records = observations()
    records.loc[records.metric.eq('dst') & records.observed_at.eq(ISSUE), 'value'] = float('nan')
    with pytest.raises(ArtifactNotReadyError, match='72 consecutive'):
        observed_driver_frame(records, ISSUE)
    records = observations()
    records = records.loc[~(records.metric.eq('dst') & records.observed_at.gt(ISSUE-timedelta(hours=4)))]
    with pytest.raises(ArtifactNotReadyError, match='recent'):
        observed_driver_frame(records, ISSUE)


def test_quiet_dtc_reference_formula_and_cap():
    assert quiet_dtc(0) == 0
    assert quiet_dtc(5) == pytest.approx(37.9679953964)
    assert quiet_dtc(50) == pytest.approx(148.1684361111)
    assert quiet_dtc(100) == quiet_dtc(50)
    with pytest.raises(DriverDataUnavailable):
        quiet_dtc(-1)


def test_dtc_storm_heats_then_recovers_and_is_causal():
    times = pd.date_range('2026-09-01', periods=130, freq='h', tz='UTC')
    dst = pd.Series([-10.] * 80 + [-40., -80., -120., -200., -360., -450.]
                    + list(np.linspace(-440, -10, 44)), index=times)
    ap = pd.Series(5., index=times)
    result = causal_dtc(dst, ap)
    assert result.iloc[85] > 200
    assert result.iloc[-1] == pytest.approx(quiet_dtc(5))
    assert (result.dropna() >= 0).all()
    for end in (82, 85, 90, 110):
        assert causal_dtc(dst.iloc[:end], ap.iloc[:end]).iloc[-1] == result.iloc[end-1]


def test_dtc_requires_quiet_initialization():
    times = pd.date_range('2026-09-01', periods=80, freq='h', tz='UTC')
    with pytest.raises(DriverDataUnavailable, match='initialization'):
        causal_dtc(pd.Series(-200., index=times), pd.Series(5., index=times))


def test_ap_lag_excludes_recent_spike():
    records = observations()
    records.loc[records.metric.eq('ap') & records.observed_at.ge(ISSUE-timedelta(hours=6)), 'value'] = 100
    result = observed_driver_frame(records, ISSUE)
    assert result.dtc.iloc[0] == pytest.approx(quiet_dtc(5))


def test_database_loader_only_reads_source_metrics_and_long_history():
    import asyncio
    from unittest.mock import Mock
    session = AsyncMock()
    session.execute.return_value = Mock(all=lambda: list(observations().itertuples(index=False, name=None)))
    # DataFrame fixture column order is metric/value/observed_at, matching SELECT.
    result = asyncio.run(load_density_drivers(session, ISSUE))
    assert len(result) == 49
    query = session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert set(query.params['metric_1']) == {'f10_7', 's10', 'm10', 'y10', 'dst', 'ap'}
    assert query.params['observed_at_1'] == ISSUE - timedelta(days=88)
    session.commit.assert_not_called()
    session.add.assert_not_called()


def test_two_missing_days_are_filled_only_for_background():
    import json
    records = observations()
    cutoff = ISSUE - timedelta(days=1)
    selected = records.metric.eq('s10') & records.observed_at.between(cutoff-timedelta(days=80), cutoff)
    records.loc[selected, 'value'] = np.arange(1, 82)
    missing = [ISSUE-timedelta(days=3), ISSUE-timedelta(days=2)]
    records = records.loc[~(records.metric.eq('s10') & records.observed_at.isin(missing))]
    before = records.copy(deep=True)
    result = observed_driver_frame(records, ISSUE)
    assert result.s10.iloc[0] == 81
    assert result.s10_81mean.iloc[0] == 41
    assert result.background_method.eq('trailing_81_daily_values_linear_gapfill').all()
    assert json.loads(result.background_interpolated_days.iloc[0]) == {'s10': ['2026-09-03', '2026-09-04']}
    pd.testing.assert_frame_equal(records, before)


@pytest.mark.parametrize('positions', [[0], [80], [1, 2, 3], [1, 3, 5]])
def test_gap_fill_rejects_edges_and_more_than_two_days(positions):
    from forecast.inference.jb2008_drivers import fill_background_gaps
    daily = pd.Series(100., index=pd.date_range('2026-01-01', periods=81, tz='UTC'))
    daily.iloc[positions] = np.nan
    filled, dates = fill_background_gaps(daily)
    assert dates == []
    pd.testing.assert_series_equal(filled, daily)


def test_invalid_observed_day_is_not_replaced():
    records = observations()
    records.loc[records.metric.eq('s10') & records.observed_at.eq(ISSUE-timedelta(days=2)), 'value'] = np.nan
    with pytest.raises(ArtifactNotReadyError, match='81 consecutive'):
        observed_driver_frame(records, ISSUE)
