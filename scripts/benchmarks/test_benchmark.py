import numpy as np
import pandas as pd
import pytest

from scripts.benchmarks.benchmark import (
    observation_rows, fit_climatology, target_pairs, score_pairs, run,
)


def protocol():
    return dict(as_of='2026-01-04T00:00:00Z', issue_stride_hours=1,
                max_persistence_age_hours={'hourly': 2, 'daily': 72})


def test_climatology_never_uses_test_or_late_available_values():
    obs = observation_rows(pd.to_datetime(['2024-12-30', '2024-12-31', '2025-01-01'], utc=True),
                           [2, 4, 999], 'v', 'test', 24)
    climate = fit_climatology(obs, '2024-01-01T00:00:00Z', '2025-01-01T00:00:00Z', [3])
    assert climate['mean'] == 2
    assert climate['n'] == 1
    assert climate['probabilities']['3'] == 0


def test_persistence_causal_stale_and_missing_target():
    obs = observation_rows(pd.date_range('2025-12-31T23:00Z', periods=5, freq='h'),
                           [10, 20, np.nan, np.nan, 999], 'v', 'test', 0)
    fold = dict(test_start='2026-01-01T00:00:00Z', test_end='2026-01-02T00:00:00Z')
    pairs = target_pairs(obs, fold, protocol(), 1).set_index('issue_time')
    assert pairs.loc['2026-01-01T00:00Z', 'persistence'] == 10
    assert pairs.loc['2026-01-01T01:00Z', 'persistence'] == 20
    assert np.isnan(pairs.loc['2026-01-01T03:00Z', 'persistence'])
    assert np.isnan(pairs.loc['2026-01-01T00:00Z', 'value'])
    assert (pairs.available_at.dropna() <= pairs.available_at.dropna().index).all()


def test_daily_targets_not_replicated_and_end_boundary_respected():
    obs = observation_rows(pd.date_range('2025-12-29T00:00Z', periods=7, freq='D'),
                           np.arange(7), 's10', 'test', 24)
    fold = dict(test_start='2026-01-01T00:00:00Z', test_end='2026-01-04T00:00:00Z')
    pairs = target_pairs(obs, fold, protocol(), 1)
    assert len(pairs) == 2
    assert pairs.valid_time.dt.hour.eq(0).all()
    assert pairs.valid_time.is_unique
    assert pairs.issue_time.ge(pd.Timestamp(fold['test_start'])).all()


def test_matched_model_scores_and_threshold_baselines():
    pairs = pd.DataFrame(dict(value=[0., 10., 1000.], persistence=[0., 0., np.nan],
                              candidate=[0., 5., 1000.], valid_time=pd.date_range('2025-01-01', periods=3, tz='UTC')))
    climate = dict(mean=5., quantiles=[0., 5., 10.], probabilities={'5': .5})
    rows = score_pairs(pairs, climate, [5], model='candidate')
    assert {r['n'] for r in rows} == {2}
    assert rows[0]['mae'] == 5
    assert rows[1]['brier_ge_5'] == .25
    assert rows[2]['mae_skill_vs_persistence'] == .5
    assert rows[0]['coverage80'] == .5


def test_no_data_is_not_zero_error():
    pairs = pd.DataFrame(dict(value=[np.nan], persistence=[1.], valid_time=[pd.Timestamp('2025-01-01', tz='UTC')]))
    rows = score_pairs(pairs, None, [])
    assert all(r['mae'] is None and r['n'] == 0 for r in rows)


def test_report_cannot_overwrite_existing_directory(tmp_path):
    config = tmp_path/'protocol.json'
    config.write_text('{}')
    with pytest.raises(FileExistsError):
        run(tmp_path, tmp_path, config)


def test_control_fold_has_identical_test_pairs():
    obs = observation_rows(pd.date_range('2025-12-30T00:00Z', periods=120, freq='h'),
                           np.arange(120), 'v', 'test', 0)
    fold = dict(test_start='2026-01-01T00:00:00Z', test_end='2026-01-04T00:00:00Z')
    a = target_pairs(obs, dict(fold, train_end='2025-01-01T00:00:00Z'), protocol(), 24)
    b = target_pairs(obs, dict(fold, train_end='2026-01-01T00:00:00Z'), protocol(), 24)
    pd.testing.assert_frame_equal(a, b)


def test_candidate_rejects_test_calibration_and_duplicate_keys():
    from scripts.benchmarks.compare import validate
    p = dict(id='test', train_start='2011-01-01T00:00:00Z', max_lead_hours=96,
             folds={'a': {'train_end': '2025-01-01T00:00:00Z'}})
    metadata = dict(model_sha256='a'*64, train_start=p['train_start'],
                    fit_end='2025-01-01T00:00:00Z', selection_end='2024-01-01T00:00:00Z',
                    calibration_end='2025-02-01T00:00:00Z', upstream_training='none',
                    prior_test_exposure='none', target_definition='test')
    frame = pd.DataFrame(dict(target=['v'], issue_time=['2025-01-01T00:00:00Z'],
                              valid_time=['2025-01-02T00:00:00Z'], lead_hours=[24], prediction=[400.]))
    with pytest.raises(ValueError, match='overlaps'):
        validate(frame, metadata, p, 'a')
    metadata['calibration_end'] = metadata['fit_end']
    assert len(validate(frame, metadata, p, 'a')) == 1
    with pytest.raises(ValueError, match='duplicate'):
        validate(pd.concat([frame, frame]), metadata, p, 'a')


def test_exported_candidate_comparison_and_tamper_detection(tmp_path):
    from scripts.benchmarks.benchmark import digest, write_json
    from scripts.benchmarks.compare import compare
    base = tmp_path/'base'
    pair_dir = base/'pairs/a/v'
    pair_dir.mkdir(parents=True)
    p = dict(id='test', train_start='2011-01-01T00:00:00Z', max_lead_hours=96,
             folds={'a': {'train_end': '2025-01-01T00:00:00Z'}}, thresholds={})
    write_json(base/'protocol.json', p)
    write_json(base/'climatology.json', {'a': {'v': dict(mean=5., quantiles=[0., 5., 10.], probabilities={})}})
    pairs = pd.DataFrame(dict(issue_time=pd.to_datetime(['2025-01-01T00:00Z', '2025-01-02T00:00Z']),
        valid_time=pd.to_datetime(['2025-01-02T00:00Z', '2025-01-03T00:00Z']), lead_hours=[24,24],
        value=[0., 10.], persistence=[0., 0.]))
    pairs.to_parquet(pair_dir/'24.parquet', index=False)
    write_json(base/'manifest.json', {'artifacts': {str(f.relative_to(base)): digest(f)
        for f in base.rglob('*') if f.is_file()}})
    pred = pairs[['issue_time', 'valid_time', 'lead_hours']].assign(target='v', prediction=[0., np.nan])
    pred.to_parquet(tmp_path/'pred.parquet', index=False)
    write_json(tmp_path/'meta.json', dict(model_sha256='a'*64, train_start=p['train_start'],
        fit_end='2025-01-01T00:00:00Z', selection_end='2024-01-01T00:00:00Z',
        calibration_end='2025-01-01T00:00:00Z', upstream_training='none',
        prior_test_exposure='none', target_definition='test'))
    compare(base, tmp_path/'pred.parquet', tmp_path/'meta.json', 'a', tmp_path/'report')
    metrics = pd.read_csv(tmp_path/'report/metrics.csv')
    assert set(metrics.n) == {1}
    assert set(metrics.n_expected) == {2}
    assert set(metrics.baseline) == {'persistence', 'climatology', 'prediction'}
    assert metrics.loc[metrics.baseline.eq('prediction'), 'mae_skill_vs_persistence'].isna().all()
    write_json(base/'climatology.json', {})
    with pytest.raises(ValueError, match='Changed benchmark'):
        compare(base, tmp_path/'pred.parquet', tmp_path/'meta.json', 'a', tmp_path/'changed')
