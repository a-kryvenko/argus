from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/training/aia_wind'))
from aia_features import LAT_GRID,LON_GRID,aligned_change,attach_features,segment_dark,build_features
from backtest import load_year


def test_available_time_not_observation_time_and_no_stale_fill():
    data = pd.DataFrame({'observed_at': pd.to_datetime(['2023-01-01T00:00Z', '2023-01-01T06:00Z']), 'available_at': pd.to_datetime(['2023-01-01T02:00Z', '2023-01-01T08:00Z']), 'aia_area_x': [0.2, 0.8]})
    issues = pd.to_datetime(['2023-01-01T01:00Z', '2023-01-01T07:00Z', '2023-01-01T08:00Z', '2023-01-02T00:00Z'])
    result = attach_features(issues, data, max_age_hours=12)
    assert result.aia_available.tolist() == [0, 1, 1, 0]
    assert result.aia_area_x.iloc[1] == 0.2
    assert result.aia_area_x.iloc[2] == 0.8
    assert result.aia_area_x.iloc[[0, 3]].isna().all()
    altered = data.copy()
    altered.loc[1, 'aia_area_x'] = 100
    pd.testing.assert_frame_equal(result.iloc[:2], attach_features(issues, altered).iloc[:2])

def test_carrington_alignment_wraps_longitude_without_filling_unseen_surface():
    old = ((LON_GRID >= 0) & (LON_GRID <= 12)).astype(float)
    current = ((LON_GRID + 4 >= 0) & (LON_GRID + 4 <= 12)).astype(float)
    delta = aligned_change(current, old, current_lon=2, previous_lon=358)
    assert np.nanmax(abs(delta)) == 0
    assert np.isnan(delta[:, -2:]).all()
    assert np.isfinite(delta[:, :-2]).all()

def test_segmentation_is_exposure_scale_invariant_and_rejects_tiny_components():
    mu = np.ones(LAT_GRID.shape)
    image = np.full(LAT_GRID.shape, 100.0)
    image[20:25, 20:25] = 20
    image[40, 40] = 20
    a, _ = segment_dark(image, mu)
    b, _ = segment_dark(image * 7, mu)
    np.testing.assert_array_equal(a, b)
    assert a[20:25, 20:25].sum() == 25
    assert a[40, 40] == 0

def test_temporal_features_use_only_previous_frames(tmp_path, monkeypatch):
    import aia_features as module
    raw = tmp_path / 'raw'
    raw.mkdir()
    dates = pd.to_datetime(['2022-12-05T06:00Z', '2022-12-31T12:00Z', '2023-01-01T12:00Z'])
    for i in range(3):
        (raw / f'AIA{i}_0193.fits').write_bytes(bytes([i]))

    def extract(path, **kwargs):
        import hashlib
        i = int(path.name[3])
        dark = np.full(LAT_GRID.shape, i / 3, dtype=np.float32)
        return (dark, dark, dict(observed_at=dates[i].isoformat(), carrington_lon=0.0, b0_deg=0.0, valid_fraction=1.0, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    monkeypatch.setattr(module, 'extract_frame', extract)
    result = build_features(raw, tmp_path / 'features')
    assert pd.isna(result.aia_rotation_separation_h.iloc[0])
    assert result.aia_rotation_separation_h.iloc[2] == 654
    assert result.aia_24h_separation_h.iloc[2] == 24
    assert result.aia_delta_rotation_lat1_lon2.iloc[2] == pytest.approx(2 / 3)
    assert result.aia_delta_24h_lat1_lon2.iloc[2] == pytest.approx(1 / 3)
    pd.testing.assert_frame_equal(result, build_features(raw, tmp_path / 'features'))

@pytest.fixture
def archive(tmp_path):
    oof, omni, features_dir = [tmp_path / name for name in ['oof', 'omni', 'features']]
    for p in [oof, omni, features_dir]:
        p.mkdir()
    frames = []
    for year in [2018, 2019, 2020]:
        clock = pd.date_range(f'{year}-01-01', periods=24 * 12, freq='h', tz='UTC')
        speed = 400 + 50 * np.sin(np.arange(len(clock)) / 30)
        speed[100] = np.nan
        pd.DataFrame({'issue_time': clock, 'v': speed}).to_parquet(omni / f'omni_{year}.parquet')
        issues = clock[::6]
        rows = pd.DataFrame({'issue_time': np.repeat(issues, 2), 'lead_hours': np.tile([48, 120], len(issues))})
        rows['valid_time'] = rows.issue_time + pd.to_timedelta(rows.lead_hours, unit='h')
        rows['dlinear_v'] = 390.0
        rows['model_train_end'] = f'{year}-01-01T00:00:00Z'
        rows['model_sha256'] = 'synthetic_fixture'
        rows['strategy'] = 'fixed_initial'
        extra = rows.iloc[:1].copy()
        extra['issue_time'] = pd.Timestamp(f'{year}-12-31T18:00Z')
        extra['valid_time'] = extra.issue_time + pd.Timedelta(hours=48)
        pd.concat([rows, extra]).to_parquet(oof / f'year={year}.parquet', index=False)
        frames.append(pd.DataFrame({'observed_at': issues - pd.Timedelta(hours=3), 'available_at': issues - pd.Timedelta(hours=1), 'aia_area_x': np.sin(np.arange(len(issues)) / 5), 'aia_valid_fraction': 1.0, 'aia_b0_deg': 0.0, 'aia_delta_rotation_x': 0.2, 'aia_rotation_separation_h': 654.0}))
    features = pd.concat(frames, ignore_index=True)
    features.to_parquet(features_dir / 'features.parquet')
    (features_dir / 'extraction.json').write_text('{}')
    return (oof, omni, features_dir / 'features.parquet')

def test_labels_are_unfilled_and_split_boundary_is_enforced(archive):
    oof, omni, features = archive
    result = load_year(2018, oof, omni, pd.read_parquet(features), leads=[48, 120])
    assert (result.valid_time.dt.year == 2018).all()
    assert result.target_v.isna().any()
    p = oof / 'year=2018.parquet'
    invalid = pd.read_parquet(p)
    invalid['model_train_end'] = '2021-01-01T00:00Z'
    invalid.to_parquet(p)
    with pytest.raises(ValueError, match='chronological'):
        load_year(2018, oof, omni, pd.read_parquet(features), leads=[48])
