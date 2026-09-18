import subprocess
import sys


def test_public_solar_wind_pipeline_runs_without_private_backend():
    result = subprocess.run([sys.executable, '-c', '''
import sys
class BlockPrivate:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'forecast_core', 'intelligence_core'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockPrivate())
import pandas as pd
from forecast.demo import run_demo, ISSUE_TIME
first, second = run_demo(), run_demo()
assert [r.name for r in first] == ['plasma_speed_quantile', 'plasma_speed_threshold', 'plasma_density_quantile']
for a, b in zip(first, second):
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert len(a.frame) == 6
    assert a.frame.issue_time.eq(ISSUE_TIME).all()
    assert a.frame.lead_hours.tolist() == list(range(1, 7))
    assert a.frame.valid_time.iloc[0] == pd.Timestamp(ISSUE_TIME) + pd.Timedelta(hours=1)
for result, target in ((first[0], 'v'), (first[2], 'n')):
    assert result.frame[f'{target}_q10'].le(result.frame[f'{target}_q50']).all()
    assert result.frame[f'{target}_q50'].le(result.frame[f'{target}_q90']).all()
assert first[1].frame.p_v_ge_450.eq(0.3).all()
assert first[1].frame.p_v_ge_500.eq(0.1).all()
assert first[1].frame.p_v_ge_600.eq(0.01).all()
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
