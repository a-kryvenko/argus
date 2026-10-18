"""Public contracts and calculation work without model implementations."""
import subprocess
import sys


def test_contracts_import_without_private_backend():
    result = subprocess.run([sys.executable, '-c', '''
import sys
class Block:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'forecast_core', 'notebooks', 'torch', 'lightgbm', 'joblib', 'sklearn'}:
            raise ModuleNotFoundError(name, name=name)
sys.meta_path.insert(0, Block())
from forecast.calculation import ForecastResult, SnapshotService, calculate_snapshot
from forecast.inference.quantile import QuantileFS
from forecast.inference.probability import ProbabilityFS
from forecast.evaluation import score, validate_predictions, match_observations
import inspect
assert inspect.isabstract(QuantileFS)
assert inspect.isabstract(ProbabilityFS)
assert callable(calculate_snapshot)
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
