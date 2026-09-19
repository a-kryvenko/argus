"""Storage/scheduling tests must not need model-loading or private dependencies."""
from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def test_generation_imports_without_model_runtime():
    environment = os.environ.copy()
    environment['PYTHONPATH'] = os.pathsep.join([
        str(ROOT / 'apps/prophet/src'), str(ROOT / 'packages/common/src'),
        str(ROOT / 'packages/forecast/src'), environment.get('PYTHONPATH', ''),
    ])
    result = subprocess.run([sys.executable, '-c', '''
import sys
blocked = {'joblib', 'sklearn', 'lightgbm', 'forecast_core', 'intelligence_core', 'clio', 'argus_clio'}
class BlockModelRuntime:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockModelRuntime())
from argus_prophet import cli, generation, models, products
assert callable(cli.generate_products)
assert callable(generation.calculate)
assert callable(models.load_model)
assert products.select_products('solar-wind-speed') == ('solar-wind-speed',)
assert not blocked.intersection(sys.modules)
'''], env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
