"""Public interfaces remain importable; model recipes require the private backend."""
import subprocess
import sys
from pathlib import Path
import ast


def test_contracts_import_without_private_backend():
    result = subprocess.run([sys.executable, '-c', '''
import sys
class Block:
    def find_spec(self, name, *args):
        if name == "forecast_core" or name.startswith("forecast_core."):
            raise ModuleNotFoundError(name, name=name)
sys.meta_path.insert(0, Block())
import forecast.api as api
from forecast.evaluation import score
from forecast.inputs.aia.schema import feature_columns
assert api.ForecastResult
try:
    api.SWSpeedFS
except ImportError as exc:
    assert "private forecast-core" in str(exc)
else:
    raise AssertionError("Model unexpectedly loaded without private backend")
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_model_compatibility_files_contain_no_implementation():
    root = Path(__file__).resolve().parents[1] / 'src/forecast'
    for path in (root/'inference').glob('*.py'):
        if path.name in {'base.py','__init__.py','_forecast_service.py','plasma_fs.py'}:
            continue
        tree=ast.parse(path.read_text())
        assert not any(isinstance(n, ast.ClassDef) for n in tree.body), path
        assert all(n.name == '__getattr__' for n in tree.body if isinstance(n, ast.FunctionDef)), path
