import subprocess
import sys


def test_public_registry_imports_without_private_backend():
    result = subprocess.run([sys.executable, "-c", """
import sys
class BlockPrivate:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'forecast_core', 'intelligence_core'}:
            raise ModuleNotFoundError(fullname, name=fullname)
sys.meta_path.insert(0, BlockPrivate())
from forecast.forecast_services import ForecastService, ForecastServiceRegistry
for service in ForecastService:
    assert ForecastServiceRegistry.get(service).registry_name
from forecast.inference.plasma_fs import SWSpeedFS
try:
    SWSpeedFS({})._build_features(None)
except ModuleNotFoundError as exc:
    assert exc.name in {'forecast_core', 'forecast_core.api'}
else:
    raise AssertionError('Missing backend must fail explicitly')
"""], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
