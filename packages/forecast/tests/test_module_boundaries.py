"""Shared lifecycle and algorithms must load without product adapters."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize('module,blocked', [
    ('forecast.inference.base', ['forecast.adapters', 'forecast.inference.rotation_dlinear',
                                'forecast.inference.feature_models', 'forecast.inference.bucketed']),
    ('forecast.inference.aia_wind', ['forecast.adapters', 'forecast.inference.bucketed']),
    ('forecast.inputs.aia.alignment', ['forecast.inference', 'forecast.adapters']),
])
def test_dependency_direction(module, blocked):
    result = subprocess.run([sys.executable, '-c', f'''
import importlib
import sys
blocked = {blocked!r}
class Block:
    def find_spec(self, fullname, *args):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise ImportError(fullname)
sys.meta_path.insert(0, Block())
importlib.import_module({module!r})
'''], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_former_class_paths_resolve_for_saved_objects():
    import pickle
    from forecast.api import SWSpeedFS, SWSpeedProbaFS, SWDensityFS
    from forecast.adapters.plasma import AIAWindServiceMixin
    from forecast.inference.base import DefaultForecastService
    from forecast.inference.quantiles import QuantileForecastService
    from forecast.inference.threshold import ThresholdForecastService

    for module, classes in [
        ('forecast.inference.plasma_fs', [SWSpeedFS, SWSpeedProbaFS, SWDensityFS]),
        ('forecast.inference._forecast_service', [DefaultForecastService,
                                                 QuantileForecastService, ThresholdForecastService]),
        ('forecast.inference.aia_wind', [AIAWindServiceMixin]),
    ]:
        for cls in classes:
            # Protocol 0 GLOBAL reproduces the class reference in an old pickle.
            reference = f'c{module}\n{cls.__name__}\n.'.encode()
            assert pickle.loads(reference) is cls
