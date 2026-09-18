"""Single generation catalog; the public release contract remains independent."""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Model:
    artifact: str
    service: str | None = None

    def service_class(self):
        if self.service is None:
            raise ValueError(f'{self.artifact} does not use a model bundle')
        module, name = self.service.rsplit('.', 1)
        return getattr(import_module(module), name)


@dataclass(frozen=True)
class Product:
    models: tuple[Model, ...]
    backend: str = 'models'

    @property
    def artifacts(self) -> tuple[str, ...]:
        return tuple(model.artifact for model in self.models)


PRODUCTS = {
    'geomagnetic-activity': Product((
        Model('kp_threshold', 'forecast_core.api.KPProbaFS'),
        Model('ap_quantile', 'forecast_core.api.APFS'))),
    'dst': Product((Model('dst_quantile', 'forecast_core.api.DstFS'),)),
    'solar-wind-speed': Product((
        Model('plasma_speed_quantile', 'forecast.api.SWSpeedFS'),
        Model('plasma_speed_threshold', 'forecast.api.SWSpeedProbaFS'))),
    'solar-wind-density': Product((Model('plasma_density_quantile', 'forecast.api.SWDensityFS'),)),
    'hmf': Product((
        Model('hmf_total_threshold', 'forecast_core.api.HMFTotalProbaFS'),
        Model('hmf_southward_threshold', 'forecast_core.api.HMFSouthProbaFS'))),
    'atmospheric-density': Product((Model('atmospheric_density'),), backend='density'),
}
GENERATION_CHOICES = ('all', *PRODUCTS)


def select_products(selection: str) -> tuple[str, ...]:
    if selection == 'all':
        return tuple(PRODUCTS)
    if selection not in PRODUCTS:
        raise ValueError(f'Unsupported forecast product: {selection}')
    return (selection,)


def attempt_selections(product: str) -> list[str]:
    """Include old CLI aliases when looking up historical attempts."""
    if product not in PRODUCTS:
        return []
    historical_names = {'solar-wind-speed': 'wind', 'geomagnetic-activity': 'kp',
                        'atmospheric-density': 'density'}
    return ['all', product] + ([historical_names[product]] if product in historical_names else [])
