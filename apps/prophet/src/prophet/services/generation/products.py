"""Single generation catalog; the public release contract remains independent."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    artifact: str
    bundled: bool = True


@dataclass(frozen=True)
class Product:
    models: tuple[Model, ...]

    @property
    def artifacts(self) -> tuple[str, ...]:
        return tuple(model.artifact for model in self.models)


PRODUCTS = {
    'solar-wind-events': Product((Model('solar_wind_events', bundled=False),)),
    'geomagnetic-activity': Product((
        Model('kp_threshold'),
        Model('ap_quantile'))),
    'dst': Product((Model('dst_quantile'),)),
    'solar-wind-speed': Product((
        Model('plasma_speed_quantile'),
        Model('plasma_speed_threshold'))),
    'solar-wind-density': Product((Model('plasma_density_quantile'),)),
    'solar-wind-temperature': Product((Model('plasma_temperature_quantile'),)),
    'hmf': Product((
        Model('hmf_total_threshold'),
        Model('hmf_southward_threshold'))),
    'solar-radiation': Product(tuple(Model(name+'_quantile') for name in ('f10_7', 's10', 'm10', 'y10'))),
    'atmospheric-density': Product((Model('atmospheric_density', bundled=False),)),
}
GENERATION_CHOICES = ('all', *PRODUCTS)
# Solar verification needs independent reference observations, not our GOES estimates.
VERIFIED_PRODUCTS = tuple(name for name in PRODUCTS if name not in (
    'atmospheric-density', 'solar-wind-events', 'solar-radiation'))


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
