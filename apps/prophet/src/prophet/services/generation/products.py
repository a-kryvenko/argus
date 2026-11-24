"""Single generation catalog; the public release contract remains independent."""
from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True)
class Model:
    artifact: str
    implementation: str | None = None
    bundled: bool = True

    def service_class(self):
        module, name = self.implementation.split(':')
        return getattr(import_module('forecast_core.' + module), name)


@dataclass(frozen=True)
class Product:
    models: tuple[Model, ...]
    required_inputs: tuple[str, ...]
    optional_inputs: tuple[str, ...] = ()
    requires_normalized: bool = False

    @property
    def artifacts(self) -> tuple[str, ...]:
        return tuple(model.artifact for model in self.models)


PRODUCTS = {
    'solar-wind-events': Product((Model('solar_wind_events', bundled=False),),
                                ('events',), ('hourly', 'proswin_cache')),
    'geomagnetic-activity': Product((
        Model('kp_threshold', 'inference.geomagnetic.kp.lightgbm:LightgbmKpProbabilityFS'),
        Model('ap_quantile', 'inference.geomagnetic.ap.lightgbm:LightgbmApQuantileFS')),
        ('hourly',), requires_normalized=True),
    'dst': Product((Model('dst_quantile', 'inference.geomagnetic.dst.lightgbm:LightgbmDstQuantileFS'),),
                   ('hourly',), requires_normalized=True),
    'solar-wind-speed': Product((
        Model('plasma_speed_quantile', 'inference.plasma.speed.dlinear_proswin:DlinearProswinPlasmaSpeedQuantileFS'),
        Model('plasma_speed_threshold', 'inference.plasma.speed.dlinear_proswin:DlinearProswinPlasmaSpeedProbabilityFS')),
        ('hourly',), ('events', 'proswin_cache', 'proswin_queue')),
    'solar-wind-density': Product((Model('plasma_density_quantile',
        'inference.plasma.density.dlinear_proswin:DlinearProswinPlasmaDensityQuantileFS'),),
        ('hourly',), ('proswin_cache', 'proswin_queue')),
    'solar-wind-temperature': Product((Model('plasma_temperature_quantile',
        'inference.plasma.temperature.proswin:ProswinPlasmaTemperatureQuantileFS'),),
        ('minute',), ('hourly', 'proswin_cache', 'proswin_queue')),
    'hmf': Product((
        Model('hmf_total_threshold', 'inference.imf.bt.lightgbm:LightgbmBtProbabilityFS'),
        Model('hmf_southward_threshold', 'inference.imf.bz.lightgbm:LightgbmBzProbabilityFS')),
        ('minute',)),
    'solar-radiation': Product((
        Model('f10_7_quantile', 'inference.radiation.f107.lightgbm:LightgbmF107QuantileFS'),
        Model('s10_quantile', 'inference.radiation.daily:DailyS10QuantileFS'),
        Model('m10_quantile', 'inference.radiation.daily:DailyM10QuantileFS'),
        Model('y10_quantile', 'inference.radiation.daily:DailyY10QuantileFS')),
        ('hourly', 'goes'), requires_normalized=True),
    'atmospheric-density': Product((Model('atmospheric_density',
        'models.AtmosphericDensityForecastService:AtmosphericDensityForecastService', bundled=False),),
        ('hourly', 'goes')),
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
