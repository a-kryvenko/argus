"""Generation catalog; the versioned public release contract remains independent."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Product:
    artifacts: tuple[str, ...]
    backend: str = 'models'


PRODUCTS = {
    'geomagnetic-activity': Product(('kp_threshold', 'ap_quantile')),
    'dst': Product(('dst_quantile',)),
    'solar-wind-speed': Product(('plasma_speed_quantile', 'plasma_speed_threshold')),
    'solar-wind-density': Product(('plasma_density_quantile',)),
    'hmf': Product(('hmf_total_threshold', 'hmf_southward_threshold')),
    'atmospheric-density': Product(('atmospheric_density',), backend='density'),
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
