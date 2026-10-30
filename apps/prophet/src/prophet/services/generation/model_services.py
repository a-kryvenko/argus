"""Explicit inference implementations loaded only in the calculation process."""
from forecast_core.inference.geomagnetic.ap.lightgbm import LightgbmApQuantileFS
from forecast_core.inference.geomagnetic.dst.lightgbm import LightgbmDstQuantileFS
from forecast_core.inference.geomagnetic.kp.lightgbm import LightgbmKpProbabilityFS
from forecast_core.inference.imf.bt.lightgbm import LightgbmBtProbabilityFS
from forecast_core.inference.imf.bz.lightgbm import LightgbmBzProbabilityFS
from forecast_core.inference.plasma.speed.dlinear_proswin import (
    DlinearProswinPlasmaSpeedQuantileFS, DlinearProswinPlasmaSpeedProbabilityFS,
)
from forecast_core.inference.plasma.density.dlinear_proswin import DlinearProswinPlasmaDensityQuantileFS
from forecast_core.inference.plasma.temperature.proswin import ProswinPlasmaTemperatureQuantileFS
from forecast_core.inference.radiation.f107.lightgbm import LightgbmF107QuantileFS
from forecast_core.inference.radiation.daily import DailyS10QuantileFS, DailyM10QuantileFS, DailyY10QuantileFS
from forecast_core.models.AtmosphericDensityForecastService import AtmosphericDensityForecastService


SERVICES = {
    'kp_threshold': LightgbmKpProbabilityFS,
    'ap_quantile': LightgbmApQuantileFS,
    'dst_quantile': LightgbmDstQuantileFS,
    'plasma_speed_quantile': DlinearProswinPlasmaSpeedQuantileFS,
    'plasma_speed_threshold': DlinearProswinPlasmaSpeedProbabilityFS,
    'plasma_density_quantile': DlinearProswinPlasmaDensityQuantileFS,
    'plasma_temperature_quantile': ProswinPlasmaTemperatureQuantileFS,
    'hmf_total_threshold': LightgbmBtProbabilityFS,
    'hmf_southward_threshold': LightgbmBzProbabilityFS,
    'atmospheric_density': AtmosphericDensityForecastService,
    'f10_7_quantile': LightgbmF107QuantileFS,
    's10_quantile': DailyS10QuantileFS,
    'm10_quantile': DailyM10QuantileFS,
    'y10_quantile': DailyY10QuantileFS,
}
