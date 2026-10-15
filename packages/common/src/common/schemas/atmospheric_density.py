from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DensityCell(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    altitude_km: float = Field(ge=90, le=2500)
    latitude_deg: float = Field(ge=-90, le=90)
    rho_kg_m3: float = Field(gt=0)
    rho_lon_p10_kg_m3: float = Field(gt=0)
    rho_lon_p90_kg_m3: float = Field(gt=0)
    rho_reference_kg_m3: float | None = Field(default=None, gt=0)


class DensityBackground(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    method: Literal['jb2008_fixed_ap4_v1']
    geomagnetic_ap: Literal[4.0] = 4.0
    dtc_kelvin: float = Field(ge=0)
    spatial_support: Literal['longitude_mean'] = 'longitude_mean'
    solar_drivers: Literal['same_as_forecast'] = 'same_as_forecast'


class DensityPoint(BaseModel):
    valid_time: datetime
    lead_hours: int = Field(ge=0, le=48)
    cells: list[DensityCell]


class DensityForecast(BaseModel):
    target: Literal["atmospheric-density"] = "atmospheric-density"
    model: Literal["JB2008", "DTM2020"] = "JB2008"
    driver_mode: Literal["observed_persistence"] = "observed_persistence"
    unit: Literal["kg/m^3"] = "kg/m^3"
    issue_time: datetime
    observed_at: datetime
    history_start: datetime
    dtc_observed_at: datetime | None = None
    geomagnetic_observed_at: datetime | None = None
    background_method: Literal["trailing_81_daily_values", "trailing_81_daily_values_linear_gapfill"]
    background_interpolated_days: dict[str, list[date]] = Field(default_factory=dict)
    dtc_method: Literal["causal_dst_ap_v1"] | None = None
    geomagnetic_method: Literal["causal_dst_ap_v1", "ap_to_kp_lag3h_mean24h_v1"] | None = None
    horizon_hours: int
    predictions: list[DensityPoint]
    reference: DensityBackground | None = None

    @model_validator(mode='after')
    def model_metadata(self):
        if self.model == 'DTM2020':
            if (self.dtc_method is not None or self.dtc_observed_at is not None
                    or self.geomagnetic_method != 'ap_to_kp_lag3h_mean24h_v1'
                    or self.geomagnetic_observed_at is None):
                raise ValueError('Invalid DTM2020 driver metadata')
        else:
            if self.dtc_method != 'causal_dst_ap_v1' or self.dtc_observed_at is None:
                raise ValueError('Missing JB2008 DTC metadata')
            if self.geomagnetic_method not in (None, self.dtc_method):
                raise ValueError('Inconsistent JB2008 geomagnetic method')
            if self.geomagnetic_observed_at not in (None, self.dtc_observed_at):
                raise ValueError('Inconsistent JB2008 geomagnetic timestamp')
            self.geomagnetic_method = self.dtc_method
            self.geomagnetic_observed_at = self.dtc_observed_at
        return self
