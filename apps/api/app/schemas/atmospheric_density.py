from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DensityCell(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    altitude_km: float = Field(ge=90, le=2500)
    latitude_deg: float = Field(ge=-90, le=90)
    rho_kg_m3: float = Field(gt=0)
    rho_lon_p10_kg_m3: float = Field(gt=0)
    rho_lon_p90_kg_m3: float = Field(gt=0)


class DensityPoint(BaseModel):
    valid_time: datetime
    lead_hours: int = Field(ge=0, le=48)
    cells: list[DensityCell]


class DensityForecast(BaseModel):
    target: Literal["atmospheric-density"] = "atmospheric-density"
    model: Literal["JB2008"] = "JB2008"
    driver_mode: Literal["observed_persistence"] = "observed_persistence"
    unit: Literal["kg/m^3"] = "kg/m^3"
    issue_time: datetime
    observed_at: datetime
    history_start: datetime
    dtc_observed_at: datetime
    background_method: Literal["trailing_81_daily_values", "trailing_81_daily_values_linear_gapfill"]
    background_interpolated_days: dict[str, list[date]] = Field(default_factory=dict)
    dtc_method: Literal["causal_dst_ap_v1"]
    horizon_hours: int
    predictions: list[DensityPoint]
