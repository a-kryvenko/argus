"""Forecast-owned observation context assembled through the Clio API."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, FiniteFloat
from common.schemas.observation import Observation
from common.schemas.observation_files import RawObservationFile

DensityMetric = Literal['f10_7', 's10', 'm10', 'y10', 'dst', 'ap']
DENSITY_METRICS = ('f10_7', 's10', 'm10', 'y10', 'dst', 'ap')


class SourceMeasurement(BaseModel):
    metric: DensityMetric
    value: FiniteFloat
    observed_at: AwareDatetime
    available_at: AwareDatetime | None = None


class SpeedObservation(BaseModel):
    """Observed hourly speed, without interpolation or gap filling."""
    issue_time: AwareDatetime
    v: FiniteFloat


class DensityObservation(BaseModel):
    """Observed hourly proton density, without interpolation or gap filling."""
    issue_time: AwareDatetime
    n: FiniteFloat


class ObservationInputs(BaseModel):
    schema_version: Literal[1] = 1
    as_of: AwareDatetime
    read_at: AwareDatetime
    observations: Observation
    measurements: list[SourceMeasurement] = Field(default_factory=list)
    speed_observations: list[SpeedObservation] = Field(default_factory=list)
    density_observations: list[DensityObservation] = Field(default_factory=list)

    files: list[RawObservationFile] = Field(default_factory=list)

    # Native, unfilled hourly Clio solar-wind history for IMF inference.
    solar_wind_hourly: dict | None = None
