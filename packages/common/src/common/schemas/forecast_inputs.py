"""Version 1 observation read contract, independent of storage and model code."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, FiniteFloat
from common.schemas.observation import Observation

DensityMetric = Literal['f10_7', 's10', 'm10', 'y10', 'dst', 'ap']
DENSITY_METRICS = ('f10_7', 's10', 'm10', 'y10', 'dst', 'ap')


class SourceMeasurement(BaseModel):
    metric: DensityMetric
    value: FiniteFloat
    observed_at: AwareDatetime


class SpeedObservation(BaseModel):
    """Observed hourly speed, without interpolation or gap filling."""
    issue_time: AwareDatetime
    v: FiniteFloat


class DensityObservation(BaseModel):
    """Observed hourly proton density, without interpolation or gap filling."""
    issue_time: AwareDatetime
    n: FiniteFloat


class AIAFeatureFrame(BaseModel):
    """Six-hour model input derived from the hourly owner archive as of the read."""
    slot_at: AwareDatetime
    observed_at: AwareDatetime
    available_at: AwareDatetime
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    features: dict[str, FiniteFloat | None]


class GONGFeatureFrame(BaseModel):
    """An immutable observed magnetogram and versioned model features."""
    observed_at: AwareDatetime
    available_at: AwareDatetime
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_product: str
    feature_version: Literal['gong-bands-v1'] = 'gong-bands-v1'
    features: dict[str, FiniteFloat]


class RawObservationFile(BaseModel):
    kind: Literal['aia', 'gong', 'goes']
    slot_at: AwareDatetime
    observed_at: AwareDatetime
    available_at: AwareDatetime
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_product: str


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


class ProswinPrediction(BaseModel):
    valid_time: AwareDatetime
    image_slot: AwareDatetime
    available_at: AwareDatetime
    value: FiniteFloat = Field(gt=0)
    source_cutoff: AwareDatetime | None = None
    model_version: Literal['proswin-fold1-nrt-v1', 'proswin-fold1-science-v1'] = 'proswin-fold1-nrt-v1'


class ForecastInputs(ObservationInputs):
    """Prophet-owned enrichment of the source observation response."""
    aia_frames: list[AIAFeatureFrame] = Field(default_factory=list)
    gong: GONGFeatureFrame | None = None
    proswin_predictions: list[ProswinPrediction] = Field(default_factory=list)

    proswin_ready_at: AwareDatetime | None = None
    proswin_job: dict = Field(default_factory=dict)
