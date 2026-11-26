"""Explicit observation snapshots enriched for forecast calculation."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, FiniteFloat
from forecast.observation_inputs import ObservationInputs
from common.schemas.solar_wind_events import CoronalHoleFrame, EventInputs


class ProswinPrediction(BaseModel):
    valid_time: AwareDatetime
    image_slot: AwareDatetime
    available_at: AwareDatetime
    value: FiniteFloat = Field(gt=0)
    source_cutoff: AwareDatetime | None = None
    model_version: Literal['proswin-fold1-nrt-v1', 'proswin-fold1-science-v1'] = 'proswin-fold1-nrt-v1'


class GeomagneticKpObservation(BaseModel):
    """Kp at the start of its native three-hour observation interval."""

    time: AwareDatetime
    kp: FiniteFloat = Field(ge=0, le=9)


class GeomagneticObservationPoint(BaseModel):
    """Available values on the model's hourly feature clock."""

    time: AwareDatetime
    kp: FiniteFloat | None = Field(default=None, ge=0, le=9)
    dst: FiniteFloat | None = None
    v: FiniteFloat | None = None
    f10_7: FiniteFloat | None = None
    by: FiniteFloat | None = None
    bz: FiniteFloat | None = None
    n: FiniteFloat | None = None


class GeomagneticHistory(BaseModel):
    raw: list[GeomagneticKpObservation]
    prior: list[GeomagneticObservationPoint]


class ForecastInputs(ObservationInputs):
    """Model context enriched after reading observation resources."""
    proswin_predictions: list[ProswinPrediction] = Field(default_factory=list)

    proswin_ready_at: AwareDatetime | None = None
    proswin_job: dict = Field(default_factory=dict)
    solar_wind_events: EventInputs | None = None
    coronal_holes: CoronalHoleFrame | None = None
    hss_input_status: dict = Field(default_factory=dict)
    geomagnetic_history: GeomagneticHistory | None = None
