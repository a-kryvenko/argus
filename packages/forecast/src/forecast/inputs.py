"""Explicit observation snapshots enriched for forecast calculation."""
from datetime import UTC, time
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, FiniteFloat, model_validator
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


class F107Measurement(BaseModel):
    """Daily F10.7 flux at its original measurement and receipt times."""

    observed_at: AwareDatetime
    received_at: AwareDatetime
    value: FiniteFloat = Field(gt=0)
    source_product: Literal['celestrak.f107', 'swpc.f107', 'celestrak.f107_archive']

    @model_validator(mode='after')
    def validate_penticton_noon(self):
        if self.source_product == 'swpc.f107' and self.observed_at.astimezone(UTC).time() != time(20):
            raise ValueError('Expected the Penticton noon F10.7 sample')
        return self


class ForecastInputs(ObservationInputs):
    """Model context enriched after reading observation resources."""
    proswin_predictions: list[ProswinPrediction] = Field(default_factory=list)

    proswin_ready_at: AwareDatetime | None = None
    proswin_job: dict = Field(default_factory=dict)
    solar_wind_events: EventInputs | None = None
    coronal_holes: CoronalHoleFrame | None = None
    hss_input_status: dict = Field(default_factory=dict)
    geomagnetic_history: GeomagneticHistory | None = None
    f107_measurements: list[F107Measurement] = Field(default_factory=list)
