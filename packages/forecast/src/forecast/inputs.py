"""Explicit observation snapshots enriched for forecast calculation."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field, FiniteFloat
from common.schemas.forecast_inputs import ObservationInputs
from common.schemas.solar_wind_events import CoronalHoleFrame, EventInputs


class ProswinPrediction(BaseModel):
    valid_time: AwareDatetime
    image_slot: AwareDatetime
    available_at: AwareDatetime
    value: FiniteFloat = Field(gt=0)
    source_cutoff: AwareDatetime | None = None
    model_version: Literal['proswin-fold1-nrt-v1', 'proswin-fold1-science-v1'] = 'proswin-fold1-nrt-v1'


class ForecastInputs(ObservationInputs):
    """Prophet-owned enrichment of the source observation response."""
    proswin_predictions: list[ProswinPrediction] = Field(default_factory=list)

    proswin_ready_at: AwareDatetime | None = None
    proswin_job: dict = Field(default_factory=dict)
    solar_wind_events: EventInputs | None = None
    coronal_holes: CoronalHoleFrame | None = None
    hss_input_status: dict = Field(default_factory=dict)
