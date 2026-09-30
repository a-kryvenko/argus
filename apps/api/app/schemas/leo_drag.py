"""Public impact projection, after the internal assessment has been validated."""
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel
from common.schemas.leo_drag import LeoDragAssessment as InternalAssessment, LeoDragRequest, DragPoint
from app.schemas.atmospheric_density import DensityMetadata
from app.schemas.metadata import metadata_field


class DragSource(BaseModel):
    release_id: UUID
    issue_time: datetime
    observed_at: datetime
    dtc_observed_at: datetime
    driver_mode: Literal['observed_persistence']
    background_interpolated: bool


class DragMetadata(BaseModel):
    model: Literal['circular_leo_drag_v1']
    source: DensityMetadata


class LeoDragAssessment(BaseModel):
    computed_at: datetime
    start_time: datetime
    end_time: datetime
    inputs: LeoDragRequest
    source: DragSource
    mean_density_kg_m3: float
    mean_drag_accel_m_s2: float
    delta_v_loss_m_s: float
    estimated_altitude_loss_m: float
    drag_risk: Literal['not_assessed', 'low', 'elevated', 'high']
    risk_reason: str
    assumptions: list[str]
    predictions: list[DragPoint]
    meta: DragMetadata | None = metadata_field()

    @classmethod
    def from_internal(cls, assessment: InternalAssessment, *, meta: bool = False):
        data = assessment.model_dump()
        source = data.pop('source')
        return cls(**data, source=DragSource(**source,
                   background_interpolated=any(assessment.source.background_interpolated_days.values())),
                   meta=DragMetadata(model=assessment.model,
                       source=DensityMetadata(**source, model=assessment.source.density_model)) if meta else None)
