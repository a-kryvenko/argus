"""Projection of a validated drag assessment to the public HTTP contract."""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from common.schemas.leo_drag import DragPoint, LeoDragAssessment, LeoDragRequest


class PublicDragSource(BaseModel):
    release_id: UUID
    issue_time: datetime
    observed_at: datetime
    dtc_observed_at: datetime | None = None
    geomagnetic_observed_at: datetime | None = None
    driver_mode: Literal['observed_persistence']
    background_interpolated: bool


class PublicDensityMetadata(BaseModel):
    model: Literal['JB2008', 'DTM2020']
    history_start: datetime
    background_method: Literal[
        'trailing_81_daily_values',
        'trailing_81_daily_values_linear_gapfill',
        'published_81_day_mean_persistence',
    ]
    background_mean_observed_at: dict[str, datetime] = Field(default_factory=dict)
    background_interpolated_days: dict[str, list[date]]
    dtc_method: Literal['causal_dst_ap_v1'] | None = None
    geomagnetic_method: Literal['causal_dst_ap_v1', 'ap_to_kp_lag3h_mean24h_v1'] | None = None


class PublicDragMetadata(BaseModel):
    model: Literal['circular_leo_drag_v1']
    source: PublicDensityMetadata


class PublicLeoDragAssessment(BaseModel):
    computed_at: datetime
    start_time: datetime
    end_time: datetime
    inputs: LeoDragRequest
    source: PublicDragSource
    mean_density_kg_m3: float
    mean_drag_accel_m_s2: float
    delta_v_loss_m_s: float
    estimated_altitude_loss_m: float
    drag_risk: Literal['not_assessed', 'low', 'elevated', 'high']
    risk_reason: str
    assumptions: list[str]
    predictions: list[DragPoint]
    meta: PublicDragMetadata | None = None

    @classmethod
    def from_assessment(cls, assessment: LeoDragAssessment, *, meta: bool) -> PublicLeoDragAssessment:
        source = assessment.source
        return cls(
            **assessment.model_dump(exclude={'source', 'model'}),
            source=PublicDragSource(
                release_id=source.release_id,
                issue_time=source.issue_time,
                observed_at=source.observed_at,
                dtc_observed_at=source.dtc_observed_at,
                geomagnetic_observed_at=source.geomagnetic_observed_at,
                driver_mode=source.driver_mode,
                background_interpolated=any(source.background_interpolated_days.values()),
            ),
            meta=PublicDragMetadata(
                model=assessment.model,
                source=PublicDensityMetadata(
                    model=source.density_model,
                    history_start=source.history_start,
                    background_method=source.background_method,
                    background_interpolated_days=source.background_interpolated_days,
                    dtc_method=source.dtc_method,
                    geomagnetic_method=source.geomagnetic_method,
                ),
            ) if meta else None,
        )
