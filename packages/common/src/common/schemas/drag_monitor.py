"""Saved spacecraft and reproducible drag screening runs."""
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import Field, model_validator
from common.schemas.leo_drag import DragModel, DragThresholds, LeoDragAssessment, LeoDragRequest


class AttitudeMode(DragModel):
    name: str = Field(min_length=1, max_length=80, pattern=r'\S')
    effective_area_m2: float = Field(gt=0, le=1e6)
    drag_coefficient: float = Field(gt=0, le=10)


class DragEventPolicy(DragModel):
    growth_percent: float = Field(gt=0, le=10000)
    growth_min_m: float = Field(gt=0)


class DragProfileInput(DragModel):
    name: str = Field(min_length=1, max_length=120, pattern=r'\S')
    altitude_km: float = Field(ge=200, le=800)
    inclination_deg: float = Field(ge=0, le=180)
    mass_kg: float = Field(ge=0.01, le=1e7)
    horizon_hours: Literal[24, 48] = 24
    thresholds: DragThresholds
    modes: list[AttitudeMode] = Field(min_length=1, max_length=5)
    active_mode: str = Field(min_length=1, max_length=80)
    enabled: bool = True
    event_policy: DragEventPolicy | None = None

    @model_validator(mode='after')
    def valid_modes(self):
        names = [mode.name for mode in self.modes]
        if len(set(names)) != len(names) or self.active_mode not in names:
            raise ValueError('Mode names must be unique and include the active mode')
        return self

    def request_for(self, mode: AttitudeMode) -> LeoDragRequest:
        return LeoDragRequest(**self.model_dump(include={
            'altitude_km', 'inclination_deg', 'mass_kg', 'horizon_hours', 'thresholds'}),
            **mode.model_dump(exclude={'name'}))


class DragProfile(DragProfileInput):
    id: UUID
    revision: int
    updated_at: datetime


class ModeAssessment(DragModel):
    mode: str
    assessment: LeoDragAssessment | None = None
    error: Literal['unavailable', 'outside_model_domain'] | None = None
    elevated_at: datetime | None = None
    high_at: datetime | None = None


class DragRun(DragModel):
    id: UUID
    profile_id: UUID
    profile_revision: int
    profile_snapshot: DragProfileInput
    assessed_at: datetime
    release_id: UUID | None
    modes: list[ModeAssessment]


class DragMonitorItem(DragModel):
    profile: DragProfile
    latest: DragRun | None
    data_status: Literal['not_assessed', 'profile_changed', 'unavailable', 'stale', 'current']
    attention: Literal['not_assessed', 'low', 'elevated', 'high']
    loss_change_m: float | None = None


class DragChartPoint(DragModel):
    lead_hours: float
    valid_time: datetime
    altitude_loss_m: float


class DragChartCrossing(DragModel):
    level: Literal['elevated', 'high']
    lead_hours: float
    valid_time: datetime
    altitude_loss_m: float
    state: Literal['elapsed', 'upcoming']


class DragChartMode(DragModel):
    mode: str
    active: bool
    error: Literal['unavailable', 'outside_model_domain'] | None = None
    points: list[DragChartPoint] = Field(default_factory=list)
    elapsed: list[DragChartPoint] = Field(default_factory=list)
    remaining: list[DragChartPoint] = Field(default_factory=list)
    crossings: list[DragChartCrossing] = Field(default_factory=list)
    previous: list[DragChartPoint] = Field(default_factory=list)
    previous_issue_time: datetime | None = None
    previous_release_id: UUID | None = None
    loss_change_m: float | None = None


class DragOperatorChart(DragModel):
    run_id: UUID
    prepared_at: datetime
    start_time: datetime | None
    end_time: datetime | None
    horizon_hours: Literal[24, 48]
    now_position: Literal['before', 'within', 'after', 'unavailable']
    now_lead_hours: float | None
    elapsed_until_hours: float | None
    max_loss_m: float
    data_status: Literal['not_assessed', 'profile_changed', 'unavailable', 'stale', 'current']
    historical: bool
    thresholds: DragThresholds
    modes: list[DragChartMode]


class DragEventDraft(DragModel):
    id: UUID
    profile_id: UUID
    profile_revision: int
    spacecraft_name: str
    active_mode: str
    kind: Literal['risk_elevated', 'risk_high', 'risk_reduced', 'risk_recovered',
                  'loss_increased', 'data_stale', 'assessment_unavailable', 'data_recovered']
    severity: Literal['info', 'warning', 'high']
    detected_at: datetime
    expected_at: datetime | None = None
    run_id: UUID
    release_id: UUID | None = None
    comparison_run_id: UUID | None = None
    message: str
    values: dict[str, float | str | None] = Field(default_factory=dict)


class DragEvent(DragEventDraft):
    acknowledged_at: datetime | None = None
    acknowledged_by: int | None = None


class DragEventPage(DragModel):
    items: list[DragEvent]
    total: int
    unacknowledged: int
