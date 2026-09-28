"""Public contract for the circular LEO drag screening service."""
from datetime import date, timedelta
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator


class DragModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)


class DragThresholds(DragModel):
    elevated_altitude_loss_m: float = Field(gt=0)
    high_altitude_loss_m: float = Field(gt=0)

    @model_validator(mode='after')
    def ordered(self):
        if self.high_altitude_loss_m <= self.elevated_altitude_loss_m:
            raise ValueError('High threshold must exceed elevated threshold')
        return self


class LeoDragRequest(DragModel):
    altitude_km: float = Field(ge=200, le=800)
    inclination_deg: float = Field(ge=0, le=180)
    mass_kg: float = Field(ge=0.01, le=1e7)
    effective_area_m2: float = Field(gt=0, le=1e6)
    drag_coefficient: float = Field(gt=0, le=10)
    horizon_hours: Literal[24, 48] = 24
    thresholds: DragThresholds | None = Field(default=None, description=(
        'User-defined cumulative altitude-loss thresholds over the requested horizon. '
        'Without thresholds, drag_risk is not_assessed.'))


class DragPoint(DragModel):
    valid_time: AwareDatetime
    lead_hours: int = Field(ge=0, le=48)
    mean_density_kg_m3: float = Field(gt=0)
    mean_drag_accel_m_s2: float = Field(ge=0)
    mean_along_track_deceleration_m_s2: float = Field(ge=0)
    delta_v_loss_m_s: float = Field(ge=0, description='Cumulative along-track drag impulse per unit mass.')
    estimated_altitude_loss_m: float = Field(ge=0)


class DragSource(DragModel):
    release_id: UUID
    issue_time: AwareDatetime
    observed_at: AwareDatetime
    dtc_observed_at: AwareDatetime
    history_start: AwareDatetime
    background_method: str
    background_interpolated_days: dict[str, list[date]]
    dtc_method: Literal['causal_dst_ap_v1']
    density_model: Literal['JB2008'] = 'JB2008'
    driver_mode: Literal['observed_persistence'] = 'observed_persistence'


class LeoDragAssessment(DragModel):
    model: Literal['circular_leo_drag_v1'] = 'circular_leo_drag_v1'
    computed_at: AwareDatetime
    start_time: AwareDatetime
    end_time: AwareDatetime
    inputs: LeoDragRequest
    source: DragSource
    mean_density_kg_m3: float = Field(gt=0)
    mean_drag_accel_m_s2: float = Field(ge=0)
    delta_v_loss_m_s: float = Field(ge=0)
    estimated_altitude_loss_m: float = Field(ge=0)
    drag_risk: Literal['not_assessed', 'low', 'elevated', 'high']
    risk_reason: str
    assumptions: list[str]
    predictions: list[DragPoint]

    @model_validator(mode='after')
    def consistent_series(self):
        horizon = self.inputs.horizon_hours
        if (self.start_time != self.source.issue_time or self.start_time > self.computed_at
                or self.end_time != self.start_time + timedelta(hours=horizon)
                or len(self.predictions) != horizon + 1):
            raise ValueError('Inconsistent assessment horizon')
        previous_dv = previous_loss = 0.0
        for lead, point in enumerate(self.predictions):
            if (point.lead_hours != lead or point.valid_time != self.start_time + timedelta(hours=lead)
                    or point.delta_v_loss_m_s < previous_dv or point.estimated_altitude_loss_m < previous_loss):
                raise ValueError('Inconsistent assessment series')
            previous_dv, previous_loss = point.delta_v_loss_m_s, point.estimated_altitude_loss_m
        first, last = self.predictions[0], self.predictions[-1]
        if (first.delta_v_loss_m_s != 0 or first.estimated_altitude_loss_m != 0
                or last.delta_v_loss_m_s != self.delta_v_loss_m_s
                or last.estimated_altitude_loss_m != self.estimated_altitude_loss_m
                or self.estimated_altitude_loss_m > 1000):
            raise ValueError('Inconsistent cumulative loss')
        thresholds = self.inputs.thresholds
        expected = 'not_assessed'
        if thresholds is not None:
            expected = ('high' if self.estimated_altitude_loss_m >= thresholds.high_altitude_loss_m else
                        'elevated' if self.estimated_altitude_loss_m >= thresholds.elevated_altitude_loss_m else 'low')
        if self.drag_risk != expected:
            raise ValueError('Risk does not match the supplied thresholds')
        return self
