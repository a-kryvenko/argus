"""Transport contracts for density exposure advisories; no assessment policy."""
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class Contract(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False, str_strip_whitespace=True)


Attention = Literal['not_assessed', 'within_reference', 'watch', 'review', 'high']
DataStatus = Literal['current', 'partial', 'unavailable', 'stale', 'trajectory_expired', 'profile_changed', 'paused']


class TrajectoryPoint(Contract):
    time: AwareDatetime
    altitude_km: float = Field(ge=90, le=2500)
    latitude_deg: float = Field(ge=-90, le=90)
    longitude_deg: float = Field(ge=-180, le=180)
    reference_density_kg_m3: float | None = Field(default=None, gt=0)


class Trajectory(Contract):
    source: str = Field(min_length=1, max_length=300)
    generated_at: AwareDatetime
    valid_until: AwareDatetime
    coordinates: Literal['geocentric_spherical']
    points: list[TrajectoryPoint] = Field(min_length=2, max_length=5000)


class DensityReference(Contract):
    label: str = Field(min_length=1, max_length=120)
    source: str = Field(min_length=1, max_length=300)
    prepared_at: AwareDatetime
    kind: Literal['operator_reference', 'model_geomagnetic_reference'] = 'operator_reference'
    spatial_support: Literal['longitude_mean'] = 'longitude_mean'
    method: str | None = None


class AdvisoryPolicy(Contract):
    review_ratio: float = Field(default=2, gt=1, le=100)
    high_ratio: float = Field(default=4, gt=1, le=100)
    max_sample_gap_minutes: int = Field(default=15, ge=1, le=60)


class AdvisoryProfileInput(Contract):
    name: str = Field(min_length=1, max_length=120)
    trajectory: Trajectory
    reference: DensityReference | None = None
    reference_mode: Literal['auto', 'prophet_background', 'operator_reference'] = 'auto'
    policy: AdvisoryPolicy = Field(default_factory=AdvisoryPolicy)
    enabled: bool = True


class AdvisoryProfile(AdvisoryProfileInput):
    id: UUID
    revision: int = Field(ge=1)
    updated_at: AwareDatetime


class AdvisoryReason(Contract):
    code: str
    message: str
    source_id: str | None = None


class AdvisorySample(Contract):
    time: AwareDatetime
    altitude_km: float
    latitude_deg: float
    longitude_deg: float
    density_kg_m3: float | None = None
    reference_density_kg_m3: float | None = None
    ratio: float | None = None
    attention: Attention
    reasons: list[AdvisoryReason]


class AdvisoryLocation(Contract):
    time: AwareDatetime
    altitude_km: float
    latitude_deg: float
    longitude_deg: float


class AdvisoryInterval(Contract):
    start: AwareDatetime
    end: AwareDatetime
    attention: Attention
    peak_ratio: float | None = None
    peak_density_kg_m3: float | None = None
    sample_count: int = Field(ge=0)
    reasons: list[AdvisoryReason]
    sampled_duration_minutes: float = Field(default=0, ge=0)
    peak_location: AdvisoryLocation | None = None


class AdvisorySource(Contract):
    density_release_id: UUID | None = None
    density_issue_time: AwareDatetime | None = None
    density_observed_at: AwareDatetime | None = None
    density_model: str | None = None
    density_sha256: str | None = None
    cme_release_id: UUID | None = None
    cme_issue_time: AwareDatetime | None = None
    cme_sha256: str | None = None
    cme_status: Literal['current', 'stale', 'unavailable']
    trajectory_source: str
    trajectory_generated_at: AwareDatetime
    reference: DensityReference | None = None


class DensityAdvisory(Contract):
    computed_at: AwareDatetime
    attention: Attention
    data_status: DataStatus
    source: AdvisorySource
    samples: list[AdvisorySample]
    intervals: list[AdvisoryInterval]
    limitations: list[str]


class AdvisoryRun(Contract):
    id: UUID
    profile_id: UUID
    profile_revision: int
    profile_snapshot: AdvisoryProfileInput
    assessment: DensityAdvisory


class AdvisoryProfileSummary(Contract):
    id: UUID
    name: str
    revision: int
    updated_at: AwareDatetime
    enabled: bool


class AdvisoryRunSummary(Contract):
    id: UUID
    profile_revision: int
    computed_at: AwareDatetime


class AdvisoryItem(Contract):
    profile: AdvisoryProfileSummary
    latest: AdvisoryRunSummary | None = None
    attention: Attention
    data_status: DataStatus
    cme_status: Literal['current', 'stale', 'unavailable']


class AdvisoryEvent(Contract):
    id: UUID
    profile_id: UUID
    run_id: UUID
    detected_at: AwareDatetime
    kind: Literal['attention_changed', 'data_status_changed']
    attention: Attention
    message: str
    acknowledged_at: AwareDatetime | None = None


class AdvisoryEventPage(Contract):
    items: list[AdvisoryEvent]
    total: int
