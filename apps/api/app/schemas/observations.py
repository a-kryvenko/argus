"""Public observation contracts, independent of Clio's internal read models.

Units and coordinate frames are fixed by this API: wind v km/s, n cm^-3,
t K, bx/by/bz (GSM) and bt nT; Kp is an index, Dst nT. All times are UTC.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from app.schemas.metadata import metadata_field

Metric = Literal['bx', 'by', 'bz', 'bt', 'v', 'n', 't']
Index = Literal['kp', 'dst']
Quality = Literal['missing', 'flagged', 'unverified']
Freshness = Literal['missing', 'stale', 'fresh']


def optional_field():
    return Field(default=None, exclude_if=lambda value: value is None)


class SeriesMetadata(BaseModel):
    label: str
    unit: str
    source: str
    source_url: str
    coordinate_system: str | None = optional_field()
    location: str | None = optional_field()
    time_basis: str
    propagated: bool | None = optional_field()
    resolution_seconds: int
    aggregation: str | None = optional_field()
    selection: str | None = optional_field()
    poll_seconds: int | None = optional_field()
    freshness_basis: str | None = optional_field()
    gap_filling: str | None = optional_field()


class ObservationMetadata(BaseModel):
    series: dict[str, SeriesMetadata]
    interval: str | None = optional_field()
    selection: str | None = optional_field()
    gap_filling: str | None = optional_field()
    aggregation_version: int | None = optional_field()


class SolarSample(BaseModel):
    observed_at: datetime
    received_at: datetime
    value: float | None
    spacecraft: str
    quality: Quality
    interval_end: datetime | None = optional_field()
    min: float | None = optional_field()
    max: float | None = optional_field()
    count: int | None = optional_field()
    expected_count: int | None = optional_field()
    coverage_percent: float | None = optional_field()
    recalculation_pending: bool | None = optional_field()
    last_spacecraft: str | None = optional_field()
    source_changes: int | None = optional_field()


class IndexSample(BaseModel):
    interval_start: datetime
    interval_end: datetime
    interval_status: Literal['in_progress', 'completed']
    value: float | None
    quality: Quality
    received_at: datetime
    station_count: int | None


class SolarLatestSeries(BaseModel):
    latest: SolarSample | None
    age_seconds: int | None
    status: Freshness
    stale_after_seconds: int


class IndexLatestSeries(BaseModel):
    latest: IndexSample | None
    lag_seconds: int | None
    status: Freshness
    stale_after_seconds: int
    data_status: Literal['estimated', 'realtime']


class SolarLatest(BaseModel):
    """Native L1 measurements, not Earth arrival times. Fixed units: v km/s,
    n cm^-3, t K, bx/by/bz in GSM nT, bt nT. No propagation or gap filling.
    Quality and freshness are independent; a fresh sample may be flagged.
    """
    generated_at: datetime
    series: dict[Metric, SolarLatestSeries]
    meta: ObservationMetadata | None = metadata_field()


class IndexLatest(BaseModel):
    """Kp index (three-hour intervals) and Dst in nT (hourly intervals).
    Freshness is measured from interval end. Operational values may be revised.
    """
    generated_at: datetime
    series: dict[Index, IndexLatestSeries]
    meta: ObservationMetadata | None = metadata_field()


class TimeRange(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_: datetime = Field(alias='from')
    to: datetime


class Gap(TimeRange):
    reason: Literal['missing', 'invalid', 'partial']
    slots: int


class Coverage(BaseModel):
    expected_slots: int
    usable_slots: int
    missing_slots: int
    invalid_slots: int
    percent: float | None
    resolution_seconds: int
    evaluated_to: datetime
    basis: Literal['overlapping_intervals', 'sample_timestamps', 'available_aggregate_windows']
    gaps: list[Gap]


class Processing(BaseModel):
    unavailable_buckets: int
    recalculation_pending_buckets: int
    available_buckets: int
    expected_buckets: int


class SolarHistorySeries(BaseModel):
    points: list[SolarSample]
    coverage: Coverage
    processing: Processing | None = optional_field()


class IndexHistorySeries(BaseModel):
    points: list[IndexSample]
    coverage: Coverage
    data_status: Literal['estimated', 'realtime']


class SolarHistory(TimeRange):
    """Fixed units: v km/s, n cm^-3, t K, bx/by/bz GSM nT, bt nT.
    At 60 seconds value is the native sample; at 300/3600 seconds it is the
    mean of valid samples, with min/max and coverage over a closed UTC bucket.
    Missing timestamps are gaps; explicit missing values are null. No gap filling.
    """
    resolution_seconds: int = 60
    evaluated_from: datetime | None = optional_field()
    evaluated_to: datetime | None = optional_field()
    series: dict[Metric, SolarHistorySeries]
    meta: ObservationMetadata | None = metadata_field()


class IndexHistory(TimeRange):
    """Kp index and Dst in nT, with full source intervals overlapping [from,to).
    Kp retains three-hour intervals; Dst retains hourly intervals. No gap filling.
    """
    series: dict[Index, IndexHistorySeries]
    meta: ObservationMetadata | None = metadata_field()


class ChangeCoverage(BaseModel):
    recent: float
    previous: float
    last_hour: float


class DerivedValue(BaseModel):
    status: Literal['available', 'lower_bound', 'unavailable']
    value: float | None
    reason: str | None = optional_field()
    as_of: datetime | None = optional_field()
    coverage: ChangeCoverage | None = optional_field()


class ChangeMetadata(BaseModel):
    unit: str
    method: str | None = optional_field()
    minimum_coverage: float | None = optional_field()
    recent_mean: float | None = optional_field()
    previous_mean: float | None = optional_field()


class DurationMetadata(BaseModel):
    unit: Literal['sampled_minutes'] = 'sampled_minutes'


class SummaryMetadata(BaseModel):
    solar_wind: dict[Metric, SeriesMetadata]
    geomagnetic: dict[Index, SeriesMetadata]
    changes_1h: dict[Metric, ChangeMetadata]
    southward_bz: DurationMetadata = Field(default_factory=DurationMetadata)
    lookback_minutes: int


class ObservationSummary(BaseModel):
    """Native observations and trends in fixed units: v km/s, n cm^-3, t K,
    bx/by/bz GSM nT, bt nT, Kp index, Dst nT. changes_1h shares the variable's
    units. southward_bz.value counts sampled minutes, not continuous duration;
    lower_bound means onset predates the available window.
    """
    generated_at: datetime
    solar_wind: dict[Metric, SolarLatestSeries]
    geomagnetic: dict[Index, IndexLatestSeries]
    changes_1h: dict[Metric, DerivedValue]
    southward_bz: DerivedValue
    meta: SummaryMetadata | None = metadata_field()
