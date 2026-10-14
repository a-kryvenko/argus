"""Causal source snapshots and experimental solar-wind event forecasts."""
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, FiniteFloat, model_validator

DonkiEndpoint = Literal['CME', 'WSAEnlilSimulations', 'HSS', 'IPS']


class SourcePoll(BaseModel):
    endpoint: DonkiEndpoint
    poll_id: UUID
    received_at: AwareDatetime
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    request_start: AwareDatetime
    request_end: AwareDatetime
    records: list[dict]


class L1Sample(BaseModel):
    observed_at: AwareDatetime
    available_at: AwareDatetime
    metric: Literal['bx', 'by', 'bz', 'v', 'n', 't']
    value: FiniteFloat | None = None
    quality: str | None = None
    spacecraft: str | None = None


class HourlyWindSpeed(BaseModel):
    observed_at: AwareDatetime
    interval_end: AwareDatetime
    available_at: AwareDatetime
    value: FiniteFloat = Field(gt=0, lt=3000)
    source_product: str
    sample_count: int = Field(ge=1, le=60)


class CoronalHoleRegion(BaseModel):
    region_id: str
    longitude_deg: FiniteFloat = Field(ge=-60, le=60)
    west_edge_deg: FiniteFloat = Field(ge=-60, le=60)
    east_edge_deg: FiniteFloat = Field(ge=-60, le=60)
    latitude_deg: FiniteFloat = Field(ge=-60, le=60)
    area_fraction: FiniteFloat = Field(gt=0, le=1)
    earth_band_fraction: FiniteFloat = Field(ge=0, le=1)
    magnetic_unipolarity: FiniteFloat | None = Field(default=None, ge=0, le=1)
    mean_radial_field_gauss: FiniteFloat | None = None
    classification: Literal['euv_dark_region', 'magnetically_supported_candidate']


class CoronalHoleFrame(BaseModel):
    feature_version: Literal['euv-dark-unipolar-v1'] = 'euv-dark-unipolar-v1'
    observed_at: AwareDatetime
    available_at: AwareDatetime
    b0_deg: FiniteFloat
    source_receipts: list[dict]
    regions: list[CoronalHoleRegion]
    magnetic_state: Literal['available', 'unavailable', 'rejected']


class HSSSpeedPoint(BaseModel):
    valid_time: AwareDatetime
    speed_km_s: FiniteFloat = Field(gt=0, lt=3000)
    source: Literal['PROSWIN', 'rotation']


class HSSForecast(BaseModel):
    model_config = ConfigDict(extra='forbid')
    forecast_id: str
    kind: Literal['hss_candidate'] = 'hss_candidate'
    status: Literal['expected', 'ongoing_candidate']
    timing_method: Literal['proswin_profile', 'rotation_profile', 'coronal_hole_transport']
    arrival_at: AwareDatetime
    arrival_window_start: AwareDatetime
    arrival_window_end: AwareDatetime
    window_kind: Literal['uncalibrated_timing_envelope', 'heuristic_transport_range']
    peak_at: AwareDatetime | None = None
    peak_speed_km_s: FiniteFloat | None = Field(default=None, gt=0, lt=3000)
    end_at: AwareDatetime | None = None
    onset_left_censored: bool = False
    end_right_censored: bool = False
    evidence_sources: list[str]
    coronal_hole_region_ids: list[str] = Field(default_factory=list)
    magnetic_support: bool = False
    recurrence_support: bool = False
    cme_overlap_ids: list[str] = Field(default_factory=list)
    attribution: Literal['candidate', 'ambiguous_cme_overlap'] = 'candidate'

    @model_validator(mode='after')
    def ordered_times(self):
        if not self.arrival_window_start <= self.arrival_at <= self.arrival_window_end:
            raise ValueError('HSS arrival must be inside its timing envelope')
        if self.peak_at is not None and self.peak_at < self.arrival_at:
            raise ValueError('HSS peak precedes arrival')
        if self.end_at is not None and self.end_at <= self.arrival_at:
            raise ValueError('HSS end must follow arrival')
        return self


class SolarWindEvent(BaseModel):
    model_config = ConfigDict(extra='forbid')
    event_id: str
    kind: Literal['cme', 'hss', 'interplanetary_shock', 'solar_wind_disturbance']
    status: Literal['detected', 'predicted_hit', 'no_predicted_hit', 'prediction_elapsed', 'observed']
    source: Literal['DONKI', 'DONKI/WSA-ENLIL']
    source_ids: list[str] = Field(default_factory=list)
    source_url: str | None = None
    available_at: AwareDatetime
    observed_at: AwareDatetime | None = None
    model_completed_at: AwareDatetime | None = None
    arrival_at: AwareDatetime | None = None
    arrival_target: Literal['Earth'] | None = None
    duration_hours: FiniteFloat | None = Field(default=None, gt=0)
    glancing_blow: bool | None = None
    minor_impact: bool | None = None
    simulation_id: str | None = None
    initial_speed_km_s: FiniteFloat | None = Field(default=None, gt=0)


class EnlilProfilePoint(BaseModel):
    valid_time: AwareDatetime
    speed_km_s: FiniteFloat = Field(ge=0)
    density_cm3: FiniteFloat = Field(ge=0)


class EnlilProfile(BaseModel):
    provider: Literal['NOAA/SWPC ENLIL'] = 'NOAA/SWPC ENLIL'
    receipt_id: UUID
    source_key: str
    source_url: str
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    received_at: AwareDatetime
    source_modified_at: AwareDatetime
    model_created_at: AwareDatetime
    model_version: str
    target: Literal['L1'] = 'L1'
    density_kind: Literal['proton_equivalent'] = 'proton_equivalent'
    points: list[EnlilProfilePoint]

    @model_validator(mode='after')
    def ordered_points(self):
        times = [p.valid_time for p in self.points]
        if not times or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError('ENLIL profile times must be nonempty, unique and increasing')
        return self


class EventInputs(BaseModel):
    schema_version: Literal[1] = 1
    as_of: AwareDatetime
    read_at: AwareDatetime
    polls: list[SourcePoll] = Field(default_factory=list)
    events: list[SolarWindEvent] | None = None
    enlil_profiles: list[EnlilProfile] = Field(default_factory=list)
    enlil_profile_status: dict = Field(default_factory=lambda: {'state': 'unavailable'})
    latest_attempts: dict[str, dict] = Field(default_factory=dict)
    l1_samples: list[L1Sample] = Field(default_factory=list)
    speed_hourly: list[HourlyWindSpeed] = Field(default_factory=list)


class L1Detection(BaseModel):
    detection_id: str
    kind: Literal['compression_candidate'] = 'compression_candidate'
    onset_at: AwareDatetime
    detected_at: AwareDatetime
    spacecraft: dict[str, str]
    evidence: dict[str, FiniteFloat]
    candidate_event_ids: list[str] = Field(default_factory=list)
    attribution: Literal['unresolved'] = 'unresolved'


class CMEEnsembleForecast(BaseModel):
    """Conditional model spread, never a calibrated operational probability."""
    model_config = ConfigDict(extra='forbid')
    forecast_id: str
    model_version: Literal['argus-dbem-v1'] = 'argus-dbem-v1'
    cme_id: str
    source_poll_id: UUID
    source_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    analysis_sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    available_at: AwareDatetime
    initial_time: AwareDatetime
    initial_parameters: dict[str, FiniteFloat]
    policy: dict
    seed: int = Field(ge=0)
    ensemble_size: int = Field(gt=0)
    geometric_hit_count: int = Field(ge=0)
    geometric_hit_fraction: FiniteFloat = Field(ge=0, le=1)
    arrival_count: int = Field(ge=0)
    right_censored_count: int = Field(ge=0)
    future_arrival_count: int = Field(ge=0)
    arrival_within_horizon_count: int = Field(ge=0)
    status: Literal['geometric_miss', 'right_censored', 'expected', 'elapsed', 'timing_straddles_issue']
    arrival_feature: Literal['cme_leading_edge'] = 'cme_leading_edge'
    quantile_kind: Literal['uncalibrated_conditional_ensemble'] = 'uncalibrated_conditional_ensemble'
    calibrated: Literal[False] = False
    arrival_q10: AwareDatetime | None = None
    arrival_q50: AwareDatetime | None = None
    arrival_q90: AwareDatetime | None = None
    speed_q10_km_s: FiniteFloat | None = Field(default=None, gt=0)
    speed_q50_km_s: FiniteFloat | None = Field(default=None, gt=0)
    speed_q90_km_s: FiniteFloat | None = Field(default=None, gt=0)
    earth_distance_at_median_km: FiniteFloat | None = Field(default=None, gt=0)
    potential_interaction_ids: list[str] = Field(default_factory=list)

    @model_validator(mode='after')
    def consistent_ensemble(self):
        if (self.arrival_count + self.right_censored_count != self.geometric_hit_count or
            self.geometric_hit_count > self.ensemble_size or
            self.future_arrival_count > self.arrival_count or
            self.arrival_within_horizon_count > self.future_arrival_count or
            abs(self.geometric_hit_fraction-self.geometric_hit_count/self.ensemble_size) > 1e-12):
            raise ValueError('Inconsistent CME ensemble counts')
        times = (self.arrival_q10, self.arrival_q50, self.arrival_q90)
        speeds = (self.speed_q10_km_s, self.speed_q50_km_s, self.speed_q90_km_s)
        if self.arrival_count:
            if (any(t is None for t in times+speeds) or not times[0] <= times[1] <= times[2] or
                not speeds[0] <= speeds[1] <= speeds[2] or self.earth_distance_at_median_km is None):
                raise ValueError('Missing or unordered CME arrival quantiles')
            if times[0] < self.initial_time:
                raise ValueError('CME arrival precedes initialization')
        elif any(t is not None for t in times+speeds) or self.earth_distance_at_median_km is not None:
            raise ValueError('CME without arrivals cannot have arrival quantiles')
        expected = ('geometric_miss' if not self.geometric_hit_count else
                    'right_censored' if not self.arrival_count else
                    'elapsed' if not self.future_arrival_count else
                    'expected' if self.future_arrival_count == self.arrival_count else 'timing_straddles_issue')
        if self.status != expected:
            raise ValueError('CME ensemble status does not match counts')
        return self


class CMESpeedPoint(BaseModel):
    valid_time: AwareDatetime
    speed_background: FiniteFloat = Field(ge=0)
    speed_nominal: FiniteFloat = Field(ge=0)
    speed_early: FiniteFloat = Field(ge=0)
    speed_late: FiniteFloat = Field(ge=0)
    scenario_lower: FiniteFloat = Field(ge=0)
    scenario_upper: FiniteFloat = Field(ge=0)

    @model_validator(mode='after')
    def ordered_scenarios(self):
        values=(self.speed_nominal,self.speed_early,self.speed_late)
        if not self.scenario_lower <= min(values) <= max(values) <= self.scenario_upper:
            raise ValueError('Speed scenarios must lie inside the envelope')
        return self


class CMESpeedOutlook(BaseModel):
    model_version: str
    as_of: AwareDatetime | None = None
    state: Literal['available', 'partial', 'unavailable']
    conditional_on_impact: Literal[True] = True
    units: Literal['km/s'] = 'km/s'
    scenario_kind: Literal['conditional_scenarios_not_quantiles'] = 'conditional_scenarios_not_quantiles'
    background_model: dict = Field(default_factory=dict)
    events: list[dict] = Field(default_factory=list)
    excluded_events: list[dict] = Field(default_factory=list)
    points: list[CMESpeedPoint] = Field(default_factory=list)
    reason: str | None = None
    model_sha256: str | None = Field(default=None, pattern=r'^[0-9a-f]{64}$')

    @model_validator(mode='after')
    def ordered_points(self):
        times=[p.valid_time for p in self.points]
        if any(a >= b for a,b in zip(times,times[1:])):
            raise ValueError('CME speed outlook times must be unique and increasing')
        if self.state == 'unavailable' and self.points:
            raise ValueError('Unavailable speed outlook cannot carry predictions')
        return self


class SolarWindEventForecast(BaseModel):
    schema_version: Literal[1] = 1
    product: Literal['solar-wind-events'] = 'solar-wind-events'
    model_version: Literal['solar-wind-events-v1', 'solar-wind-events-v2', 'solar-wind-events-v3', 'solar-wind-events-v4'] = 'solar-wind-events-v1'
    issue_time: AwareDatetime
    horizon_hours: Literal[168] = 168
    experimental: Literal[True] = True
    source_status: dict[str, dict]
    events: list[SolarWindEvent]
    l1_detections: list[L1Detection]
    hss_forecasts: list[HSSForecast] = Field(default_factory=list)
    hss_speed_profile: list[HSSSpeedPoint] = Field(default_factory=list)
    coronal_holes: CoronalHoleFrame | None = None
    cme_ensembles: list[CMEEnsembleForecast] = Field(default_factory=list)
    cme_speed: CMESpeedOutlook | None = None
    limitations: list[str] = Field(default_factory=lambda: [
        'External ENLIL predictions have no locally calibrated hit probability or arrival quantiles.',
        'DONKI HSS records describe observations, not an advance HSS forecast.',
        'The L1 detector flags sustained compression candidates; it does not identify every shock or gradual HSS arrival.',
        'Temporal association does not establish CME/HSS attribution; Bz and storm strength are not predicted.',
    ])


    @model_validator(mode='after')
    def future_speed_points(self):
        if self.cme_speed and any(p.valid_time <= self.issue_time for p in self.cme_speed.points):
            raise ValueError('Speed forecast points must follow the release issue time')
        return self
