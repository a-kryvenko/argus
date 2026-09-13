"""Diagnostic Prophet status contract; it does not authorize forecast use."""
from typing import Literal
from uuid import UUID

from pydantic import AwareDatetime, BaseModel


class CurrentReleaseStatus(BaseModel):
    release_id: UUID
    run_id: UUID
    issue_time: AwareDatetime
    published_at: AwareDatetime
    started_at: AwareDatetime
    input_diagnostics: dict | None = None


class ForecastAttemptStatus(BaseModel):
    run_id: UUID
    status: Literal['running', 'succeeded', 'partial', 'failed', 'interrupted']
    started_at: AwareDatetime
    finished_at: AwareDatetime | None = None
    error: str | None = None
    input_diagnostics: dict | None = None


class AttemptArtifactStatus(BaseModel):
    name: str
    status: Literal['stored', 'skipped']
    error: str | None = None


class ForecastStatus(BaseModel):
    contract_version: Literal[1] = 1
    product: str
    assessed_at: AwareDatetime
    mode: Literal['observe'] = 'observe'
    current_release: CurrentReleaseStatus | None
    release_age_hours: float | None
    existing_public_max_age_hours: float | None
    freshness: Literal['unavailable', 'future_issue_time', 'unconfigured', 'stale', 'within_age_limit']
    latest_attempt: ForecastAttemptStatus | None
    latest_attempt_artifacts: list[AttemptArtifactStatus]
    note: str = 'Diagnostic status does not establish model readiness or individual sensor freshness.'
