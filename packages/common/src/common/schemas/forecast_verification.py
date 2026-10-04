"""Compact read contract for rolling verification of published forecasts."""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class VerificationLead(BaseModel):
    lead_hours: int
    counts: dict[str, int]
    continuous: dict[str, float] | None = None
    binary: dict[str, dict[str, float | None]] = Field(default_factory=dict)


class VerificationGroup(BaseModel):
    artifact: str
    releases: int
    evaluated_at: datetime
    counts: dict[str, int]
    by_lead_hour: list[VerificationLead] = Field(default_factory=list)


class ForecastVerification(BaseModel):
    product: str
    protocol: Literal['published-hourly-v1'] = 'published-hourly-v1'
    days: int = 30
    start: datetime
    end: datetime
    groups: list[VerificationGroup] = Field(default_factory=list)
