"""Immutable observation original metadata."""
from typing import Literal

from pydantic import AwareDatetime, BaseModel, Field


class RawObservationFile(BaseModel):
    kind: Literal['aia', 'gong', 'goes']
    slot_at: AwareDatetime
    observed_at: AwareDatetime
    available_at: AwareDatetime
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_product: str


