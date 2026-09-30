"""References to individual image observations on shared service storage."""
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field


class SDOImageReference(BaseModel):
    path: str = Field(description='Absolute NPZ path in the shared service filesystem')
    shape: tuple[Literal[512], Literal[512]] = (512, 512)
    dtype: Literal['float32'] = 'float32'
    metadata: dict[str, Any] = Field(description='Observation receipt; sha256 identifies the original FITS, not the NPZ')


class MissingSDOImage(BaseModel):
    slot_at: AwareDatetime
    channel: str


class SDOImageCatalog(BaseModel):
    start: AwareDatetime
    end: AwareDatetime = Field(description='Exclusive UTC hour boundary')
    as_of: AwareDatetime
    read_at: AwareDatetime
    items: list[SDOImageReference]
    missing: list[MissingSDOImage] = Field(description='Absent observations or observations unavailable at as_of')
