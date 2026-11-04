"""References to individual image observations in object storage."""
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, Field
from common.schemas.files import FileReference


OBSERVED_CHANNELS = (
    'aia94',
    'aia131',
    'aia171',
    'aia193',
    'aia211',
    'aia304',
    'aia335',
    'aia1600',
    'hmi_m'
)


class SDOImageReference(BaseModel):
    file: FileReference
    original: FileReference | None = None
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
