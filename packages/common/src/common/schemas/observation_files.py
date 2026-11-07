"""Immutable observation original metadata."""
from pydantic import AwareDatetime, BaseModel, Field
from common.schemas.files import FileReference


class RawObservationFile(BaseModel):
    observation: str = Field(min_length=1)
    observed_at: AwareDatetime
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    source_product: str
    file: FileReference


