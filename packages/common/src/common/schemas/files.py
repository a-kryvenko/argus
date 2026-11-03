"""References to uploaded files; access URLs are generated at read time."""
from uuid import UUID

from pydantic import AwareDatetime, BaseModel


class FileReference(BaseModel):
    id: UUID
    timestamp: AwareDatetime
    bucket: str
    key: str


class FileObservationReference(BaseModel):
    id: UUID
    kind: str
    variable: str | None
    timestamp: AwareDatetime | None
    entity: str
    source_product: str
    source_key: str
    file: FileReference
