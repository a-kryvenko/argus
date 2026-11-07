"""Completed original-file observations exposed by the read API."""
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, ForeignKey, Index, JSON, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base
from clio.db.models.file import File


class FileObservation(Base):
    __tablename__ = 'file_observation'
    __table_args__ = (
        Index('ix_file_observation_observation_observed', 'observation', 'observed_at'),
        UniqueConstraint('observation', 'file_id', name='uq_file_observation_original'),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    observation: Mapped[str] = mapped_column(String(64))
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_product: Mapped[str] = mapped_column(String(80))
    file_id: Mapped[UUID] = mapped_column(ForeignKey(File.id))
    sha256: Mapped[str] = mapped_column(String(64))
    details: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), 'postgresql'))
