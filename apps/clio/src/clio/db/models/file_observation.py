"""Observation context, S3 references and successful archive receipts."""
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base
from clio.db.models.file import File


class FileObservation(Base):
    __tablename__ = 'file_observation'
    __table_args__ = (
        Index('ix_file_observation_kind_received', 'kind', 'received_at'),
        Index('uq_file_observation_source', 'kind', 'source_key', unique=True),
        CheckConstraint('file_id IS NOT NULL', name='file_observation_success_file'),
        CheckConstraint('file_id IS NULL OR entity IS NOT NULL',
                        name='file_observation_context'),
        UniqueConstraint('file_id', 'variable', 'observed_at', 'entity', name='uq_file_observation_file_context'),
        Index('ix_file_observation_variable_observed', 'variable', 'observed_at'),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    kind: Mapped[str] = mapped_column(String(32))
    source_product: Mapped[str] = mapped_column(String(80))
    source_key: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    file_id: Mapped[UUID | None] = mapped_column(ForeignKey(File.id))
    variable: Mapped[str | None] = mapped_column(String(32))
    observed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    entity: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict | None] = mapped_column(JSON().with_variant(JSONB(), 'postgresql'))
