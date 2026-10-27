"""Receipts of source files; compressed originals and metadata live in the archive."""
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import CheckConstraint, DateTime, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base


class FileObservation(Base):
    __tablename__ = 'file_observation'
    __table_args__ = (
        Index('ix_file_observation_kind_received', 'kind', 'received_at'),
        Index('ix_file_observation_kind_sha256', 'kind', 'sha256'),
        Index('uq_file_observation_source', 'kind', 'source_key', unique=True,
              postgresql_where=text('error IS NULL')),
        CheckConstraint('(raw_path IS NULL) = (sha256 IS NULL)', name='file_observation_file_checksum'),
        CheckConstraint('error IS NOT NULL OR raw_path IS NOT NULL', name='file_observation_success_file'),
    )
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    kind: Mapped[str] = mapped_column(String(32))
    source_product: Mapped[str] = mapped_column(String(80))
    source_key: Mapped[str] = mapped_column(Text)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_path: Mapped[str | None] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
