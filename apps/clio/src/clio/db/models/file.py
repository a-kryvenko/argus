"""Uploaded object references."""
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Index, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base


class File(Base):
    __tablename__ = 'files'
    __table_args__ = (
        UniqueConstraint('bucket', 'key', name='uq_files_bucket_key'),
        Index('ix_files_timestamp', 'timestamp'),
    )

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    bucket: Mapped[str] = mapped_column(String(63))
    key: Mapped[str] = mapped_column(Text)
