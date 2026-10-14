"""Immutable receipts of compact external NOAA ENLIL profiles and failures."""
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Index, LargeBinary, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base


class EnlilProfileReceipt(Base):
    __tablename__ = 'enlil_profile_receipt'
    __table_args__ = (Index('ix_enlil_profile_received', 'received_at'),)
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str] = mapped_column(Text)
    source_key: Mapped[str | None] = mapped_column(Text)
    source_modified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    model_created_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sha256: Mapped[str | None] = mapped_column(String(64))
    raw_gzip: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    payload: Mapped[dict | None] = mapped_column(JSONB, deferred=True)
    error: Mapped[str | None] = mapped_column(Text)
