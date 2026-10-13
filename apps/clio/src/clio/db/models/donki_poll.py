"""Append-only receipts, including empty responses, errors and exact source bytes."""
from datetime import datetime
from uuid import UUID, uuid4

from sqlalchemy import DateTime, Index, LargeBinary, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base


class DonkiPoll(Base):
    __tablename__ = 'donki_poll'
    __table_args__ = (Index('ix_donki_poll_endpoint_received', 'endpoint', 'received_at'),)
    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    endpoint: Mapped[str] = mapped_column(String(32))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    request_start: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    request_end: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str | None] = mapped_column(String(64))
    raw_gzip: Mapped[bytes | None] = mapped_column(LargeBinary, deferred=True)
    payload: Mapped[list | None] = mapped_column(JSONB, deferred=True)
    error: Mapped[str | None] = mapped_column(Text)
