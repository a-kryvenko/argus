"""Latest completed poll attempt per DONKI endpoint."""
from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base


class DonkiPollStatus(Base):
    __tablename__ = 'donki_poll_status'

    endpoint: Mapped[str] = mapped_column(String(32), primary_key=True)
    attempted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(String(128))
