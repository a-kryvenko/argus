from datetime import datetime
from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column
from argus_clio.db.base import Base


class ScheduledJob(Base):
    __tablename__ = 'scheduled_job'
    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    completed_slot: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
