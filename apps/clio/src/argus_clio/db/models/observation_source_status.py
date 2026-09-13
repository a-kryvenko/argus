from datetime import datetime
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import Mapped, mapped_column
from argus_clio.db.base import Base


class ObservationSourceStatus(Base):
    __tablename__ = 'observation_source_status'

    source_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    last_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_response_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_error_code: Mapped[str | None] = mapped_column(String(32))
    last_error_message: Mapped[str | None] = mapped_column(String(160))
    consecutive_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latest_observation_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    latest_interval_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    data_quality: Mapped[str | None] = mapped_column(String(16))
