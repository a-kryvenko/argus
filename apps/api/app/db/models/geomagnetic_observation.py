from datetime import datetime
from sqlalchemy import DateTime, Double, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class GeomagneticObservation(Base):
    __tablename__ = 'geomagnetic_observation'

    metric: Mapped[str] = mapped_column(String(16), primary_key=True)
    interval_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    interval_end: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[float | None] = mapped_column(Double, nullable=True)
    quality: Mapped[str] = mapped_column(String(16), nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw: Mapped[dict] = mapped_column(JSONB, nullable=False)
