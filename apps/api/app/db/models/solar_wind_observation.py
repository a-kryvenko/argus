from datetime import datetime

from sqlalchemy import Boolean, DateTime, Index, String, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SolarWindObservation(Base):
    """Native minute samples; separate from propagated, gap-filled model inputs."""

    __tablename__ = "solar_wind_observation"
    __table_args__ = (Index("ix_solar_wind_active_time", "kind", "observed_at", postgresql_where=text("active")),)

    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    spacecraft: Mapped[str] = mapped_column(String(32), primary_key=True)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    values: Mapped[dict] = mapped_column(JSONB, nullable=False)
    raw: Mapped[dict] = mapped_column(JSONB, nullable=False)
