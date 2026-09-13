from datetime import datetime
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from argus_clio.db.base import Base


class SolarWindAggregate(Base):
    __tablename__ = 'solar_wind_aggregate'
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    resolution_seconds: Mapped[int] = mapped_column(Integer, primary_key=True)
    bucket_start: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    calculated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    statistics: Mapped[dict] = mapped_column(JSONB, nullable=False)


class SolarWindAggregatePending(Base):
    __tablename__ = 'solar_wind_aggregate_pending'
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    hour: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)


class SolarWindRetiredHour(Base):
    __tablename__ = 'solar_wind_retired_hour'
    kind: Mapped[str] = mapped_column(String(16), primary_key=True)
    hour: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    retired_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    raw_rows: Mapped[int] = mapped_column(Integer, nullable=False)
