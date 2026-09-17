"""Shared project state and anonymous traffic aggregates."""
from datetime import datetime
from sqlalchemy import DateTime, String, Integer, Float, JSON, BigInteger
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base

class MonitorState(Base):
    __tablename__ = 'monitor_state'
    name: Mapped[str] = mapped_column(String(32), primary_key=True)
    checked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict] = mapped_column(JSON)

class TrafficMetric(Base):
    __tablename__ = 'traffic_metric'
    time: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    resolution: Mapped[str] = mapped_column(String(8), primary_key=True)
    channel: Mapped[str] = mapped_column(String(8), primary_key=True)
    status: Mapped[int] = mapped_column(Integer, primary_key=True)
    bucket_ms: Mapped[int] = mapped_column(Integer, primary_key=True)
    count: Mapped[int] = mapped_column(BigInteger)
    duration_ms: Mapped[float] = mapped_column(Float)
