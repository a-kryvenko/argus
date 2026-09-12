"""Dashboard identities and bounded, anonymous API metric buckets."""
from datetime import datetime
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column
from app.db.base import Base


class User(Base):
    __tablename__ = 'dashboard_user'
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    active: Mapped[bool] = mapped_column(Boolean, default=True)


class Group(Base):
    __tablename__ = 'dashboard_group'
    name: Mapped[str] = mapped_column(String(80), primary_key=True)
    permissions: Mapped[list] = mapped_column(JSON)


class Membership(Base):
    __tablename__ = 'dashboard_membership'
    user_id: Mapped[int] = mapped_column(ForeignKey('dashboard_user.id', ondelete='CASCADE'), primary_key=True)
    group_name: Mapped[str] = mapped_column(ForeignKey('dashboard_group.name', ondelete='CASCADE'), primary_key=True)


class Session(Base):
    __tablename__ = 'dashboard_session'
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey('dashboard_user.id', ondelete='CASCADE'), index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class LoginAttempt(Base):
    __tablename__ = 'dashboard_login_attempt'
    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    window: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    count: Mapped[int] = mapped_column(Integer)


class ApiMetric(Base):
    __tablename__ = 'api_metric'
    hour: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    route: Mapped[str] = mapped_column(String(256), primary_key=True)
    method: Mapped[str] = mapped_column(String(16), primary_key=True)
    status: Mapped[int] = mapped_column(Integer, primary_key=True)
    bucket_ms: Mapped[int] = mapped_column(Integer, primary_key=True)
    count: Mapped[int] = mapped_column(Integer)
    duration_ms: Mapped[float] = mapped_column(Float)
