from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Double,
    Identity,
    Integer,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base


class Measurement(Base):
    __tablename__ = "measurement"
    __table_args__ = (
        UniqueConstraint(
            "metric",
            "observed_at",
            name="uq_measurement_metric",
        ),
        Index("ix_measurement_observed_at", "observed_at"),
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        Identity(always=True),
        primary_key=True,
    )
    metric: Mapped[str] = mapped_column(String(16), nullable=False)
    value: Mapped[float | None] = mapped_column(Double, nullable=True)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
    source_product: Mapped[str | None] = mapped_column(String(80), nullable=True)
    received_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    interval_end: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    quality: Mapped[str | None] = mapped_column(String(16))
    spacecraft: Mapped[str | None] = mapped_column(String(32))
    provider_quality: Mapped[int | None] = mapped_column(Integer)
    station_count: Mapped[int | None] = mapped_column(Integer)
