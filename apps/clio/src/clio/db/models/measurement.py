from datetime import datetime, timedelta

from sqlalchemy import (
    BigInteger,
    DateTime,
    Double,
    Identity,
    Index,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base


REQUIRED_METRICS = ('bx', 'by', 'bz', 'v', 'n', 't', 'kp', 'dst', 'ap', 'f10_7')
SOLAR_INDEX_METRICS = ('s10', 'm10', 'y10')
OBSERVATION_METRICS = (*REQUIRED_METRICS, *SOLAR_INDEX_METRICS)

INTERVALS = {'kp': timedelta(hours=3), 'ap': timedelta(hours=3), 'dst': timedelta(hours=1)}


class Measurement(Base):
    __tablename__ = "measurement"
    __table_args__ = (
        UniqueConstraint(
            "metric",
            "observed_at",
            "source_product",
            name="uq_measurement_metric",
            postgresql_nulls_not_distinct=True,
        ),
        Index("ix_measurement_observed_at", "observed_at"),
        Index("ix_measurement_metric_source_time", "metric", "source_product", "observed_at"),
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
    quality: Mapped[str | None] = mapped_column(String(16), nullable=True)
    spacecraft: Mapped[str | None] = mapped_column(String(32), nullable=True)
