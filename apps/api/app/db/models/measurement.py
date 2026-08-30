from datetime import datetime

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

from app.db.base import Base


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
    value: Mapped[float] = mapped_column(Double, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )
