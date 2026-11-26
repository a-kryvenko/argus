from datetime import datetime

from sqlalchemy import DateTime, Double, String
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base


class NormalizedObservation(Base):
    """Prepared hourly model inputs."""

    __tablename__ = "normalized_observation"

    observed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        primary_key=True,
    )
    bx: Mapped[float] = mapped_column(Double, nullable=False)
    by: Mapped[float] = mapped_column(Double, nullable=False)
    bz: Mapped[float] = mapped_column(Double, nullable=False)
    v: Mapped[float] = mapped_column(Double, nullable=False)
    n: Mapped[float] = mapped_column(Double, nullable=False)
    t: Mapped[float] = mapped_column(Double, nullable=False)
    kp: Mapped[float] = mapped_column(Double, nullable=False)
    dst: Mapped[float] = mapped_column(Double, nullable=False)
    ap: Mapped[float] = mapped_column(Double, nullable=False)
    f10_7: Mapped[float] = mapped_column(Double, nullable=False)
    s10: Mapped[float | None] = mapped_column(Double, nullable=True)
    m10: Mapped[float | None] = mapped_column(Double, nullable=True)
    y10: Mapped[float | None] = mapped_column(Double, nullable=True)
    wind_time_basis: Mapped[str | None] = mapped_column(String(16), nullable=True)
