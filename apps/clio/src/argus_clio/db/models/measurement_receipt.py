from datetime import datetime
from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column
from argus_clio.db.base import Base

class MeasurementReceipt(Base):
    __tablename__ = 'measurement_receipt'
    metric: Mapped[str] = mapped_column(String(16), primary_key=True)
    latest_observation_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
