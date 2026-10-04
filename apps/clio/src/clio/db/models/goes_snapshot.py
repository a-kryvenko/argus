from datetime import datetime
from sqlalchemy import DateTime, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from clio.db.base import Base


class GOESSnapshot(Base):
    __tablename__ = 'goes_snapshot'
    slot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sha256: Mapped[str] = mapped_column(String(64))
    source_product: Mapped[str] = mapped_column(String(80), primary_key=True)
    raw_path: Mapped[str] = mapped_column(Text)
