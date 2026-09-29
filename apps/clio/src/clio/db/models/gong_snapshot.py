from datetime import datetime

from sqlalchemy import DateTime, LargeBinary, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from clio.db.base import Base


class GONGSnapshot(Base):
    __tablename__ = 'gong_snapshot'
    slot_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    sha256: Mapped[str] = mapped_column(String(64))
    source_product: Mapped[str] = mapped_column(String(80))
    source_url: Mapped[str] = mapped_column(Text)
    feature_version: Mapped[str] = mapped_column(String(40))
    features: Mapped[dict] = mapped_column(JSONB)
    fits_gzip: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
