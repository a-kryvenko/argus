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
    raw_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    feature_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    features: Mapped[dict | None] = mapped_column(JSONB, nullable=True, deferred=True)
    fits_gzip: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True, deferred=True)
