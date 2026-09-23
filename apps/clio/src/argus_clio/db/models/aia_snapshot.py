from datetime import datetime
from sqlalchemy import DateTime,Float,String,Text
from sqlalchemy.orm import Mapped,mapped_column
from argus_clio.db.base import Base


class AIASnapshot(Base):
    __tablename__='aia_snapshot'
    slot_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),primary_key=True)
    observed_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),index=True)
    available_at: Mapped[datetime]=mapped_column(DateTime(timezone=True),index=True)
    sha256: Mapped[str]=mapped_column(String(64))
    raw_path: Mapped[str]=mapped_column(Text)
    cache_path: Mapped[str]=mapped_column(Text)
    b0_deg: Mapped[float]=mapped_column(Float)
    valid_fraction: Mapped[float]=mapped_column(Float)
    carrington_lon: Mapped[float]=mapped_column(Float)
