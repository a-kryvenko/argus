from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ObservationPoint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    issue_time: datetime
    bx: float
    by: float
    bz: float
    v: float
    n: float
    t: float
    kp: int
    dst: int
    ap: int
    f10_7: int

class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: list[ObservationPoint]
