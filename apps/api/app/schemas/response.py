from pydantic import BaseModel
from typing import Literal

from common.schemas.forecast import Forecast

class SuccessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    data: Forecast

class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: str