from pydantic import BaseModel
from typing import Literal

from common.schemas.forecast import Forecast

class SuccessForecastResponse(BaseModel):
    status: Literal["ok"] = "ok"
    data: Forecast

class SuccessResponse(BaseModel):
    status: Literal["ok"] = "ok"
    data: dict

class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    error: str