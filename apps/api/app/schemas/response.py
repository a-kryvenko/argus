from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    success: bool
    data: T|None = None
    error: dict|None = None

def success_response(data: T):
    return ApiResponse[T](
        success=True,
        data=data
    )

def error_response(msg: str, code: str):
    return ApiResponse[None](
        success=False,
        error={
            "code": code,
            "message": msg
        }
    )