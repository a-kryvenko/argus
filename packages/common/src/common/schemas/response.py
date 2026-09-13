from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    """Response envelope. Successful responses contain data; errors contain code and message."""

    success: bool = Field(description="True for a successful response; false for an API error.", examples=[True])
    data: T|None = Field(default=None, description="Response payload on success; null or omitted on error.")
    error: dict|None = Field(
        default=None,
        description="Error with string fields code (machine-readable identifier) and message (explanation); null on success.",
        examples=[{"code": "NOT_FOUND", "message": "Forecast product not found"}],
        json_schema_extra={"additionalProperties": False, "properties": {
            "code": {"type": "string", "description": "Machine-readable error identifier."},
            "message": {"type": "string", "description": "Explanation of the error."},
        }},
    )

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
