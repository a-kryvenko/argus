from fastapi import APIRouter, HTTPException

from api_public.schemas.forecast import ForecastRequest, ForecastResponse
from api_public.dependencies.container import get_forecast_service

router = APIRouter(prefix="/forecast", tags=["forecast"])


@router.post("", response_model=ForecastResponse)
def forecast(payload: ForecastRequest):
    service = get_forecast_service()
    try:
        result = service.predict(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ForecastResponse(**result)
