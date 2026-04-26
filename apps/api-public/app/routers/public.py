from fastapi import APIRouter
from core.cache import get_forecast

router = APIRouter(prefix="/api", tags=["public"])

@router.get("/forecast/solar-wind")
async def get_solar_wind(hours: int = 96):
    forecast = get_forecast()
    if not forecast:
        return {"error": "Forecast not ready yet. Please wait."}

    return {
        **forecast,
        "source": "cache",
        "requested_hours": min(hours, 72)
    }