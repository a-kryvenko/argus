from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.routers.auth import router as auth_router

from forecast_core.predictor import forecast

app = FastAPI(
    title=settings.APP_NAME,
    debug=settings.DEBUG
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)

@app.get("/api/forecast/solar-wind")
async def get_solar_wind(hours: int = 96):
    f = forecast()

    if not f:
        return {"error": "Forecast not ready yet. Please wait."}

    return f

@app.get("/health")
async def health():
    f = forecast()

    return {
        "status": "ok",
        "last_update": f.get("forecast_generated_at") if f else None
    }