from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime

from app.core.config import settings
from app.core.predictor import Predictor
from app.core.cache import set_forecast, get_forecast
from app.routers.public import router as public_router
from app.routers.auth import router as auth_router

_predictor: Predictor | None = None
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor

    _predictor = Predictor()

    # Initial forecast
    if _predictor:
        print("Generating initial forecast...")
        forecast = _predictor.predict_solar_wind(hours=72)
        set_forecast(forecast)
        print(f"Initial forecast cached at {datetime.utcnow()}")

    # Updated each hour
    scheduler.add_job(update_forecast_job, 'interval', minutes=60, id='forecast_update')
    scheduler.start()
    print("Hourly forecast scheduler started.")

    yield
    scheduler.shutdown()

async def update_forecast_job():
    global _predictor
    if _predictor:
        print(f"Updating forecast at {datetime.utcnow()}")
        forecast = _predictor.predict_solar_wind(hours=72)
        set_forecast(forecast)
        print("Forecast updated and cached.")

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan,
    debug=settings.DEBUG
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(public_router)
app.include_router(auth_router)

@app.get("/health")
async def health():
    forecast = get_forecast()
    return {
        "status": "ok",
        "gpu": torch.cuda.is_available(),
        "model_loaded": _predictor is not None,
        "forecast_cached": forecast is not None,
        "last_update": forecast.get("forecast_generated_at") if forecast else None
    }