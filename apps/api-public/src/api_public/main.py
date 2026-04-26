from fastapi import FastAPI
from api_public.routes.forecast import router as forecast_router

app = FastAPI(title="Argus Sunwatch Public API")
app.include_router(forecast_router)
