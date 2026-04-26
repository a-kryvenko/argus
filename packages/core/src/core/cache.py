from datetime import datetime

# Global forecast cache
latest_forecast: dict | None = None
last_update: datetime | None = None

def set_forecast(forecast: dict):
    global latest_forecast, last_update
    latest_forecast = forecast
    last_update = datetime.utcnow()

def get_forecast() -> dict | None:
    return latest_forecast

def get_last_update() -> datetime | None:
    return last_update