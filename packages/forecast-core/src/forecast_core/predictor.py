from datetime import datetime
from surya_adapter.wind.solar_wind import forecast as wind_forecast
import random
from forecast_core.data_pipelines.live import get_observation

_live_forecast = None

def forecast():
    global _live_forecast

    if _live_forecast is None:
        _live_forecast = _create_forecast()
    
    return _live_forecast

def _create_forecast():
    now = datetime.utcnow()
    
    observation = get_observation()
    print(observation)
    exit()

    # Generate response
    base_density = 2
    base_bz = 0
    forecast = []
    forecast_result = wind_forecast(
        observation=observation,
        output_dir="data/forecast",
        device="cpu"
    )
    for row in forecast_result.rows:
        r_d = (random.random()) * 3
        r_bz = (random.random()) * 5 - 2.5
        forecast.append({
            "timestamp": row["target_timestamp"],
            "V": row["V"],
            "N": base_density + r_d,
            "BZ": base_bz + r_bz,
            "KP": 3
        })
        
    return {
        "forecast_generated_at": now.isoformat() + "Z",
        "model": "Surya",
        "note": "Backend is stable. Real forecast for N, BZ and KP will be added later",
        "variables": {
            "V": "Solar wind speed near L1 Lagrange Point, Surya forecast",
            "N": "Proton density near L1 Lagrange Point",
            "Bz": "Magnetic field azimuth"
        },
        "forecast": forecast,
        "confidence": 0.82
    }

forecast()