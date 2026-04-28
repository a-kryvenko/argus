from datetime import datetime, timedelta
from typing import Dict, Any
import random

class Predictor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        print(f"SuryaPredictor loaded on {device} (placeholder mode for now)")
        self.ready = True

    def predict_solar_wind(self, hours: int = 96) -> Dict[str, Any]:
        """Realistic placeholder. Later will be changed to real one."""
        now = datetime.utcnow()
        timestamps = [(now + timedelta(hours=i)).isoformat() + "Z" for i in range(hours)]

        # Generate realistic placeholders
        base_speed = 410
        base_density = 2
        base_bz = 0
        forecast = []
        for t in timestamps:
            r_v = random.randrange(-20, 20)
            r_d = (random.random()) * 3
            r_bz = (random.random()) * 5 - 2.5
            forecast.append({
                "timestamp": t,
                "V": base_speed + r_v,
                "N": base_density + r_d,
                "BZ": base_bz + r_bz,
                "KP": 3
            })
        
        return {
            "forecast_generated_at": now.isoformat() + "Z",
            "model": "Surya-1.0 (placeholder - ready for real inference)",
            "note": "Backend is stable. Real Surya inference will be added later.",
            "variables": {
                "V": "Solar wind speed near L1 Lagrange Point",
                "N": "Proton density near L1 Lagrange Point",
                "Bz": "Magnetic field azimuth"
            },
            "forecast": forecast,
            "lead_time_hours": min(hours, 96),
            "confidence": 0.82
        }