from common.schemas.observation import Observation
from argus_prophet.observations import load_inputs


def load_sensor_observations() -> Observation:
    return load_inputs().observations
