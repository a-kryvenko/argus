from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    APP_NAME: str = "Argus SunWatch Public API"
    DEBUG: bool = True
    DEVICE: str = "cpu"

    class Config:
        env_file = ".env"

settings = Settings()