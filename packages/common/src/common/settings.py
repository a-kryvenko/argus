import os
import yaml
from pathlib import Path
from pydantic_settings import BaseSettings

class CommonSettings(BaseSettings):
    data_root: Path = Path("data")
    artifacts_root: Path = Path("data/artifacts")

    def load_config(path=None):

        config_path = path if path is not None else "configs/project.yaml"
        
        if not os.path.isfile(config_path):
            raise Exception("Config file not found")
        
        with open(config_path) as f:
            raw = f.read()
        expanded = os.path.expandvars(raw)
        return yaml.safe_load(expanded)
