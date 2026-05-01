import os
import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Dict

_config = None

class _CommonConfig(BaseSettings):
    workdir: Path 
    data_root: Path
    config_root: Path
    project_config: Dict

def get_config() -> _CommonConfig:
    global _config

    if _config is None:
        workdir = Path(__file__).parents[4]
        config_root = workdir / "configs"

        load_dotenv(workdir / ".env")
        if os.path.isfile(workdir / ".env.local"):
            load_dotenv(workdir / ".env.local", override=True)

        config_path = config_root / "project.yaml"
        if not os.path.isfile(config_path):
            raise Exception("Config file not found")
        
        with open(config_path) as f:
            raw = f.read()

        expanded = os.path.expandvars(raw)
    
        _config = _CommonConfig(
            workdir=workdir,
            data_root= workdir / "data",
            config_root=config_root,
            project_config=yaml.safe_load(expanded)
        )
    
    return _config
