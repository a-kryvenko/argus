import os
import yaml
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Dict

_config = None

class _CommonConfig(BaseSettings):
    debug: bool
    workdir: Path 
    data_root: Path
    config_root: Path
    project_config: Dict
    models_registry: Dict

def get_config() -> _CommonConfig:
    global _config

    if _config is None:
        workdir = Path(__file__).parents[4]
        config_root = workdir / "configs"

        load_dotenv(workdir / ".env")
        if os.path.isfile(workdir / ".env.local"):
            load_dotenv(workdir / ".env.local", override=True)
    
        _config = _CommonConfig(
            debug=os.getenv("DEBUG") and (os.getenv("DEBUG") == "true" or os.getenv("DEBUG") == "True"),
            workdir=workdir,
            data_root= workdir / "data",
            config_root=config_root,
            project_config=_load_yml_config( config_root / "project.yaml"),
            models_registry=_load_yml_config( config_root / "models_registry.yaml")
        )
    
    return _config

def _load_yml_config(p: Path):
    if not os.path.isfile(p):
        raise Exception(f"{p} not found")
        
    with open(p) as f:
        raw = f.read()
    
    expanded = os.path.expandvars(raw)

    return yaml.safe_load(expanded)
