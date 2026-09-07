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
        workdir = _find_workdir()
        config_root = workdir / "configs"

        load_dotenv(workdir / ".env")
        if os.path.isfile(workdir / ".env.local"):
            load_dotenv(workdir / ".env.local", override=True)
    
        _config = _CommonConfig(
            debug=os.getenv("DEBUG", "false").lower() == "true",
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


def _find_workdir() -> Path:
    configured = os.getenv("ARGUS_WORKDIR")
    if configured:
        return Path(configured).expanduser().resolve()
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "configs" / "project.yaml").is_file():
            return candidate
    raise RuntimeError("Set ARGUS_WORKDIR to the application configuration directory")
