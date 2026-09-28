import os
from pathlib import Path
import subprocess
import sys


CLIO_ROOT = Path(__file__).resolve().parents[2]


def test_ingestion_command_imports_as_application_module() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    api_python = CLIO_ROOT / ".venv" / "bin" / "python"
    python = str(api_python) if api_python.is_file() else sys.executable

    result = subprocess.run(
        [python, "-c", "import clio.commands.normalize"],
        cwd=CLIO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_worker_and_file_command_setup_do_not_load_numeric_stack():
    result = subprocess.run(
        [sys.executable, "-c", """
import sys
import clio.cli
import clio.worker
from clio.config import load_observation_config
load_observation_config()
import clio.commands.collect
import clio.commands.backfill_observations
import clio.commands.check_collector_health
heavy = {'pandas', 'numpy', 'pyarrow', 'scipy', 'astropy', 'sunpy'}
assert not heavy.intersection(sys.modules), heavy.intersection(sys.modules)
assert 'clio.observations.live' not in sys.modules
assert 'clio.observations.backfill' not in sys.modules
"""],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
