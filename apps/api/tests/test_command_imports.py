import os
from pathlib import Path
import subprocess
import sys


API_ROOT = Path(__file__).resolve().parents[1]


def test_forecast_command_imports_as_application_module() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    api_python = API_ROOT / ".venv" / "bin" / "python"
    python = str(api_python) if api_python.is_file() else sys.executable

    result = subprocess.run(
        [python, "-c", "import app.commands.generate_forecast"],
        cwd=API_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
