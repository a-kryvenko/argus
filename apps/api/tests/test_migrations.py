"""Alembic resolves packaged migrations outside the project directory."""
import os
from pathlib import Path
import subprocess
import sys


def test_migration_history_outside_project_directory(tmp_path):
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-m', 'alembic', '-c',
                             str(Path(__file__).resolve().parents[1] / 'alembic.ini'), 'heads'], cwd=tmp_path, env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert '(head)' in result.stdout
