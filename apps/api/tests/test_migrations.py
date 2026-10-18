"""Migration commands work from the installed API package without a checkout cwd."""
import os
import subprocess
import sys


def test_migration_history_outside_project_directory(tmp_path):
    environment = os.environ.copy()
    environment.pop('PYTHONPATH', None)
    result = subprocess.run([sys.executable, '-c', 'from api.cli import main; main()',
                             'migrate', 'heads'], cwd=tmp_path, env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '20261001_0012 (head)'
