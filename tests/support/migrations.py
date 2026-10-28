import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def migration_sql(domain, revision, tmp_path):
    (tmp_path / 'configs').mkdir()
    (tmp_path / 'configs/project.yaml').write_text('project: {name: test}\n')
    (tmp_path / 'configs/models_registry.yaml').write_text('models: {}\n')
    environment = {**os.environ, 'ARGUS_WORKDIR': str(tmp_path),
                   **{domain.upper() + '_DB_' + key: value for key, value in
                      dict(HOST='localhost', PORT='5432', NAME=domain, USER='owner', PASSWORD='p@ss:word/%').items()},
                   'PYTHONPATH': ''}
    python = str(ROOT / 'apps' / domain / '.venv/bin/python')
    command = [python, '-m', 'alembic', '-c', str(ROOT / 'apps' / domain / 'alembic.ini')]
    return subprocess.run([*command, 'upgrade', revision, '--sql'], env=environment,
                          capture_output=True, text=True)
