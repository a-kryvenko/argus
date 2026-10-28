import importlib.util
from pathlib import Path

from alembic.script import ScriptDirectory

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "migration_status", ROOT / "scripts/deployment/migration_status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)


def test_current_intelligence_upgrade_is_compatible():
    scripts = ScriptDirectory(str(ROOT / 'apps/intelligence/src/intelligence/db/migrations'))
    assert status.revision_status(scripts, {'20261017_forecast_inputs'}, online=True) == 'pending'
    assert status.revision_status(scripts, set(), online=True) == 'pending'
