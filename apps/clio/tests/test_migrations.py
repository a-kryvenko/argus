from tests.support.migrations import migration_sql


def test_file_observation_expansion_only_adds_storage(tmp_path):
    result = migration_sql('clio', '20261014_enlil_profile:20261027_file_observation', tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'CREATE TABLE file_observation' in result.stdout
    assert 'DROP TABLE' not in result.stdout
    assert 'SELECT * FROM' not in result.stdout


def test_file_cutover_requires_online_validation(tmp_path):
    result = migration_sql('clio', '20261027_file_observation:head', tmp_path)
    assert result.returncode != 0
    assert 'requires an online database connection' in result.stdout + result.stderr
    assert 'DROP TABLE' not in result.stdout
    assert 'UPDATE clio.alembic_version' not in result.stdout


def test_legacy_file_cleanup_drops_only_retired_tables(tmp_path):
    result = migration_sql('clio', '20261027_file_cutover:head', tmp_path)
    assert result.returncode == 0, result.stderr
    dropped = [line.strip() for line in result.stdout.splitlines() if line.startswith('DROP TABLE')]
    assert dropped == [f'DROP TABLE clio.{table};' for table in (
        'gong_snapshot', 'goes_snapshot', 'donki_poll', 'enlil_profile_receipt', 'scheduled_job')]
