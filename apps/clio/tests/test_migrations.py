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
    result = migration_sql('clio', '20261027_file_cutover:20261027_measurement_metadata', tmp_path)
    assert result.returncode == 0, result.stderr
    dropped = [line.strip() for line in result.stdout.splitlines() if line.startswith('DROP TABLE')]
    assert dropped == [f'DROP TABLE clio.{table};' for table in (
        'gong_snapshot', 'goes_snapshot', 'donki_poll', 'enlil_profile_receipt', 'scheduled_job')]


def test_provider_contract_keeps_legacy_column_and_allows_rolling_upgrade(tmp_path):
    from pathlib import Path
    from alembic.script import ScriptDirectory
    from tests.deployment.test_migrations import status

    result = migration_sql('clio', '20261101_drop_source_status:20261102_provider_contract', tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'SET value = NULL, quality = NULL' in result.stdout
    assert "WHERE quality IN ('flagged', 'missing')" in result.stdout
    assert 'DROP' not in result.stdout and 'DELETE' not in result.stdout
    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[1] / 'src/clio/db/migrations'))
    assert scripts.get_revision('20261102_provider_contract').module.online_compatible is True
    assert status.revision_status(scripts, {'20261101_drop_source_status'}, online=True) == 'blocked:20261102_files'


def test_provider_migration_preserves_legacy_reads_and_valid_measurements():
    from pathlib import Path
    import sqlalchemy as sa
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from alembic.script import ScriptDirectory

    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[1] / 'src/clio/db/migrations'))
    revision = scripts.get_revision('20261102_provider_contract')
    engine = sa.create_engine('sqlite://')
    with engine.begin() as connection:
        connection.exec_driver_sql("ATTACH DATABASE ':memory:' AS clio")
        connection.exec_driver_sql('CREATE TABLE clio.measurement (metric TEXT, value FLOAT, quality TEXT)')
        connection.exec_driver_sql("""INSERT INTO clio.measurement VALUES
            ('v', 400, 'unverified'), ('bz', -3, NULL), ('kp', 5, 'flagged'), ('n', NULL, 'missing')""")
        with Operations.context(MigrationContext.configure(connection)):
            revision.module.upgrade()
        # The old HTTP process can still select quality; new readers see no rejected values.
        rows = connection.exec_driver_sql('SELECT metric, value, quality FROM clio.measurement').all()
        assert rows == [('v', 400., 'unverified'), ('bz', -3., None), ('kp', None, None), ('n', None, None)]
    engine.dispose()


def test_files_migration_adds_registry_and_requires_maintenance(tmp_path):
    from pathlib import Path
    from alembic.script import ScriptDirectory
    from tests.deployment.test_migrations import status

    result = migration_sql('clio', '20261102_provider_contract:20261102_files', tmp_path)
    assert result.returncode == 0, result.stderr
    assert 'CREATE TABLE clio.files' in result.stdout
    assert 'UNIQUE (bucket, key)' in result.stdout
    assert 'sha256' not in result.stdout and 'size_bytes' not in result.stdout
    assert 'DROP TABLE' not in result.stdout and 'DROP COLUMN' not in result.stdout
    assert 'file_observation ADD COLUMN file_id' in result.stdout
    assert 'REFERENCES clio.files (id)' in result.stdout
    scripts = ScriptDirectory(str(Path(__file__).resolve().parents[1] / 'src/clio/db/migrations'))
    assert status.revision_status(scripts, {'20261102_provider_contract'}, online=True) == 'blocked:20261102_files'
    assert status.revision_status(scripts, {'20261102_provider_contract'}) == 'pending'
    assert scripts.get_revision('20261107_file_observation').module.online_compatible is True
    assert status.revision_status(scripts, {'20261102_files'}, online=True) == 'pending'
    assert status.revision_status(scripts, {'20261107_file_observation'}, online=True) == 'current'
