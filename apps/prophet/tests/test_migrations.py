from tests.support.migrations import migration_sql


def test_prophet_cleanup_preserves_forecast_history(tmp_path):
    result = migration_sql('prophet', '20260929_prophet_product_slots:head', tmp_path)
    assert result.returncode == 0, result.stderr
    dropped = [line.strip() for line in result.stdout.splitlines() if line.startswith('DROP TABLE')]
    assert dropped == [f'DROP TABLE prophet.{table};' for table in (
        'scheduled_job', 'forecast_slot', 'current_forecast')]
    dropped_columns = [line.strip() for line in result.stdout.splitlines() if 'DROP COLUMN' in line]
    assert dropped_columns == [
        'ALTER TABLE prophet.forecast_run DROP COLUMN scheduled_slot;',
        'ALTER TABLE prophet.forecast_run DROP COLUMN scheduled_product;',
    ]
