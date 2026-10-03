"""Cross-service database provisioning, migrations and ownership."""
import os

from domain_storage import database, migrate, provisioning, runtime

import psycopg
import pytest
from sqlalchemy.engine import make_url


def test_fresh_databases_repeat_migrations_and_reject_foreign_connections(database):
    dsns, urls, environment = database
    migrate(environment)
    provisioning.provision(os.environ['TEST_DATABASE_ADMIN_DSN'], urls, apply=True)
    migrate(environment)
    for domain in provisioning.DOMAINS:
        with runtime(dsns, domain, urls) as conn:
            assert conn.execute(f'SELECT count(*) FROM {domain}.alembic_version').fetchone()[0] == 1
            # The one service owner can also perform its migrations.
            conn.execute(f'CREATE TABLE {domain}.owner_test(id int)')
            conn.execute(f'DROP TABLE {domain}.owner_test')
        for other in provisioning.DOMAINS:
            if other == domain:
                continue
            foreign = make_url(urls[domain]).set(database=make_url(urls[other]).database)
            with pytest.raises(psycopg.OperationalError):
                psycopg.connect(foreign.render_as_string(hide_password=False))
    with runtime(dsns, 'clio', urls) as conn:
        conn.execute('SET search_path TO pg_catalog')
        conn.execute("""INSERT INTO clio.solar_wind_observation(kind, observed_at, spacecraft, active, received_at, "values", raw)
            VALUES ('mag', '2026-09-02 00:00:00+00', 'A', true, '2026-09-02 00:00:00+00', '{"bz": -5}', '{}')""")
        assert conn.execute("SELECT count(*) FROM clio.solar_wind_aggregate_pending WHERE hour='2026-09-02 00:00:00+00'").fetchone()[0] == 1


def test_provisioning_does_not_rotate_existing_passwords(database):
    _, urls, _ = database
    changed = dict(urls)
    changed['api'] = make_url(urls['api']).set(password='wrong-password').render_as_string(hide_password=False)
    with pytest.raises(psycopg.OperationalError):
        provisioning.provision(os.environ['TEST_DATABASE_ADMIN_DSN'], changed, apply=True)
    with psycopg.connect(urls['api']) as conn:
        assert conn.execute('SELECT 1').fetchone()[0] == 1
