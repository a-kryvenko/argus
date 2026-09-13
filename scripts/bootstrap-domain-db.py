"""Operator-only provisioning and atomic adoption of the legacy Argus database.

Requires stopped application writers and a backup for adoption. Run without
--apply to validate all operations and roll the transaction back. Runtime apps
never import this administrative tool or receive its credentials.
"""
import argparse
import os
from pathlib import Path

import psycopg
from psycopg import sql

TABLES = {
    'clio': ('measurement', 'normalized_observation', 'solar_wind_observation',
             'geomagnetic_observation', 'observation_source_status',
             'solar_wind_aggregate', 'solar_wind_aggregate_pending', 'solar_wind_retired_hour'),
    'api': ('dashboard_user', 'dashboard_group', 'dashboard_membership',
            'dashboard_session', 'dashboard_login_attempt', 'api_metric'),
}
DOMAINS = (*TABLES, 'prophet')
FUNCTIONS = ('queue_solar_wind_aggregate', 'protect_retired_solar_wind')
LEGACY_HEAD = '20260911_0010'


def exists(conn, schema, table):
    return conn.execute('SELECT to_regclass(%s)', (f'{schema}.{table}',)).fetchone()[0] is not None


def provision(conn, passwords):
    # Serialize operator invocations; application writers must already be stopped.
    conn.execute('SELECT pg_advisory_xact_lock(730999)')
    reserved = {f'argus_{domain}{suffix}' for domain in DOMAINS for suffix in ('', '_migrator')}
    current, database_owner = conn.execute('SELECT current_user, pg_get_userbyid(datdba) FROM pg_database WHERE datname=current_database()').fetchone()
    if current in reserved or database_owner in reserved:
        raise RuntimeError('Bootstrap/database ownership must use a separate administrative role')
    legacy = exists(conn, 'public', 'alembic_version')
    if legacy:
        revisions = conn.execute('SELECT version_num FROM public.alembic_version').fetchall()
        if revisions != [(LEGACY_HEAD,)]:
            raise RuntimeError(f'Legacy database must be at {LEGACY_HEAD} before adoption')
        for domain, tables in TABLES.items():
            if exists(conn, domain, 'alembic_version'):
                raise RuntimeError('Legacy and domain migration histories coexist; refusing adoption')
            for table in tables:
                if not exists(conn, 'public', table) or exists(conn, domain, table):
                    raise RuntimeError(f'Missing legacy table or conflicting target: {domain}.{table}')
    elif any(exists(conn, 'public', table) for tables in TABLES.values() for table in tables):
        raise RuntimeError('Legacy tables exist without their migration history')

    for domain in DOMAINS:
        runtime, owner = f'argus_{domain}', f'argus_{domain}_migrator'
        for role, password_key in ((runtime, f'{domain.upper()}_DB_PASSWORD'),
                                   (owner, f'{domain.upper()}_MIGRATION_PASSWORD')):
            if not conn.execute('SELECT 1 FROM pg_roles WHERE rolname=%s', (role,)).fetchone():
                conn.execute(sql.SQL('CREATE ROLE {} LOGIN').format(sql.Identifier(role)))
            if conn.execute('SELECT 1 FROM pg_auth_members m JOIN pg_roles r ON r.oid=m.member WHERE r.rolname=%s', (role,)).fetchone():
                raise RuntimeError(f'Reserved role {role} has unexpected memberships')
            # SQL literals are escaped by psycopg, not interpolated or logged.
            conn.execute(sql.SQL('ALTER ROLE {} WITH LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS NOINHERIT PASSWORD {}')
                         .format(sql.Identifier(role), sql.Literal(passwords[password_key])))
        conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {} AUTHORIZATION {}').format(sql.Identifier(domain), sql.Identifier(owner)))
        conn.execute(sql.SQL('ALTER SCHEMA {} OWNER TO {}').format(sql.Identifier(domain), sql.Identifier(owner)))
        conn.execute(sql.SQL('REVOKE ALL ON SCHEMA {} FROM PUBLIC').format(sql.Identifier(domain)))
        conn.execute(sql.SQL('GRANT USAGE ON SCHEMA {} TO {}').format(sql.Identifier(domain), sql.Identifier(runtime)))
        for objects, privileges in [('TABLES', 'SELECT, INSERT, UPDATE, DELETE'), ('SEQUENCES', 'USAGE, SELECT'), ('FUNCTIONS', 'EXECUTE')]:
            conn.execute(sql.SQL('ALTER DEFAULT PRIVILEGES FOR ROLE {} IN SCHEMA {} GRANT '+privileges+' ON '+objects+' TO {}')
                         .format(sql.Identifier(owner), sql.Identifier(domain), sql.Identifier(runtime)))
        # EXECUTE on functions is granted to PUBLIC by default globally. A
        # per-schema revoke cannot cancel that global default.
        conn.execute(sql.SQL('ALTER DEFAULT PRIVILEGES FOR ROLE {} REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC').format(sql.Identifier(owner)))

    conn.execute('REVOKE CREATE ON SCHEMA public FROM PUBLIC')
    if legacy:
        conn.execute('SET LOCAL lock_timeout = \'15s\'')
        for domain, tables in TABLES.items():
            for table in tables:
                conn.execute(sql.SQL('ALTER TABLE public.{} SET SCHEMA {}').format(sql.Identifier(table), sql.Identifier(domain)))
                conn.execute(sql.SQL('ALTER TABLE {}.{} OWNER TO {}').format(sql.Identifier(domain), sql.Identifier(table), sql.Identifier(f'argus_{domain}_migrator')))
        for function in FUNCTIONS:
            conn.execute(sql.SQL('ALTER FUNCTION public.{}() SET SCHEMA clio').format(sql.Identifier(function)))
            conn.execute(sql.SQL('ALTER FUNCTION clio.{}() OWNER TO argus_clio_migrator').format(sql.Identifier(function)))
            conn.execute(sql.SQL('ALTER FUNCTION clio.{}() SET search_path TO clio, pg_catalog, pg_temp').format(sql.Identifier(function)))
        for domain, revision in [('clio', '20260908_0009'), ('api', LEGACY_HEAD)]:
            conn.execute(sql.SQL('CREATE TABLE {}.alembic_version (version_num varchar(32), CONSTRAINT {} PRIMARY KEY (version_num))').format(sql.Identifier(domain), sql.Identifier(domain + '_migration_version_pk')))
            conn.execute(sql.SQL('INSERT INTO {}.alembic_version VALUES (%s)').format(sql.Identifier(domain)), (revision,))
            conn.execute(sql.SQL('ALTER TABLE {}.alembic_version OWNER TO {}').format(sql.Identifier(domain), sql.Identifier(f'argus_{domain}_migrator')))
        # Keep the old marker for audit; no old migrator can discover it in public.
        conn.execute('ALTER TABLE public.alembic_version RENAME TO legacy_alembic_version')
        conn.execute('ALTER TABLE public.legacy_alembic_version SET SCHEMA api')
        conn.execute('ALTER TABLE api.legacy_alembic_version OWNER TO argus_api_migrator')

    for domain in DOMAINS:
        runtime = f'argus_{domain}'
        others = [f'argus_{other}{suffix}' for other in DOMAINS if other != domain
                  for suffix in ('', '_migrator')]
        for objects, privileges in [('TABLES', 'SELECT, INSERT, UPDATE, DELETE'), ('SEQUENCES', 'USAGE, SELECT'), ('FUNCTIONS', 'EXECUTE')]:
            conn.execute(sql.SQL('REVOKE ALL ON ALL '+objects+' IN SCHEMA {} FROM PUBLIC').format(sql.Identifier(domain)))
            conn.execute(sql.SQL('GRANT '+privileges+' ON ALL '+objects+' IN SCHEMA {} TO {}').format(sql.Identifier(domain), sql.Identifier(runtime)))
            for role in others:
                conn.execute(sql.SQL('REVOKE ALL ON ALL '+objects+' IN SCHEMA {} FROM {}').format(sql.Identifier(domain), sql.Identifier(role)))
        for role in others:
            conn.execute(sql.SQL('REVOKE ALL ON SCHEMA {} FROM {}').format(sql.Identifier(domain), sql.Identifier(role)))
        if exists(conn, domain, 'alembic_version'):
            conn.execute(sql.SQL('REVOKE ALL ON {}.alembic_version FROM {}').format(sql.Identifier(domain), sql.Identifier(runtime)))
    return 'adopted legacy tables' if legacy else 'provisioned domain roles and schemas'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    from dotenv import load_dotenv
    root = Path(os.getenv('ARGUS_WORKDIR', Path(__file__).resolve().parents[1]))
    load_dotenv(root / '.env')
    load_dotenv(root / '.env.local', override=True)
    required = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'API_DB_PASSWORD',
                'API_MIGRATION_PASSWORD', 'CLIO_DB_PASSWORD', 'CLIO_MIGRATION_PASSWORD',
                'PROPHET_DB_PASSWORD', 'PROPHET_MIGRATION_PASSWORD']
    if any(not os.getenv(key) for key in required):
        raise RuntimeError('Missing required variables: ' + ', '.join(key for key in required if not os.getenv(key)))
    passwords = {key: os.environ[key] for key in required if key not in ('DB_NAME', 'DB_USER')}
    with psycopg.connect(dbname=os.environ['DB_NAME'], user=os.environ['DB_USER'],
                         password=os.environ['DB_PASSWORD'], host=os.getenv('DB_HOST', 'localhost'),
                         port=int(os.getenv('DB_PORT', '5432'))) as conn:
        outcome = provision(conn, passwords)
        if not args.apply:
            conn.rollback()
        print(outcome + ('; committed' if args.apply else '; validated and rolled back (use --apply)'))


if __name__ == '__main__':
    main()
