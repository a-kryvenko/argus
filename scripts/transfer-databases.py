"""Database checks for the host's one-time transfer command; never handles Docker."""
import argparse
from contextlib import redirect_stdout
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

import psycopg
from psycopg import sql
from sqlalchemy import URL
from common.database import database_parameters

DOMAINS = ('api', 'clio', 'prophet', 'intelligence')


def connect(url):
    return psycopg.connect(url.set(drivername='postgresql').render_as_string(hide_password=False))


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True) + '\n')
    temporary.replace(path)


def snapshot(url, domain):
    """Stable data fingerprints plus table/sequence/function/trigger definitions."""
    result = {'tables': {}, 'sequences': {}}
    with connect(url) as conn:
        conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        conn.execute("SET LOCAL timezone = 'UTC'")
        conn.execute('SET LOCAL search_path = pg_catalog')
        tables = conn.execute("SELECT tablename FROM pg_tables WHERE schemaname=%s ORDER BY tablename", (domain,)).fetchall()
        if not any(name == 'alembic_version' for (name,) in tables):
            raise ValueError(f'{domain}: missing Alembic history')
        for (name,) in tables:
            columns = conn.execute("""SELECT a.attname,format_type(a.atttypid,a.atttypmod),a.attnotnull,
                pg_get_expr(d.adbin,d.adrelid),a.attidentity,a.attgenerated
                FROM pg_attribute a JOIN pg_class c ON c.oid=a.attrelid
                JOIN pg_namespace n ON n.oid=c.relnamespace
                LEFT JOIN pg_attrdef d ON d.adrelid=c.oid AND d.adnum=a.attnum
                WHERE n.nspname=%s AND c.relname=%s AND a.attnum>0 AND NOT a.attisdropped ORDER BY a.attnum""", (domain, name)).fetchall()
            digest, count = hashlib.sha256(), 0
            # Sorting canonical row JSON also covers tables without a primary key.
            with conn.cursor(name='transfer_rows') as rows:
                # t.* explicitly denotes the whole row, even when a column is named t.
                rows.execute(sql.SQL('SELECT row_to_json(t.*)::text FROM {}.{} t ORDER BY (row_to_json(t.*)::text) COLLATE "C"')
                             .format(sql.Identifier(domain), sql.Identifier(name)))
                for (row,) in rows:
                    encoded = row.encode()
                    digest.update(len(encoded).to_bytes(8, 'big'))
                    digest.update(encoded)
                    count += 1
            result['tables'][name] = {'columns': columns, 'count': count, 'sha256': digest.hexdigest()}
        for (name,) in conn.execute("SELECT sequencename FROM pg_sequences WHERE schemaname=%s ORDER BY sequencename", (domain,)):
            state = conn.execute(sql.SQL('SELECT last_value,is_called FROM {}.{}').format(sql.Identifier(domain), sql.Identifier(name))).fetchone()
            definition = conn.execute("SELECT start_value,min_value,max_value,increment_by,cycle,cache_size FROM pg_sequences WHERE schemaname=%s AND sequencename=%s", (domain, name)).fetchone()
            result['sequences'][name] = {'state': state, 'definition': definition}
        result['indexes'] = conn.execute("SELECT indexname,indexdef FROM pg_indexes WHERE schemaname=%s ORDER BY indexname", (domain,)).fetchall()
        result['constraints'] = conn.execute("SELECT c.relname,k.conname,pg_get_constraintdef(k.oid) FROM pg_constraint k JOIN pg_class c ON c.oid=k.conrelid JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s ORDER BY c.relname,k.conname", (domain,)).fetchall()
        result['triggers'] = conn.execute("SELECT c.relname,t.tgname,pg_get_triggerdef(t.oid) FROM pg_trigger t JOIN pg_class c ON c.oid=t.tgrelid JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname=%s AND NOT t.tgisinternal ORDER BY c.relname,t.tgname", (domain,)).fetchall()
        result['functions'] = conn.execute("SELECT pg_get_functiondef(p.oid) FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname=%s AND p.prokind IN ('f','p') ORDER BY p.proname,pg_get_function_identity_arguments(p.oid)", (domain,)).fetchall()
    return json.loads(json.dumps(result))


def target_empty(admin, database):
    with connect(admin.set(database=database)) as conn:
        return not conn.execute("""SELECT 1 FROM pg_namespace WHERE nspname NOT IN ('public','information_schema') AND nspname NOT LIKE 'pg_%'
            UNION ALL SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE n.nspname='public'
            UNION ALL SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace WHERE n.nspname='public' LIMIT 1""").fetchone()


def verify_target(admin, target, domain, expected):
    if snapshot(target, domain) != expected:
        raise ValueError(f'{domain}: restored database does not match the saved source snapshot')
    with connect(admin.set(database=target.database)) as conn:
        foreign = conn.execute("SELECT nspname FROM pg_namespace WHERE nspname NOT IN ('public','information_schema',%s) AND nspname NOT LIKE 'pg_%%'", (domain,)).fetchall()
        wrong_owner = conn.execute("""SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
            WHERE n.nspname=%s AND pg_get_userbyid(c.relowner)<>%s
            UNION ALL SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace
            WHERE n.nspname=%s AND pg_get_userbyid(p.proowner)<>%s
            UNION ALL SELECT 1 FROM pg_namespace WHERE nspname=%s AND pg_get_userbyid(nspowner)<>%s LIMIT 1""",
            (domain, target.username, domain, target.username, domain, target.username)).fetchone()
        if foreign or wrong_owner:
            raise ValueError(f'{domain}: unexpected schema or object ownership')


def configuration(source_name=None):
    admin = URL.create('postgresql', **database_parameters(''))
    source = admin.set(database=source_name or admin.database)
    admin = admin.set(database='postgres')
    targets = {d: URL.create('postgresql', **database_parameters(d.upper())) for d in DOMAINS}
    for url in [source, *targets.values()]:
        for name in (url.database, url.username):
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,63}', name):
                raise ValueError('Transfer requires database and role names containing letters, digits, underscores or hyphens')
    if source.database in {u.database for u in targets.values()}:
        raise ValueError('Source and target databases must be different')
    identity = {'source': [source.host, source.port, source.database, source.username],
                'targets': {d: [u.host, u.port, u.database, u.username] for d, u in targets.items()}}
    return admin, source, targets, identity


def run(command, state, source_name=None, domain=None, system_identifier=None):
    admin, source, targets, identity = configuration(source_name)
    if system_identifier is not None:
        with connect(admin) as conn:
            actual = conn.execute('SELECT system_identifier FROM pg_control_system()').fetchone()[0]
        if str(actual) != system_identifier:
            raise ValueError('Configured database server does not match the installed PostgreSQL container')
    plan_path = state / 'identity.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != identity:
        raise ValueError('Transfer settings differ from the saved operation; refusing to reuse backups')
    if command == 'identity':
        if not plan_path.exists():
            raise ValueError('Missing saved transfer identity')
        return
    if command == 'preflight':
        with connect(source) as conn:
            for d in DOMAINS:
                if not conn.execute('SELECT to_regclass(%s)', (d + '.alembic_version',)).fetchone()[0]:
                    raise ValueError(f'Source is missing {d}.alembic_version; this command requires the existing domain schemas')
        spec = importlib.util.spec_from_file_location('provisioning', Path(__file__).with_name('provision-databases.py'))
        provisioning = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(provisioning)
        with redirect_stdout(sys.stderr):
            provisioning.provision(admin, targets, apply=False)
        ready = {}
        with connect(admin) as conn:
            for d, target in targets.items():
                exists = conn.execute('SELECT 1 FROM pg_database WHERE datname=%s', (target.database,)).fetchone()
                ready[d] = False
                if exists and not target_empty(admin, target.database):
                    checkpoint = state / (d + '.source.json')
                    if not plan_path.exists() or not checkpoint.exists():
                        raise ValueError(f'{d}: target is not empty and has no transfer checkpoint')
                    verify_target(admin, target, d, json.loads(checkpoint.read_text()))
                    ready[d] = True
        save(plan_path, identity)
        print('source\t' + source.database)
        for d, target in targets.items():
            print('\t'.join([d, target.database, target.username, str(int(ready[d]))]))
        return
    if command == 'freeze':
        with connect(admin) as conn:
            conn.execute(sql.SQL('ALTER DATABASE {} SET default_transaction_read_only = on').format(sql.Identifier(source.database)))
            conn.commit()
            if conn.execute("SELECT 1 FROM pg_stat_activity WHERE datname=%s AND backend_type='client backend' AND application_name<>'pg_isready' LIMIT 1", (source.database,)).fetchone():
                raise ValueError('Source still has client sessions; stop direct/manual clients and retry')
        return
    if command == 'provision':
        spec = importlib.util.spec_from_file_location('provisioning', Path(__file__).with_name('provision-databases.py'))
        provisioning = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(provisioning)
        provisioning.provision(admin, targets, apply=True)
        return
    checkpoint = state / (domain + '.source.json')
    if command == 'snapshot':
        value = snapshot(source, domain)
        if checkpoint.exists() and json.loads(checkpoint.read_text()) != value:
            raise ValueError(f'{domain}: source changed since the saved snapshot')
        save(checkpoint, value)
    elif command == 'verify':
        verify_target(admin, targets[domain], domain, json.loads(checkpoint.read_text()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('identity', 'preflight', 'freeze', 'provision', 'snapshot', 'verify'))
    parser.add_argument('--state', type=Path, default=Path('/transfer'))
    parser.add_argument('--source')
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--system-identifier', required=True)
    args = parser.parse_args()
    if args.command in ('snapshot', 'verify') and not args.domain:
        parser.error('--domain is required')
    try:
        run(args.command, args.state, args.source, args.domain, args.system_identifier)
    except ValueError as exc:
        parser.exit(1, str(exc) + '\n')
    except (psycopg.Error, OSError):
        parser.exit(1, 'Database transfer check failed; verify connection settings, permissions and storage.\n')


if __name__ == '__main__':
    main()
