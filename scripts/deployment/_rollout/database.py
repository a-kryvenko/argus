"""Run schema upgrades and create maintenance backups."""
from backup import create_backup


class Database:
    def __init__(self, root, compose):
        self.root = root
        self.compose = compose

    def migrate(self):
        # Preserve the shell's deployment lock for the nested argus command.
        self.compose.execute(str(self.root / 'argus'), 'db', 'migrate', close_fds=False)

    def backup(self):
        return create_backup(self.compose.command('exec', '-T', 'postgres',
            'sh', '-c', 'pg_dumpall --username="$POSTGRES_USER"'), self.root / 'backups')

    def rotate_backups(self):
        backups = sorted(
            {
                path
                for pattern in ('pre-migration-*.sql', 'pre-migration-*.sql.gz')
                for path in (self.root / 'backups').glob(pattern) if path.is_file()
            },
            reverse=True
        )
        
        for backup in backups[3:]:
            backup.unlink()
