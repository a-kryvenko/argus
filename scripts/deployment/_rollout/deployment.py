"""Coordinate the rolling and maintenance deployment workflows."""
from _rollout.applications import Applications
from _rollout.compose import Compose
from _rollout.database import Database
from _rollout.proxy import NginxProxy
from _rollout.release import Release
from _rollout.state import DeploymentState

from .applications import HTTP, WORKERS, slot_names


class Deployment:
    def __init__(
        self,
        *,
        compose: Compose,
        release: Release,
        database: Database,
        applications: Applications,
        proxy: NginxProxy,
        state: DeploymentState
    ):
        self.compose = compose
        self.release = release
        self.database = database
        self.applications = applications
        self.proxy = proxy
        self.state = state

    def apply(self):
        self.manifest = self.release.read_manifest()
        self.release.verify(self.manifest)
        
        self.color = self.state.read_color(required=self.manifest['mode'] == 'rolling')
        self.state.ensure_recovered()
        print(f'[deploy] Release mode: {self.manifest["mode"]}', flush=True)

        self.release.install()

        self.compose.run('pull', '--policy', 'always')

        if self.manifest['mode'] == 'maintenance':
            self.apply_maintenance()
        else:
            self.apply_rolling()
        
        print('[deploy] Release applied successfully', flush=True)

    def apply_rolling(self):
        split_public = self.has_public_slot(self.color)
        self.proxy.prepare(self.color, split_public=split_public)
        running = self.compose.run('ps', '--status', 'running', '--services', capture=True).split()
        workers = [name for name in WORKERS if name in running]
        try:
            if workers:
                self.compose.run('stop', *workers)
            self.database.migrate()
            self.swap_blue_green_services(previous_split=split_public)
            self.finish_release()
        except BaseException:
            if workers:
                self.compose.run('start', *workers)
            raise

    def has_public_slot(self, color):
        return bool(self.compose.run('ps', '--all', '-q', f'public-api-{color}', capture=True).strip())

    def swap_blue_green_services(self, *, previous_split=True):
        previous = self.color
        target = 'green' if previous == 'blue' else 'blue'
        
        try:
            self.applications.start(target)

            self.proxy.switch(target)

            self.state.save_color(target)
            self.color = target
        except BaseException:
            self.proxy.switch(previous, split_public=previous_split)
            self.compose.run('stop', *slot_names(target))
            raise

        self.compose.run('stop', *slot_names(previous))

    def apply_maintenance(self):
        record = {'color': self.color, 'backup': None, 'migration_started': False}
        self.state.record_maintenance(record)
        try:
            self.compose.run(
                'up', '-d', '--no-deps', '--pull', 'never',
                '--scale', 'maintenance=1', '--wait', 'maintenance'
            )
            self.compose.run('stop', 'nginx')
            self.compose.run('stop', *WORKERS, *HTTP, *slot_names('blue'), *slot_names('green'))
            record['backup'] = str(self.database.backup())
            self.state.record_maintenance(record)

            record['migration_started'] = True
            self.state.record_maintenance(record)

            self.database.migrate()

            self.proxy.prepare('green')
            self.applications.start('green')
            self.proxy.switch('green')
            self.state.save_color('green')
            self.color = 'green'

            self.finish_release()
            self.compose.run('stop', 'maintenance')
        except BaseException:
            if record['migration_started'] or record['color'] is None:
                self.compose.run('stop', 'nginx', *WORKERS, *slot_names('green'))

                raise RuntimeError(
                    f'Maintenance failed. Traffic remains closed. Recovery state: {self.state.recovery_path}'
                )
            
            previous_color = record['color']
            previous = slot_names(previous_color)
            if not self.has_public_slot(previous_color):
                previous.remove(f'public-api-{previous_color}')
            self.compose.run('start', *previous, *WORKERS)
            self.compose.run('start', 'nginx')
            self.compose.run('stop', 'maintenance')
            self.state.clear_maintenance()

            raise

        self.state.clear_maintenance()
        self.database.rotate_backups()

    def finish_release(self):
        self.compose.run('up', '-d', '--no-deps', '--force-recreate', '--pull', 'never',
                         '--wait', '--wait-timeout', '300', *WORKERS)
        self.release.install_crontab()
        self.compose.run('up', '-d', '--no-deps', '--no-recreate', '--pull', 'never', '--wait', 'nginx')
        self.proxy.reload_public()
        self.compose.run('up', '-d', '--no-deps', '--pull', 'never', '--wait', 'alloy')
        self.compose.run('restart', 'alloy')

        self.state.record_release(self.manifest)
