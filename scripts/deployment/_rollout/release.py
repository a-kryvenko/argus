"""Verify and install the prepared release bundle."""
import json
import shlex
from pathlib import Path

from .checksum import get_checksum


class Release:
    def __init__(self, root: Path, bundle: Path, execute):
        self.root = root
        self.bundle = bundle
        self.execute = execute

    def read_manifest(self) -> dict:
        manifest = json.loads((self.root / '.release_manifest').read_text())
        if not isinstance(manifest, dict) or 'mode' not in manifest:
            raise RuntimeError('Missing release mode in .release_manifest')
        mode = manifest['mode']
        if mode not in ('rolling', 'maintenance'):
            raise RuntimeError(f'Unknown release mode: {mode!r}')
        return manifest

    def verify(self, manifest):
        checksum = get_checksum(self.bundle)
        if checksum != manifest['checksum']:
            raise RuntimeError('Release checksum mismatch')

    def install(self):
        if self.root == self.bundle or self.root.is_relative_to(self.bundle):
            raise RuntimeError('Release bundle must be separate from the application root')
        
        self.execute('rsync', '-a', '--no-owner', '--no-group', '--inplace',
                     '--exclude=/.docker/nginx-proxy/', str(self.bundle) + '/', str(self.root) + '/')
        (self.root / '.argus-mode').write_text('prod\n')
        (self.root / 'data/next-static').mkdir(parents=True, exist_ok=True)

    def install_crontab(self):
        if any(char in str(self.root) for char in ('\n', '\r', '%')):
            raise RuntimeError('Cron deployment path must not contain newlines or %')
        contents = (self.root / 'cronjobs.txt').read_text().rstrip()
        self.execute('crontab', '-', input=contents.replace('@ROOT@', shlex.quote(str(self.root))) + '\n', text=True)
