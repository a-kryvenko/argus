"""Persist the active color and maintenance recovery checkpoint."""
import json
import os


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.next')
    with open(temporary, 'w', opener=lambda name, flags: os.open(name, flags, 0o600)) as output:
        json.dump(value, output, indent=2)
        os.fchmod(output.fileno(), 0o600)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


class DeploymentState:
    def __init__(self, root):
        self.root = root
        self.recovery_path = root / '.maintenance-recovery.json'

    def read_color(self, *, required=True):
        marker = self.root / '.active-color'
        if not marker.is_file():
            if not required:
                return None
            raise RuntimeError('Missing .active-color; run a maintenance deployment first')
        
        color = marker.read_text().strip()
        if color not in ('blue', 'green'):
            raise RuntimeError(f'Invalid active color: {color!r}')
        
        return color

    def save_color(self, color):
        path = self.root / '.active-color'
        temporary = path.with_name(path.name + '.next')
        with temporary.open('w') as output:
            output.write(color + '\n')
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def ensure_recovered(self):
        if self.recovery_path.exists():
            raise RuntimeError('Finish maintenance recovery before applying a release')

    def record_maintenance(self, record):
        write_json(self.recovery_path, record)

    def clear_maintenance(self):
        self.recovery_path.unlink()

    def record_release(self, manifest):
        write_json(self.root / '.release.json', manifest)
