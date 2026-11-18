"""Execute production Compose commands through argus."""
import subprocess


class Compose:
    def __init__(self, root):
        self.root = root

    @staticmethod
    def execute(*args, **kwargs):
        subprocess.run(args, check=True, **kwargs)

    def run(self, *args, capture=False, **kwargs):
        command = self.command(*args)
        if capture:
            return subprocess.run(command, check=True, capture_output=True, text=True, **kwargs).stdout
        self.execute(*command, **kwargs)

    def command(self, *args):
        return [str(self.root / 'argus'), 'compose', *args]
