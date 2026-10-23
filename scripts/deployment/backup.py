"""Stream a database dump to a verified archive with progress while services stay online."""
import gzip
import os
import subprocess
import threading
import time


def create_backup(command, directory):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / time.strftime('pre-migration-%Y%m%dT%H%M%S.sql.gz', time.gmtime())
    partial = path.with_name(path.name + '.partial')
    started = time.monotonic()
    finished = threading.Event()

    def progress():
        while not finished.wait(15):
            size = partial.stat().st_size if partial.exists() else 0
            print(f'[deploy] database-backup: {time.monotonic()-started:.0f}s, {size / 1024**2:.1f} MiB compressed', flush=True)

    monitor = threading.Thread(target=progress, daemon=True)
    monitor.start()
    try:
        with open(partial, 'wb', opener=lambda name, flags: os.open(name, flags, 0o600)) as output:
            os.fchmod(output.fileno(), 0o600)
            with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
                try:
                    with gzip.GzipFile(fileobj=output, mode='wb', compresslevel=1) as compressed:
                        while chunk := process.stdout.read(1024 * 1024):
                            compressed.write(chunk)
                except BaseException:
                    process.terminate()
                    raise
                finally:
                    process.stdout.close()
                if process.wait():
                    raise RuntimeError('Database dump failed; incomplete archive retained')
            output.flush()
            os.fsync(output.fileno())
        print(f'[deploy] database-backup: dump/compression complete after {time.monotonic()-started:.0f}s; verifying archive', flush=True)
        subprocess.run(['gzip', '-t', str(partial)], check=True)
        partial.replace(path)
        print(f'[deploy] database-backup: verified {path.name}, {path.stat().st_size / 1024**2:.1f} MiB', flush=True)
        return path
    finally:
        finished.set()
        monitor.join()
