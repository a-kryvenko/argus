"""Run the Go observation API against a test database."""
from contextlib import contextmanager
from pathlib import Path
import selectors
import subprocess

CLIO_ROOT = Path(__file__).resolve().parents[2] / 'apps/clio'


def migrate_clio(python, environment):
    subprocess.run(
        [python, '-m', 'alembic', '-c', str(CLIO_ROOT / 'alembic.ini'), 'upgrade', 'head'],
        env=environment, check=True, timeout=60,
    )


@contextmanager
def clio_server(directory, environment):
    with go_http_server(directory, environment, CLIO_ROOT / 'api-go', 'clio-api',
                        'CLIO_HTTP_ADDR', 'Clio listening on ') as url:
        yield url


@contextmanager
def go_http_server(directory, environment, module, name, bind_setting, listen_message):
    binary = directory / name
    subprocess.run(['go', 'build', '-o', str(binary), '.'],
                   cwd=module, check=True, timeout=120)
    with subprocess.Popen([binary], env={**environment, bind_setting: '127.0.0.1:0'},
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
        failed = False
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=10), 'Go API did not start'
                line = process.stdout.readline().rstrip()
            _, marker, address = line.partition(listen_message)
            assert marker, line
            yield f'http://{address}'
        except BaseException:
            failed = True
            raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            if failed:
                print(process.stdout.read(), end="")
