"""Exercise the executor with real spawn, including oversized reports and crashes."""
import subprocess
import sys

import pytest


@pytest.mark.parametrize('mode', ['success', 'failure', 'crash', 'shutdown', 'cancel'])
def test_isolated_executor_releases_process_and_reports_outcome(tmp_path, mode):
    script = tmp_path / 'exercise_executor.py'
    script.write_text("""
import asyncio
import multiprocessing
import os
import sys
import signal
import threading
from types import SimpleNamespace, ModuleType
from common import runtime
runtime.run_command = lambda run: run()

async def run(args):
    if args.mode == 'cancel':
        await asyncio.sleep(30)
    if args.mode == 'failure':
        raise ValueError('provider failed')
    if args.mode == 'crash':
        os._exit(17)
    if args.mode == 'shutdown':
        os.kill(os.getppid(), signal.SIGTERM)
        await asyncio.sleep(0.05)
    return {'pid': os.getpid(), 'failed_metrics': ['v'], 'payload': 'x' * 1000000}

module = ModuleType('clio.commands.aggregate_solar_wind')
module.run = run
sys.modules[module.__name__] = module

if __name__ == '__main__':
    from clio.scheduling.execution import invoke_isolated
    mode = sys.argv[1]
    stopped = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.append(True))
    abort = threading.Event()
    if mode == 'cancel':
        threading.Timer(0.5, abort.set).start()
    try:
        result = invoke_isolated('aggregate', SimpleNamespace(mode=mode), abort=abort)
    except RuntimeError as error:
        assert mode != 'success'
        assert {'failure': 'provider failed', 'crash': 'without a report (17)',
                'cancel': 'cancelled at shutdown deadline'}[mode] in str(error)
    else:
        assert mode in ('success', 'shutdown')
        assert result['pid'] != os.getpid()
        assert result['failed_metrics'] == ['v']
        assert len(result['payload']) == 1000000
        try:
            os.kill(result['pid'], 0)
        except ProcessLookupError:
            pass
        else:
            raise AssertionError('executor was not reaped')
    assert bool(stopped) == (mode == 'shutdown')
    assert not multiprocessing.active_children()
""")
    result = subprocess.run([sys.executable, str(script), mode],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
