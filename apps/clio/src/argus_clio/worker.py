"""Run Clio's existing collectors and schedules as one supervised service."""
import logging
import signal
import subprocess
import sys
import threading
from time import monotonic

logger = logging.getLogger(__name__)
COMMANDS = (
    ('collect', 'solar-wind', '--watch'),
    ('collect', 'geomagnetic', '--watch'),
    ('schedule', 'refresh'),
    ('schedule', 'aggregate'),
)
# Leave time to reap children before Compose's ten-minute stop deadline.
STOP_TIMEOUT_SECONDS = 570


def stop_children(children):
    for child in children:
        if child.poll() is None:
            child.terminate()
    deadline = monotonic() + STOP_TIMEOUT_SECONDS
    for child in children:
        try:
            child.wait(timeout=max(0, deadline - monotonic()))
        except subprocess.TimeoutExpired:
            logger.error('Clio child %s exceeded shutdown deadline; killing it', child.pid)
            child.kill()
            child.wait()


def work():
    stopped = threading.Event()
    previous = {sig: signal.signal(sig, lambda *_: stopped.set())
                for sig in (signal.SIGTERM, signal.SIGINT)}
    children = []
    try:
        for command in COMMANDS:
            if stopped.is_set():
                break
            children.append(subprocess.Popen(
                [sys.executable, '-m', 'argus_clio.cli', *command],
                start_new_session=True,
            ))
            logger.info('Started Clio %s (pid %s)', ' '.join(command), children[-1].pid)
        while not stopped.wait(1):
            for command, child in zip(COMMANDS, children):
                code = child.poll()
                if code is not None:
                    # Fail the whole service so Docker restarts it. Do not keep
                    # a silently incomplete worker or add another retry system.
                    raise RuntimeError(f'Clio {" ".join(command)} exited unexpectedly ({code})')
    finally:
        try:
            stop_children(children)
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
