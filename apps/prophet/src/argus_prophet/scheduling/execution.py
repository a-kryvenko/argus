"""Supervise bounded child tasks; generation keeps its writer in the caller."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import logging
import multiprocessing
import signal
import threading
from time import monotonic
import traceback

logger = logging.getLogger(__name__)


class ShutdownRequested(BaseException):
    """Stop the cycle without dispatching another product."""


@dataclass
class ExecutionControl:
    grace_seconds: float = 570
    stopped: threading.Event = field(default_factory=lambda: threading.Event())
    deadline: float | None = None

    def stop(self):
        if self.deadline is None:
            self.deadline = monotonic() + self.grace_seconds
        self.stopped.set()

    def cancel(self):
        self.deadline = monotonic()
        self.stopped.set()

    def before_dispatch(self):
        if self.stopped.is_set():
            raise ShutdownRequested('Prophet shutdown requested')

    def check_deadline(self):
        if self.deadline is not None and monotonic() >= self.deadline:
            raise ShutdownRequested('Prophet shutdown deadline reached')


_control = ContextVar('prophet_execution_control', default=None)


def before_dispatch():
    if (control := _control.get()) is not None:
        control.before_dispatch()


@contextmanager
def supervise(grace_seconds=570):
    control = ExecutionControl(grace_seconds=grace_seconds)
    token = _control.set(control)
    previous = {sig: signal.signal(sig, lambda *_: control.stop())
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield control
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        _control.reset(token)


def _invoke(sender, target, args, kwargs):
    try:
        # Group SIGINT belongs to the coordinator. Termination remains available
        # after its grace period, and spawn never inherits a writer connection.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        sender.send((True, target(*args, **kwargs)))
    except BaseException:
        sender.send((False, traceback.format_exc()))
    finally:
        sender.close()


def execute(target, *args, timeout_seconds=540, control=None, **kwargs):
    control = control if control is not None else _control.get()
    if control is not None:
        control.before_dispatch()
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_invoke, args=(sender, target, args, kwargs))
    deadline = monotonic() + timeout_seconds

    def check():
        if control is not None:
            control.check_deadline()
        if monotonic() >= deadline:
            raise TimeoutError(f'Calculation exceeded {timeout_seconds}s')

    try:
        process.start()
        sender.close()
        try:
            while not receiver.poll(0.1):
                check()
            succeeded, result = receiver.recv()
        except EOFError as exc:
            raise RuntimeError('Calculation process exited without a result') from exc
        while process.is_alive():
            process.join(timeout=0.1)
            check()
        if process.exitcode != 0 or not succeeded:
            raise RuntimeError(f'Calculation process failed ({process.exitcode}): {result}')
        return result
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()
