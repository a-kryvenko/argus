"""Run occasional jobs in fresh interpreters; retain only their reports."""
import logging
import multiprocessing
import traceback


def _invoke(sender, name, args):
    try:
        from common.runtime import run_command
        from clio.cli import invoke
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        result = run_command(lambda: invoke(name, args))
        sender.send((True, result))
    except BaseException:
        sender.send((False, traceback.format_exc()))
    finally:
        sender.close()


def invoke_isolated(name, args, *, abort=None):
    # Spawn avoids inheriting database connections or scientific-library state.
    # The scheduling process retains its advisory lock until this process exits.
    context = multiprocessing.get_context('spawn')
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_invoke, args=(sender, name, args))
    try:
        process.start()
        sender.close()
        try:
            while not receiver.poll(0.2):
                if abort is not None and abort.is_set():
                    raise RuntimeError(f'Clio {name} executor cancelled at shutdown deadline')
            succeeded, result = receiver.recv()
        except EOFError as exc:
            process.join()
            raise RuntimeError(f'Clio {name} executor exited without a report ({process.exitcode})') from exc
        while process.is_alive():
            process.join(timeout=0.2)
            if abort is not None and abort.is_set():
                raise RuntimeError(f'Clio {name} executor cancelled at shutdown deadline')
        if process.exitcode != 0 or not succeeded:
            raise RuntimeError(f'Clio {name} executor failed ({process.exitcode}): {result}')
        return result
    finally:
        sender.close()
        receiver.close()
        if process.pid is not None:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
            process.close()
