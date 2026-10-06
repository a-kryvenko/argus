"""Dispatch independent product schedules; persist results on the owning thread."""
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from time import monotonic, sleep

from argus_prophet.config import ProductSchedule, load_config
from argus_prophet.scheduling.execution import ShutdownRequested, execute, supervise
from argus_prophet.scheduling.jobs import GenerationBusy, generation_lock, product_pending
from argus_prophet.scheduling.verification import run_due as verify_due
from argus_prophet.services.generation.calculation import calculate_product
from argus_prophet.services.generation.products import PRODUCTS
from argus_prophet.services.inputs import load_inputs
from argus_prophet.services.generation.cycle import ProductRun
from argus_prophet.services.runs import provenance

logger = logging.getLogger(__name__)


@dataclass
class Task:
    product: str
    schedule: ProductSchedule
    next_check: float = 0


@dataclass
class Active:
    run: ProductRun
    future: Future
    phase: str = 'inputs'


class Dispatcher:
    def __init__(self, config, runtime, control, pool):
        self.config, self.runtime, self.control, self.pool = config, runtime, control, pool
        self.tasks = [Task(name, config.schedules.get(name, ProductSchedule())) for name in PRODUCTS]
        self.active = {}
        self.writer = ExitStack()
        self.connection = None
        self.verification = None
        self.verify_after = 0
        self.details = provenance(runtime)

    def submit(self, target, *args, timeout=None):
        return self.pool.submit(execute, target, *args, control=self.control,
                                timeout_seconds=timeout or self.config.calculation_timeout_seconds)

    def finish_tasks(self):
        for product, task in list(self.active.items()):
            if not task.future.done():
                continue
            try:
                result = task.future.result()
                if task.phase == 'inputs':
                    self.control.before_dispatch()
                    task.future = self.submit(calculate_product, task.run.prepare_calculation(result))
                    task.phase = 'calculation'
                    continue
                task.run.complete(result)
            except (Exception, ShutdownRequested) as exc:
                task.run.fail(exc)
            del self.active[product]
        if not self.active:
            self.writer.close()
            self.connection = None
        if self.verification is not None and self.verification.done():
            try:
                if self.verification.result():
                    logger.info('Scheduled forecast verification completed')
            except (Exception, ShutdownRequested):
                logger.exception('Scheduled verification failed; retrying in 60 seconds')
            self.verification = None
            self.verify_after = monotonic() + 60

    def launch_due(self, now, clock):
        self.control.before_dispatch()
        inputs = None
        # Oldest check first: frequent products cannot starve other products.
        for task in sorted(self.tasks, key=lambda item: item.next_check):
            if len(self.active) >= self.config.max_parallel_products:
                break
            if task.product in self.active or task.next_check > clock:
                continue
            task.next_check = clock + 60
            if self.connection is None:
                try:
                    self.connection = self.writer.enter_context(generation_lock())
                except GenerationBusy:
                    continue
            slot = task.schedule.due_slot(now)
            if not product_pending(task.product, slot, writer=self.connection):
                continue
            run = ProductRun.begin(task.product, 'scheduled', self.runtime, self.config,
                                   writer=self.connection, scheduled_slot=slot, details=self.details)
            if inputs is None:
                inputs = self.submit(load_inputs)
            self.active[task.product] = Active(run, inputs)
        if not self.active:
            self.writer.close()
            self.connection = None
        if self.config.verification.enabled and self.verification is None and clock >= self.verify_after:
            self.verification = self.submit(verify_due, self.config.verification,
                                            timeout=self.config.verification.timeout_seconds + 30)

    def close(self):
        # Keep the generation lock until every calculation has been reaped.
        self.control.cancel()
        self.pool.shutdown(wait=True)
        self.writer.close()


def work():
    from common.config import get_config
    config = load_config()
    with supervise(config.shutdown_grace_seconds) as control:
        pool = ThreadPoolExecutor(max_workers=config.max_parallel_products + 1)
        dispatcher = Dispatcher(config, get_config(), control, pool)
        try:
            while not control.stopped.is_set() or dispatcher.active or dispatcher.verification is not None:
                dispatcher.finish_tasks()
                if not control.stopped.is_set():
                    try:
                        dispatcher.launch_due(datetime.now(UTC), monotonic())
                    except ShutdownRequested:
                        pass
                # Poll completed calculations during shutdown as well; Event.wait
                # would return immediately once stopped and cause a busy loop.
                sleep(0.2)
        finally:
            dispatcher.close()
