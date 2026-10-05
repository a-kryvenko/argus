"""Shared product lifecycle for manual cycles and the scheduled dispatcher."""
import logging
from dataclasses import dataclass

from argus_prophet.config import InputPolicy, ProphetConfig
from argus_prophet.services.generation.products import select_products, PRODUCTS
from argus_prophet.services.generation.calculation import calculate_product
from argus_prophet.services.runs import RunRecorder, provenance
from argus_prophet.scheduling.execution import execute, before_dispatch

logger = logging.getLogger(__name__)


class GenerationFailed(RuntimeError):
    def __init__(self, failures):
        self.failures = failures
        super().__init__('Forecast generation failed: ' + '; '.join(
            f'{product}: {error}' for product, error in failures.items()))


@dataclass
class ProductRun:
    product: str
    recorder: RunRecorder
    runtime: object
    policy: ProphetConfig

    @classmethod
    def begin(cls, product, trigger, runtime, policy, *, writer, scheduled_slot=None, details=None):
        recorder = RunRecorder.begin(product, trigger, runtime,
                                     writer=writer, scheduled_slot=scheduled_slot, details=details)
        logger.info('Prophet run %s started (%s)', recorder.run_id, product)
        return cls(product, recorder, runtime, policy)

    def prepare(self, inputs):
        before_dispatch()
        self.recorder.snapshot(inputs)
        self.policy.inputs.get(self.product, InputPolicy()).validate_inputs(inputs)
        return (self.product, inputs, self.runtime.workdir, self.runtime.models_registry['models'])

    def complete(self, artifacts):
        for artifact in artifacts:
            self.recorder.store(artifact.name, artifact.content, artifact.model_info,
                                artifact.row_count, artifact.columns)
        self.recorder.finish()
        logger.info('Prophet run %s published (%s)', self.recorder.run_id, self.product)

    def fail(self, error):
        # Failure to record the outcome is fatal: the writer may have lost its lock.
        self.recorder.finish(error=error)
        logger.warning('Prophet run %s failed (%s): %s', self.recorder.run_id, self.product, error)


def generate(product: str, *, writer) -> None:
    generate_products(select_products(product), writer=writer)


def generate_products(products, *, writer) -> None:
    from common.config import get_config
    from argus_prophet.services.inputs import load_inputs
    if not products or len(set(products)) != len(products) or any(name not in PRODUCTS for name in products):
        raise ValueError('Expected distinct supported forecast products')
    runtime = get_config()
    policy = ProphetConfig.model_validate(getattr(runtime, 'project_config', {}).get('prophet', {}))
    details = provenance(runtime)
    inputs = None
    input_error = None
    failures = {}
    for product in products:
        before_dispatch()
        run = ProductRun.begin(product, 'manual', runtime, policy, writer=writer, details=details)
        try:
            if inputs is None and input_error is None:
                try:
                    inputs = execute(load_inputs, timeout_seconds=policy.calculation_timeout_seconds)
                except Exception as exc:
                    input_error = exc
            if input_error is not None:
                raise input_error
            artifacts = execute(calculate_product, *run.prepare(inputs),
                                timeout_seconds=policy.calculation_timeout_seconds)
            run.complete(artifacts)
        except BaseException as exc:
            run.fail(exc)
            if not isinstance(exc, Exception):
                raise
            failures[product] = exc
    if failures:
        raise GenerationFailed(failures)
