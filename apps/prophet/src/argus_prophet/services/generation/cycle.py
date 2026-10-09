"""Shared product lifecycle for manual cycles and the scheduled dispatcher."""
import logging
from dataclasses import dataclass
from functools import partial

from common.schemas.forecast_inputs import ForecastInputs

from argus_prophet.config import InputPolicy, ProphetConfig
from argus_prophet.services.generation.products import select_products, PRODUCTS
from argus_prophet.services.generation.calculation import CalculationRequest, calculate_serialized as calculate_product
from argus_prophet.services.runs import RunRecorder, provenance
from argus_prophet.scheduling.execution import execute, before_dispatch
from argus_prophet.services.generation.service import ForecastGenerationService, GenerationFailed

logger = logging.getLogger(__name__)


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

    def prepare_calculation(self, inputs: ForecastInputs) -> CalculationRequest:
        """Persist and validate inputs before dispatching model calculation."""
        self.recorder.snapshot(inputs)
        self.policy.inputs.get(self.product, InputPolicy()).validate_inputs(inputs)
        return CalculationRequest(self.product, inputs, self.runtime.workdir, self.runtime.models_registry['models'])

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
    """Composition root for manual generation; bind infrastructure to the use case."""
    from common.config import get_config
    from argus_prophet.services.inputs import load_inputs
    if not products or len(set(products)) != len(products) or any(name not in PRODUCTS for name in products):
        raise ValueError('Expected distinct supported forecast products')
    runtime = get_config()
    policy = ProphetConfig.model_validate(getattr(runtime, 'project_config', {}).get('prophet', {}))
    details = provenance(runtime)
    service = ForecastGenerationService(
        input_loader=partial(load_inputs, prepare_proswin=True) if {'solar-wind-speed', 'solar-wind-density'}.intersection(products) else load_inputs,
        calculator=calculate_product,
        executor=execute,
        run_factory=partial(ProductRun.begin, trigger='manual', runtime=runtime,
                            policy=policy, writer=writer, details=details),
        before_dispatch=before_dispatch,
        timeout_seconds=policy.calculation_timeout_seconds,
    )
    service.generate(products)
