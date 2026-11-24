"""One-shot product generation and publication."""
import logging
import os
from dataclasses import dataclass
from functools import partial

from forecast.inputs import ForecastInputs

from prophet.config import InputPolicy, ProphetConfig
from prophet.services.generation.products import select_products, PRODUCTS
from prophet.services.generation.calculation import CalculationRequest, calculate_serialized as calculate_product
from prophet.services.releases.recording import RunRecorder, provenance
from prophet.execution import execute, before_dispatch
from prophet.services.generation.service import ForecastGenerationService, GenerationFailed

logger = logging.getLogger(__name__)


@dataclass
class ProductRun:
    product: str
    recorder: RunRecorder
    runtime: object
    policy: ProphetConfig

    @classmethod
    async def begin(cls, product, trigger, runtime, policy, *, writer, details=None):
        recorder = await RunRecorder.begin(product, trigger, runtime,
                                     writer=writer, details=details)
        logger.info('Prophet run %s started (%s)', recorder.run_id, product)
        return cls(product, recorder, runtime, policy)

    async def prepare_calculation(self, inputs: ForecastInputs) -> CalculationRequest:
        """Persist and validate inputs before dispatching model calculation."""
        await self.recorder.snapshot(inputs)
        self.policy.inputs.get(self.product, InputPolicy()).validate_inputs(inputs)
        return CalculationRequest(self.product, inputs, self.runtime.workdir, self.runtime.models_registry['models'])

    async def complete(self, artifacts):
        for artifact in artifacts:
            await self.recorder.store(artifact.name, artifact.content, artifact.model_info,
                                artifact.row_count, artifact.columns)
        publication = await self.recorder.finish()
        if publication.status == 'published':
            logger.info('Prophet run %s published %s (%s)', self.recorder.run_id, publication.release_id, self.product)
        else:
            logger.info('Prophet run %s kept the current release (%s): %s',
                        self.recorder.run_id, self.product, publication.reason)

    async def fail(self, error):
        # Failure to record the outcome is fatal: the writer may have lost its lock.
        await self.recorder.finish(error=error)
        logger.warning('Prophet run %s failed (%s): %s', self.recorder.run_id, self.product, error)


async def generate(product: str, *, writer) -> None:
    await generate_products(select_products(product), writer=writer)


async def generate_products(products, *, writer) -> None:
    """Composition root for manual generation; bind infrastructure to the use case."""
    from common.config import get_config
    from prophet.services.inputs.loader import load_product_inputs
    if not products or len(set(products)) != len(products) or any(name not in PRODUCTS for name in products):
        raise ValueError('Expected distinct supported forecast products')
    runtime = get_config()
    policy = ProphetConfig.model_validate(getattr(runtime, 'project_config', {}).get('prophet', {}))
    details = provenance(runtime)
    service = ForecastGenerationService(
        input_loader=partial(load_product_inputs, executor=execute, timeout_seconds=policy.calculation_timeout_seconds),
        calculator=calculate_product,
        executor=execute,
        run_factory=partial(ProductRun.begin, trigger='scheduled' if os.getenv('ARGUS_SCHEDULED_JOB') else 'manual', runtime=runtime,
                            policy=policy, writer=writer, details=details),
        before_dispatch=before_dispatch,
        timeout_seconds=policy.calculation_timeout_seconds,
    )
    await service.generate(products)
