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
        await self.recorder.finish()
        logger.info('Prophet run %s published (%s)', self.recorder.run_id, self.product)

    async def fail(self, error):
        # Failure to record the outcome is fatal: the writer may have lost its lock.
        await self.recorder.finish(error=error)
        logger.warning('Prophet run %s failed (%s): %s', self.recorder.run_id, self.product, error)


async def generate(product: str, *, writer) -> None:
    await generate_products(select_products(product), writer=writer)


async def generate_products(products, *, writer) -> None:
    """Composition root for manual generation; bind infrastructure to the use case."""
    from common.config import get_config
    from prophet.services.inputs.loader import load_inputs, load_event_inputs
    if not products or len(set(products)) != len(products) or any(name not in PRODUCTS for name in products):
        raise ValueError('Expected distinct supported forecast products')
    runtime = get_config()
    policy = ProphetConfig.model_validate(getattr(runtime, 'project_config', {}).get('prophet', {}))
    details = provenance(runtime)
    loader = partial(load_event_inputs, include_speed=True) if tuple(products) == ('solar-wind-events',) else load_inputs
    options = {}
    if {'solar-wind-speed', 'solar-wind-density', 'solar-wind-temperature'}.intersection(products):
        options['prepare_proswin'] = True
    if {'solar-wind-events', 'solar-wind-speed'}.intersection(products) and loader is load_inputs:
        options['include_events'] = True
    service = ForecastGenerationService(
        input_loader=partial(loader, **options) if options else loader,
        calculator=calculate_product,
        executor=execute,
        run_factory=partial(ProductRun.begin, trigger='scheduled' if os.getenv('ARGUS_CRON_JOB') else 'manual', runtime=runtime,
                            policy=policy, writer=writer, details=details),
        before_dispatch=before_dispatch,
        timeout_seconds=policy.calculation_timeout_seconds,
    )
    await service.generate(products)
