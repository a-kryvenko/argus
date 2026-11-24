"""Manual generation use case with explicitly supplied infrastructure."""
from collections.abc import Awaitable, Callable, Sequence
import asyncio
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from forecast.inputs import ForecastInputs
from prophet.services.generation.calculation import Artifact, CalculationRequest
from prophet.services.generation.products import PRODUCTS

T = TypeVar('T')


class Executor(Protocol):
    def __call__(self, target: Callable[..., T], *args: object,
                 timeout_seconds: float) -> T: ...


class GenerationRun(Protocol):
    async def prepare_calculation(self, inputs: ForecastInputs) -> CalculationRequest:
        """Record the input snapshot, validate policy and create the request."""
        ...

    async def complete(self, artifacts: list[Artifact]) -> None:
        """Store artifacts and transactionally publish the completed run."""
        ...

    async def fail(self, error: BaseException) -> None: ...


class GenerationFailed(RuntimeError):
    def __init__(self, failures: dict[str, Exception]):
        self.failures = failures
        super().__init__('Forecast generation failed: ' + '; '.join(
            f'{product}: {error}' for product, error in failures.items()))


class ForecastGenerationService:
    """Coordinate independent product attempts sharing an input cutoff and source reads.

    Run creation and recording stay in the caller.
    All per-cycle state is local so the service can be reused safely.
    """

    def __init__(self, *, input_loader: Callable[[str, datetime, dict], Awaitable[ForecastInputs]],
                 calculator: Callable[[CalculationRequest], list[Artifact]],
                 executor: Executor, run_factory: Callable[[str], Awaitable[GenerationRun]],
                 before_dispatch: Callable[[], None], timeout_seconds: float):
        self.input_loader = input_loader
        self.calculator = calculator
        self.executor = executor
        self.run_factory = run_factory
        self.before_dispatch = before_dispatch
        self.timeout_seconds = timeout_seconds

    async def generate(self, products: Sequence[str]) -> None:
        if not products or len(set(products)) != len(products) or any(name not in PRODUCTS for name in products):
            raise ValueError('Expected distinct supported forecast products')
        as_of = datetime.now(UTC)
        source_cache = {}
        failures = {}
        for product in products:
            self.before_dispatch()
            run = await self.run_factory(product)
            try:
                inputs = await self.input_loader(product, as_of, source_cache)
                self.before_dispatch()
                request = await run.prepare_calculation(inputs)
                artifacts = await asyncio.to_thread(self.executor, self.calculator, request, timeout_seconds=self.timeout_seconds)
                await run.complete(artifacts)
            except BaseException as exc:
                # A failed writer must abort the cycle instead of starting more runs.
                await run.fail(exc)
                if not isinstance(exc, Exception):
                    raise
                failures[product] = exc
        if failures:
            raise GenerationFailed(failures)
