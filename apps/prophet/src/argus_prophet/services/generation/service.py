"""Manual generation use case with explicitly supplied infrastructure."""
from collections.abc import Callable, Sequence
from typing import Protocol, TypeVar

from common.schemas.forecast_inputs import ForecastInputs
from argus_prophet.services.generation.calculation import Artifact, CalculationRequest
from argus_prophet.services.generation.products import PRODUCTS

T = TypeVar('T')


class Executor(Protocol):
    def __call__(self, target: Callable[..., T], *args: object,
                 timeout_seconds: float) -> T: ...


class GenerationRun(Protocol):
    def prepare_calculation(self, inputs: ForecastInputs) -> CalculationRequest:
        """Record the input snapshot, validate policy and create the request."""
        ...

    def complete(self, artifacts: list[Artifact]) -> None:
        """Store artifacts and transactionally publish the completed run."""
        ...

    def fail(self, error: BaseException) -> None: ...


class GenerationFailed(RuntimeError):
    def __init__(self, failures: dict[str, Exception]):
        self.failures = failures
        super().__init__('Forecast generation failed: ' + '; '.join(
            f'{product}: {error}' for product, error in failures.items()))


class ForecastGenerationService:
    """Coordinate independent product attempts sharing one input snapshot.

    Run creation and recording stay in the caller. Only the supplied input loader
    and calculator are passed to the executor, which owns process supervision.
    All per-cycle state is local so the service can be reused safely.
    """

    def __init__(self, *, input_loader: Callable[[], ForecastInputs],
                 calculator: Callable[[CalculationRequest], list[Artifact]],
                 executor: Executor, run_factory: Callable[[str], GenerationRun],
                 before_dispatch: Callable[[], None], timeout_seconds: float):
        self.input_loader = input_loader
        self.calculator = calculator
        self.executor = executor
        self.run_factory = run_factory
        self.before_dispatch = before_dispatch
        self.timeout_seconds = timeout_seconds

    def generate(self, products: Sequence[str]) -> None:
        if not products or len(set(products)) != len(products) or any(name not in PRODUCTS for name in products):
            raise ValueError('Expected distinct supported forecast products')
        inputs = None
        input_error = None
        failures = {}
        for product in products:
            self.before_dispatch()
            run = self.run_factory(product)
            try:
                if inputs is None and input_error is None:
                    try:
                        inputs = self.executor(self.input_loader, timeout_seconds=self.timeout_seconds)
                    except Exception as exc:
                        input_error = exc
                if input_error is not None:
                    raise input_error
                self.before_dispatch()
                request = run.prepare_calculation(inputs)
                artifacts = self.executor(self.calculator, request, timeout_seconds=self.timeout_seconds)
                run.complete(artifacts)
            except BaseException as exc:
                # A failed writer must abort the cycle instead of starting more runs.
                run.fail(exc)
                if not isinstance(exc, Exception):
                    raise
                failures[product] = exc
        if failures:
            raise GenerationFailed(failures)
