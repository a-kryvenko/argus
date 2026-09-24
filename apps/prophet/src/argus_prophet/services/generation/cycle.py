"""Generate a product cycle from one input snapshot and record each outcome."""
import logging

from argus_prophet.services.generation.products import select_products


class GenerationFailed(RuntimeError):
    def __init__(self, failures):
        self.failures = failures
        super().__init__('Forecast generation failed for: ' + ', '.join(failures))


def generate(product: str, trigger='manual', *, scheduled_slot=None) -> None:
    generate_products(select_products(product), trigger, scheduled_slot=scheduled_slot)


def generate_products(products, trigger='manual', *, scheduled_slot=None) -> None:
    from argus_prophet.services.generation.products import PRODUCTS
    from argus_prophet.services.generation.calculation import calculate
    from common.config import get_config
    from argus_prophet.services.runs import RunRecorder, provenance
    from argus_prophet.services.inputs import load_inputs
    if not products or len(set(products)) != len(products) or any(name not in PRODUCTS for name in products):
        raise ValueError('Expected distinct supported forecast products')
    config = get_config()
    details = provenance(config)
    inputs = None
    input_error = None
    failures = {}
    for product in products:
        recorder = RunRecorder.begin(product, trigger, config, scheduled_slot=scheduled_slot,
                                     details=details)
        logging.info('Prophet run %s started (%s)', recorder.run_id, product)
        try:
            if inputs is None and input_error is None:
                try:
                    inputs = load_inputs()
                except Exception as exc:
                    input_error = exc
            if input_error is not None:
                raise input_error
            recorder.snapshot(inputs)
            calculate(product, inputs=inputs, recorder=recorder)
            recorder.finish()
        except BaseException as exc:
            # If failure recording also fails (e.g. lost lock/DB connection), stop.
            # A new writer must recover this attempt before any more work starts.
            recorder.finish(error=exc)
            if not isinstance(exc, Exception):
                raise
            failures[product] = exc
            logging.exception('Prophet run %s failed (%s)', recorder.run_id, product)
        else:
            logging.info('Prophet run %s published (%s)', recorder.run_id, product)
    if failures:
        raise GenerationFailed(failures)
