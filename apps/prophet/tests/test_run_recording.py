from unittest.mock import Mock, call
from types import SimpleNamespace

import pytest

from argus_prophet.services.generation import cycle
from argus_prophet.services.generation.products import PRODUCTS


def setup_run():
    from argus_prophet.config import ProphetConfig
    from argus_prophet.services.generation.service import ForecastGenerationService
    recorders = {name: Mock(run_id=name) for name in PRODUCTS}
    runtime = SimpleNamespace(workdir='.', models_registry={'models': {}})
    policy = ProphetConfig()
    begin = Mock(side_effect=lambda name: cycle.ProductRun(name, recorders[name], runtime, policy))
    inputs = SimpleNamespace(observations=SimpleNamespace(points=[]))
    read = Mock(return_value=inputs)
    command = Mock(return_value=[])
    service = ForecastGenerationService(
        input_loader=read, calculator=command,
        executor=lambda target, *args, **kw: target(*args),
        run_factory=begin, before_dispatch=lambda: None,
        timeout_seconds=policy.calculation_timeout_seconds,
    )
    return recorders, inputs, command, read, begin, service


def test_full_cycle_reads_once_and_publishes_each_product_before_next():
    recorders, inputs, command, read, begin, service = setup_run()
    seen = []
    def compute(request):
        name = request.product
        recorders[name].snapshot.assert_called_once_with(inputs)
        if seen:
            recorders[seen[-1]].finish.assert_called_once_with()
        seen.append(name)
        return []
    command.side_effect = compute
    service.generate(tuple(PRODUCTS))
    read.assert_called_once_with()
    assert seen == list(PRODUCTS)
    for name, recorder in recorders.items():
        recorder.snapshot.assert_called_once_with(inputs)
        recorder.finish.assert_called_once_with()
    assert [c.args[0] for c in begin.call_args_list] == list(PRODUCTS)


@pytest.mark.parametrize('failed_product', ['geomagnetic-activity', 'atmospheric-density'])
def test_model_failure_does_not_block_other_products(failed_product):
    recorders, _, command, read, _, service = setup_run()
    failure = ValueError('model failed')
    def compute(request):
        name = request.product
        if name == failed_product:
            raise failure
        return []
    command.side_effect = compute
    with pytest.raises(cycle.GenerationFailed) as error:
        service.generate(tuple(PRODUCTS))
    assert error.value.failures == {failed_product: failure}
    assert f"{failed_product}: model failed" in str(error.value)
    assert command.call_count == len(PRODUCTS)
    read.assert_called_once_with()
    for name, recorder in recorders.items():
        if name == failed_product:
            recorder.finish.assert_called_once_with(error=failure)
        else:
            recorder.finish.assert_called_once_with()


def test_publication_failure_is_recorded_and_other_products_continue():
    recorders, _, command, _, _, service = setup_run()
    failure = ValueError('checksum mismatch')
    recorders['dst'].finish.side_effect = [failure, None]
    with pytest.raises(cycle.GenerationFailed):
        service.generate(tuple(PRODUCTS))
    assert recorders['dst'].finish.call_args_list == [call(), call(error=failure)]
    assert command.call_count == len(PRODUCTS)


def test_snapshot_failure_prevents_that_product_calculation():
    recorders, _, command, _, _, service = setup_run()
    failure = RuntimeError('database unavailable')
    recorders['dst'].snapshot.side_effect = failure
    with pytest.raises(cycle.GenerationFailed):
        service.generate(('dst',))
    recorders['dst'].finish.assert_called_once_with(error=failure)
    command.assert_not_called()


def test_lost_writer_stops_cycle_without_starting_more_products():
    recorders, _, command, _, begin, service = setup_run()
    first = next(iter(PRODUCTS))
    command.side_effect = RuntimeError('connection lost')
    recorders[first].finish.side_effect = RuntimeError('lock lost')
    with pytest.raises(RuntimeError, match='lock lost'):
        service.generate(tuple(PRODUCTS))
    assert begin.call_count == 1


def test_shared_input_failure_records_all_selected_products_and_reads_once():
    recorders, _, command, read, _, service = setup_run()
    failure = RuntimeError('Clio unavailable')
    read.side_effect = failure
    with pytest.raises(cycle.GenerationFailed) as error:
        service.generate(tuple(PRODUCTS))
    assert set(error.value.failures) == set(PRODUCTS)
    read.assert_called_once_with()
    command.assert_not_called()
    for recorder in recorders.values():
        recorder.snapshot.assert_not_called()
        recorder.finish.assert_called_once_with(error=failure)


def test_retry_subset_does_not_create_successful_product_attempts():
    recorders, inputs, command, read, begin, service = setup_run()
    service.generate(('dst', 'atmospheric-density'))
    assert [c.args[0] for c in begin.call_args_list] == ['dst', 'atmospheric-density']
    read.assert_called_once_with()
    recorders['solar-wind-speed'].finish.assert_not_called()


def test_manual_input_preparation_uses_supervised_timeout():
    recorders, _, command, read, begin, service = setup_run()
    timeout = TimeoutError('input preparation exceeded budget')
    execute = Mock(side_effect=timeout)
    service.executor = execute
    with pytest.raises(cycle.GenerationFailed) as error:
        service.generate(tuple(PRODUCTS))
    execute.assert_called_once_with(read, timeout_seconds=540)
    assert set(error.value.failures.values()) == {timeout}
    command.assert_not_called()
    for recorder in recorders.values():
        recorder.finish.assert_called_once_with(error=timeout)


def test_manual_shutdown_records_interruption_and_stops_dispatch():
    from argus_prophet.scheduling.execution import ShutdownRequested
    recorders, _, command, _, begin, service = setup_run()
    interruption = ShutdownRequested('shutdown deadline')
    service.executor = Mock(side_effect=interruption)
    with pytest.raises(ShutdownRequested):
        service.generate(tuple(PRODUCTS))
    assert begin.call_count == 1
    recorders[next(iter(PRODUCTS))].finish.assert_called_once_with(error=interruption)
    command.assert_not_called()


@pytest.mark.parametrize('products', [(), ('dst', 'dst'), ('unsupported',)])
def test_invalid_selection_has_no_side_effects(products):
    _, _, command, read, begin, service = setup_run()
    with pytest.raises(ValueError, match='distinct supported'):
        service.generate(products)
    begin.assert_not_called()
    read.assert_not_called()
    command.assert_not_called()


def test_reusing_service_loads_a_new_snapshot_per_cycle():
    _, inputs, command, read, _, service = setup_run()
    next_inputs = SimpleNamespace(observations=SimpleNamespace(points=[]))
    read.side_effect = [inputs, next_inputs]
    service.generate(('dst',))
    service.generate(('dst',))
    assert read.call_count == 2
    assert command.call_count == 2
    request = command.call_args.args[0]
    assert request.product == 'dst'
    assert command.call_args_list[0].args[0].inputs is inputs
    assert request.inputs is next_inputs


def test_shutdown_after_loading_inputs_does_not_record_snapshot_or_calculate():
    from argus_prophet.scheduling.execution import ShutdownRequested
    recorders, _, command, read, begin, service = setup_run()
    interruption = ShutdownRequested('stopped after inputs')
    service.before_dispatch = Mock(side_effect=[None, interruption])
    with pytest.raises(ShutdownRequested):
        service.generate(('dst',))
    read.assert_called_once()
    recorders['dst'].snapshot.assert_not_called()
    recorders['dst'].finish.assert_called_once_with(error=interruption)
    command.assert_not_called()


def test_manual_composition_preserves_writer_and_trigger(monkeypatch):
    from common import config
    from argus_prophet.services import inputs as observations
    from argus_prophet.services import runs as ledger
    runtime = SimpleNamespace(workdir='.', models_registry={'models': {}})
    monkeypatch.setattr(config, 'get_config', lambda: runtime)
    monkeypatch.setattr(cycle, 'provenance', lambda _: {'source': 'test'})
    monkeypatch.setattr(cycle, 'execute', lambda target, *args, **kw: target(*args))
    inputs = SimpleNamespace(observations=SimpleNamespace(points=[]))
    monkeypatch.setattr(observations, 'load_inputs', lambda: inputs)
    calculate = Mock(return_value=[])
    monkeypatch.setattr(cycle, 'calculate_product', calculate)
    recorder = Mock()
    begin = Mock(return_value=recorder)
    monkeypatch.setattr(ledger.RunRecorder, 'begin', begin)
    writer = object()
    cycle.generate('dst', writer=writer)
    assert begin.call_args.args[:3] == ('dst', 'manual', runtime)
    assert begin.call_args.kwargs['writer'] is writer
    assert begin.call_args.kwargs['details'] == {'source': 'test'}
    recorder.snapshot.assert_called_once_with(inputs)
    recorder.finish.assert_called_once_with()
    assert calculate.call_args.args[0].product == 'dst'


def test_calculation_uses_configured_timeout_and_only_serializable_request():
    from argus_prophet.services.generation.calculation import CalculationRequest
    _, inputs, command, read, _, service = setup_run()
    executor = Mock(side_effect=lambda target, *args, **kw: target(*args))
    service.executor = executor
    service.timeout_seconds = 17
    service.generate(('dst',))
    assert executor.call_count == 2
    assert executor.call_args_list[0] == call(read, timeout_seconds=17)
    target, request = executor.call_args_list[1].args
    assert target is command
    assert isinstance(request, CalculationRequest)
    assert request.inputs is inputs
    assert executor.call_args_list[1].kwargs == {'timeout_seconds': 17}
