from unittest.mock import Mock, call
from types import SimpleNamespace

import pytest

from argus_prophet.services.generation import cycle
from argus_prophet.services import runs as ledger
from argus_prophet.services import inputs as observations
from argus_prophet.services.generation.products import PRODUCTS


def setup_run(monkeypatch):
    recorders = {name: Mock(run_id=name) for name in PRODUCTS}
    begin = Mock(side_effect=lambda name, *args, **kwargs: recorders[name])
    monkeypatch.setattr(ledger.RunRecorder, 'begin', begin)
    monkeypatch.setattr(cycle, 'provenance', lambda _: {})
    monkeypatch.setattr(cycle, 'execute', lambda target, *args, **kw: target(*args))
    inputs = SimpleNamespace(observations=SimpleNamespace(points=[]))
    from common import config
    monkeypatch.setattr(config, 'get_config', lambda: SimpleNamespace(workdir='.', models_registry={'models': {}}))
    read = Mock(return_value=inputs)
    monkeypatch.setattr(observations, 'load_inputs', read)
    command = Mock(return_value=[])
    monkeypatch.setattr(cycle, 'calculate_product', command)
    return recorders, inputs, command, read, begin


def test_full_cycle_reads_once_and_publishes_each_product_before_next(monkeypatch):
    recorders, inputs, command, read, begin = setup_run(monkeypatch)
    seen = []
    def compute(name, *args):
        recorders[name].snapshot.assert_called_once_with(inputs)
        if seen:
            recorders[seen[-1]].finish.assert_called_once_with()
        seen.append(name)
        return []
    command.side_effect = compute
    cycle.generate('all', writer='writer')
    read.assert_called_once_with()
    assert seen == list(PRODUCTS)
    for name, recorder in recorders.items():
        recorder.snapshot.assert_called_once_with(inputs)
        recorder.finish.assert_called_once_with()
    assert [c.args[0] for c in begin.call_args_list] == list(PRODUCTS)


@pytest.mark.parametrize('failed_product', ['geomagnetic-activity', 'atmospheric-density'])
def test_model_failure_does_not_block_other_products(monkeypatch, failed_product):
    recorders, _, command, read, _ = setup_run(monkeypatch)
    failure = ValueError('model failed')
    def compute(name, *args):
        if name == failed_product:
            raise failure
        return []
    command.side_effect = compute
    with pytest.raises(cycle.GenerationFailed) as error:
        cycle.generate('all', writer='writer')
    assert error.value.failures == {failed_product: failure}
    assert f"{failed_product}: model failed" in str(error.value)
    assert command.call_count == len(PRODUCTS)
    read.assert_called_once_with()
    for name, recorder in recorders.items():
        if name == failed_product:
            recorder.finish.assert_called_once_with(error=failure)
        else:
            recorder.finish.assert_called_once_with()


def test_publication_failure_is_recorded_and_other_products_continue(monkeypatch):
    recorders, _, command, _, _ = setup_run(monkeypatch)
    failure = ValueError('checksum mismatch')
    recorders['dst'].finish.side_effect = [failure, None]
    with pytest.raises(cycle.GenerationFailed):
        cycle.generate('all', writer='writer')
    assert recorders['dst'].finish.call_args_list == [call(), call(error=failure)]
    assert command.call_count == len(PRODUCTS)


def test_snapshot_failure_prevents_that_product_calculation(monkeypatch):
    recorders, _, command, _, _ = setup_run(monkeypatch)
    failure = RuntimeError('database unavailable')
    recorders['dst'].snapshot.side_effect = failure
    with pytest.raises(cycle.GenerationFailed):
        cycle.generate('dst', writer='writer')
    recorders['dst'].finish.assert_called_once_with(error=failure)
    command.assert_not_called()


def test_lost_writer_stops_cycle_without_starting_more_products(monkeypatch):
    recorders, _, command, _, begin = setup_run(monkeypatch)
    first = next(iter(PRODUCTS))
    command.side_effect = RuntimeError('connection lost')
    recorders[first].finish.side_effect = RuntimeError('lock lost')
    with pytest.raises(RuntimeError, match='lock lost'):
        cycle.generate('all', writer='writer')
    assert begin.call_count == 1


def test_shared_input_failure_records_all_selected_products_and_reads_once(monkeypatch):
    recorders, _, command, read, _ = setup_run(monkeypatch)
    failure = RuntimeError('Clio unavailable')
    read.side_effect = failure
    with pytest.raises(cycle.GenerationFailed) as error:
        cycle.generate('all', writer='writer')
    assert set(error.value.failures) == set(PRODUCTS)
    read.assert_called_once_with()
    command.assert_not_called()
    for recorder in recorders.values():
        recorder.snapshot.assert_not_called()
        recorder.finish.assert_called_once_with(error=failure)


def test_retry_subset_does_not_create_successful_product_attempts(monkeypatch):
    recorders, inputs, command, read, begin = setup_run(monkeypatch)
    cycle.generate_products(('dst', 'atmospheric-density'), writer='writer')
    assert [c.args[0] for c in begin.call_args_list] == ['dst', 'atmospheric-density']
    assert all(c.args[1] == 'manual' for c in begin.call_args_list)
    read.assert_called_once_with()
    recorders['solar-wind-speed'].finish.assert_not_called()


def test_manual_input_preparation_uses_supervised_timeout(monkeypatch):
    recorders, _, command, read, begin = setup_run(monkeypatch)
    timeout = TimeoutError('input preparation exceeded budget')
    execute = Mock(side_effect=timeout)
    monkeypatch.setattr(cycle, 'execute', execute)
    with pytest.raises(cycle.GenerationFailed) as error:
        cycle.generate('all', writer='writer')
    execute.assert_called_once_with(read, timeout_seconds=540)
    assert set(error.value.failures.values()) == {timeout}
    assert all(call.kwargs['writer'] == 'writer' for call in begin.call_args_list)
    command.assert_not_called()
    for recorder in recorders.values():
        recorder.finish.assert_called_once_with(error=timeout)


def test_manual_shutdown_records_interruption_and_stops_dispatch(monkeypatch):
    from argus_prophet.scheduling.execution import ShutdownRequested
    recorders, _, command, _, begin = setup_run(monkeypatch)
    interruption = ShutdownRequested('shutdown deadline')
    monkeypatch.setattr(cycle, 'execute', Mock(side_effect=interruption))
    with pytest.raises(ShutdownRequested):
        cycle.generate('all', writer='writer')
    assert begin.call_count == 1
    recorders[next(iter(PRODUCTS))].finish.assert_called_once_with(error=interruption)
    command.assert_not_called()
