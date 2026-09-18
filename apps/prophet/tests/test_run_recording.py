from unittest.mock import Mock, call

import pytest

from argus_prophet import cli, generation, ledger, observations
from argus_prophet.products import PRODUCTS


def setup_run(monkeypatch):
    recorders = {name: Mock(run_id=name) for name in PRODUCTS}
    begin = Mock(side_effect=lambda name, *args, **kwargs: recorders[name])
    monkeypatch.setattr(ledger.RunRecorder, 'begin', begin)
    monkeypatch.setattr(ledger, 'provenance', lambda _: {})
    inputs = object()
    read = Mock(return_value=inputs)
    monkeypatch.setattr(observations, 'load_inputs', read)
    command = Mock()
    monkeypatch.setattr(generation, 'calculate', command)
    return recorders, inputs, command, read, begin


def test_full_cycle_reads_once_and_publishes_each_product_before_next(monkeypatch):
    recorders, inputs, command, read, begin = setup_run(monkeypatch)
    seen = []
    def compute(name, **kwargs):
        recorders[name].snapshot.assert_called_once_with(inputs)
        if seen:
            recorders[seen[-1]].finish.assert_called_once_with()
        seen.append(name)
    command.side_effect = compute
    cli.generate('all')
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
    def compute(name, **kwargs):
        if name == failed_product:
            raise failure
    command.side_effect = compute
    with pytest.raises(cli.GenerationFailed) as error:
        cli.generate('all')
    assert error.value.failures == {failed_product: failure}
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
    with pytest.raises(cli.GenerationFailed):
        cli.generate('all')
    assert recorders['dst'].finish.call_args_list == [call(), call(error=failure)]
    assert command.call_count == len(PRODUCTS)


def test_snapshot_failure_prevents_that_product_calculation(monkeypatch):
    recorders, _, command, _, _ = setup_run(monkeypatch)
    failure = RuntimeError('database unavailable')
    recorders['dst'].snapshot.side_effect = failure
    with pytest.raises(cli.GenerationFailed):
        cli.generate('dst')
    recorders['dst'].finish.assert_called_once_with(error=failure)
    command.assert_not_called()


def test_lost_writer_stops_cycle_without_starting_more_products(monkeypatch):
    recorders, _, command, _, begin = setup_run(monkeypatch)
    first = next(iter(PRODUCTS))
    command.side_effect = RuntimeError('connection lost')
    recorders[first].finish.side_effect = RuntimeError('lock lost')
    with pytest.raises(RuntimeError, match='lock lost'):
        cli.generate('all')
    assert begin.call_count == 1


def test_shared_input_failure_records_all_selected_products_and_reads_once(monkeypatch):
    recorders, _, command, read, _ = setup_run(monkeypatch)
    failure = RuntimeError('Clio unavailable')
    read.side_effect = failure
    with pytest.raises(cli.GenerationFailed) as error:
        cli.generate('all')
    assert set(error.value.failures) == set(PRODUCTS)
    read.assert_called_once_with()
    command.assert_not_called()
    for recorder in recorders.values():
        recorder.snapshot.assert_not_called()
        recorder.finish.assert_called_once_with(error=failure)


def test_retry_subset_does_not_create_successful_product_attempts(monkeypatch):
    recorders, inputs, command, read, begin = setup_run(monkeypatch)
    cli.generate_products(('dst', 'atmospheric-density'), 'scheduled', scheduled_slot='slot')
    assert [c.args[0] for c in begin.call_args_list] == ['dst', 'atmospheric-density']
    assert all(c.kwargs['scheduled_slot'] == 'slot' for c in begin.call_args_list)
    read.assert_called_once_with()
    recorders['solar-wind-speed'].finish.assert_not_called()
