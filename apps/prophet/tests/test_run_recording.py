from unittest.mock import AsyncMock, Mock, call
from types import SimpleNamespace

import pytest

from common.schemas.forecast_status import ForecastPublication
from prophet.services.generation import cycle
from prophet.services.generation.products import PRODUCTS


def setup_run():
    from prophet.config import ProphetConfig
    from prophet.services.generation.service import ForecastGenerationService
    recorders = {name: Mock(run_id=name, snapshot=AsyncMock(), store=AsyncMock(), finish=AsyncMock()) for name in PRODUCTS}
    runtime = SimpleNamespace(workdir='.', models_registry={'models': {}})
    policy = ProphetConfig()
    begin = AsyncMock(side_effect=lambda name: cycle.ProductRun(name, recorders[name], runtime, policy))
    inputs = SimpleNamespace(observations=SimpleNamespace(points=[]))
    read = AsyncMock(return_value=inputs)
    command = Mock(return_value=[])
    service = ForecastGenerationService(
        input_loader=read, calculator=command,
        executor=lambda target, *args, **kw: target(*args),
        run_factory=begin, before_dispatch=lambda: None,
        timeout_seconds=policy.calculation_timeout_seconds,
    )
    return recorders, inputs, command, read, begin, service


async def test_full_cycle_shares_cutoff_and_cache_and_publishes_each_product_before_next():
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
    (await service.generate(tuple(PRODUCTS)))
    assert [call.args[0] for call in read.await_args_list] == list(PRODUCTS)
    assert len({call.args[1] for call in read.await_args_list}) == 1
    assert len({id(call.args[2]) for call in read.await_args_list}) == 1
    assert seen == list(PRODUCTS)
    for name, recorder in recorders.items():
        recorder.snapshot.assert_called_once_with(inputs)
        recorder.finish.assert_called_once_with()
    assert [c.args[0] for c in begin.call_args_list] == list(PRODUCTS)


@pytest.mark.parametrize('failed_product', ['geomagnetic-activity', 'atmospheric-density'])
async def test_model_failure_does_not_block_other_products(failed_product):
    recorders, _, command, read, _, service = setup_run()
    failure = ValueError('model failed')
    def compute(request):
        name = request.product
        if name == failed_product:
            raise failure
        return []
    command.side_effect = compute
    with pytest.raises(cycle.GenerationFailed) as error:
        (await service.generate(tuple(PRODUCTS)))
    assert error.value.failures == {failed_product: failure}
    assert f"{failed_product}: model failed" in str(error.value)
    assert command.call_count == len(PRODUCTS)
    assert read.await_count == len(PRODUCTS)
    for name, recorder in recorders.items():
        if name == failed_product:
            recorder.finish.assert_called_once_with(error=failure)
        else:
            recorder.finish.assert_called_once_with()


async def test_publication_failure_is_recorded_and_other_products_continue():
    recorders, _, command, _, _, service = setup_run()
    failure = ValueError('checksum mismatch')
    recorders['dst'].finish.side_effect = [failure, None]
    with pytest.raises(cycle.GenerationFailed):
        (await service.generate(tuple(PRODUCTS)))
    assert recorders['dst'].finish.call_args_list == [call(), call(error=failure)]
    assert command.call_count == len(PRODUCTS)


async def test_snapshot_failure_prevents_that_product_calculation():
    recorders, _, command, _, _, service = setup_run()
    failure = RuntimeError('database unavailable')
    recorders['dst'].snapshot.side_effect = failure
    with pytest.raises(cycle.GenerationFailed):
        (await service.generate(('dst',)))
    recorders['dst'].finish.assert_called_once_with(error=failure)
    command.assert_not_called()


async def test_lost_writer_stops_cycle_without_starting_more_products():
    recorders, _, command, _, begin, service = setup_run()
    first = next(iter(PRODUCTS))
    command.side_effect = RuntimeError('connection lost')
    recorders[first].finish.side_effect = RuntimeError('lock lost')
    with pytest.raises(RuntimeError, match='lock lost'):
        (await service.generate(tuple(PRODUCTS)))
    assert begin.call_count == 1


async def test_product_input_failure_is_recorded_and_other_products_continue():
    recorders, _, command, read, _, service = setup_run()
    failure = RuntimeError('hourly input unavailable')
    read.side_effect = [failure, read.return_value]
    with pytest.raises(cycle.GenerationFailed) as error:
        await service.generate(('dst', 'hmf'))
    assert error.value.failures == {'dst': failure}
    assert command.call_args.args[0].product == 'hmf'
    recorders['dst'].snapshot.assert_not_called()
    recorders['dst'].finish.assert_awaited_once_with(error=failure)
    recorders['hmf'].finish.assert_awaited_once_with()


async def test_retry_subset_does_not_create_successful_product_attempts():
    recorders, inputs, command, read, begin, service = setup_run()
    (await service.generate(('dst', 'atmospheric-density')))
    assert [c.args[0] for c in begin.call_args_list] == ['dst', 'atmospheric-density']
    assert read.await_count == 2
    recorders['solar-wind-speed'].finish.assert_not_called()


async def test_timed_out_optional_input_uses_fallback(monkeypatch):
    from datetime import UTC, datetime
    from forecast.inputs import ForecastInputs
    from common.schemas.observation import Observation
    from prophet.services.inputs.loader import load_product_inputs, read_source

    now = datetime.now(UTC)
    inputs = ForecastInputs(as_of=now, read_at=now, observations=Observation(points=[]))
    timeout = TimeoutError('input preparation exceeded budget')
    execute = Mock(side_effect=[inputs, timeout, []])
    result = await load_product_inputs('solar-wind-speed', now, {}, executor=execute, timeout_seconds=540)
    assert result.solar_wind_events is None
    assert execute.call_args_list == [
        call(read_source, source, now, timeout_seconds=540) for source in ('hourly', 'events', 'proswin_cache')
    ]


async def test_manual_shutdown_records_interruption_and_stops_dispatch():
    from prophet.execution import ShutdownRequested
    recorders, _, command, read, begin, service = setup_run()
    interruption = ShutdownRequested('shutdown deadline')
    read.side_effect = interruption
    with pytest.raises(ShutdownRequested):
        (await service.generate(tuple(PRODUCTS)))
    assert begin.call_count == 1
    recorders[next(iter(PRODUCTS))].finish.assert_called_once_with(error=interruption)
    command.assert_not_called()


@pytest.mark.parametrize('products', [(), ('dst', 'dst'), ('unsupported',)])
async def test_invalid_selection_has_no_side_effects(products):
    _, _, command, read, begin, service = setup_run()
    with pytest.raises(ValueError, match='distinct supported'):
        (await service.generate(products))
    begin.assert_not_called()
    read.assert_not_called()
    command.assert_not_called()


async def test_reusing_service_loads_a_new_snapshot_per_cycle():
    _, inputs, command, read, _, service = setup_run()
    next_inputs = SimpleNamespace(observations=SimpleNamespace(points=[]))
    read.side_effect = [inputs, next_inputs]
    (await service.generate(('dst',)))
    (await service.generate(('dst',)))
    assert read.call_count == 2
    assert read.await_args_list[0].args[2] is not read.await_args_list[1].args[2]
    assert command.call_count == 2
    request = command.call_args.args[0]
    assert request.product == 'dst'
    assert command.call_args_list[0].args[0].inputs is inputs
    assert request.inputs is next_inputs


async def test_shutdown_after_loading_inputs_does_not_record_snapshot_or_calculate():
    from prophet.execution import ShutdownRequested
    recorders, _, command, read, begin, service = setup_run()
    interruption = ShutdownRequested('stopped after inputs')
    service.before_dispatch = Mock(side_effect=[None, interruption])
    with pytest.raises(ShutdownRequested):
        (await service.generate(('dst',)))
    read.assert_called_once()
    recorders['dst'].snapshot.assert_not_called()
    recorders['dst'].finish.assert_called_once_with(error=interruption)
    command.assert_not_called()


@pytest.mark.parametrize('scheduled_job,trigger', [(None, 'manual'), ('generation:dst', 'scheduled')])
async def test_command_composition_preserves_writer_and_trigger(monkeypatch, scheduled_job, trigger):
    if scheduled_job:
        monkeypatch.setenv('ARGUS_SCHEDULED_JOB', scheduled_job)
    else:
        monkeypatch.delenv('ARGUS_SCHEDULED_JOB', raising=False)
    from common import config
    from prophet.services.inputs import loader as observations
    from prophet.services.releases import recording as ledger
    runtime = SimpleNamespace(workdir='.', models_registry={'models': {}})
    monkeypatch.setattr(config, 'get_config', lambda: runtime)
    monkeypatch.setattr(cycle, 'provenance', lambda _: {'source': 'test'})
    monkeypatch.setattr(cycle, 'execute', lambda target, *args, **kw: target(*args))
    inputs = SimpleNamespace(observations=SimpleNamespace(points=[]))
    monkeypatch.setattr(observations, 'load_product_inputs', AsyncMock(return_value=inputs))
    calculate = Mock(return_value=[])
    monkeypatch.setattr(cycle, 'calculate_product', calculate)
    recorder = Mock(snapshot=AsyncMock(), store=AsyncMock(), finish=AsyncMock())
    begin = AsyncMock(return_value=recorder)
    monkeypatch.setattr(ledger.RunRecorder, 'begin', begin)
    writer = object()
    (await cycle.generate('dst', writer=writer))
    assert begin.call_args.args[:3] == ('dst', trigger, runtime)
    assert begin.call_args.kwargs['writer'] is writer
    assert begin.call_args.kwargs['details'] == {'source': 'test'}
    recorder.snapshot.assert_called_once_with(inputs)
    recorder.finish.assert_called_once_with()
    assert calculate.call_args.args[0].product == 'dst'


async def test_calculation_uses_configured_timeout_and_only_serializable_request():
    from prophet.services.generation.calculation import CalculationRequest
    _, inputs, command, read, _, service = setup_run()
    executor = Mock(side_effect=lambda target, *args, **kw: target(*args))
    service.executor = executor
    service.timeout_seconds = 17
    (await service.generate(('dst',)))
    executor.assert_called_once()
    target, request = executor.call_args.args
    assert target is command
    assert isinstance(request, CalculationRequest)
    assert request.inputs is inputs
    assert executor.call_args.kwargs == {'timeout_seconds': 17}


async def test_kept_release_is_logged_without_claiming_publication(caplog):
    from logging import INFO

    recorders, _, _, _, _, service = setup_run()
    recorders['dst'].finish.return_value = ForecastPublication(status='kept_current', reason='newer_issue_time')
    caplog.set_level(INFO)
    await service.generate(('dst',))
    assert 'kept the current release (dst): newer_issue_time' in caplog.text


pytestmark = pytest.mark.anyio
