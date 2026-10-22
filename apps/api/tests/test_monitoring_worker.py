import asyncio
from unittest.mock import AsyncMock

import pytest

from api import worker


@pytest.mark.parametrize('fail', [False, True])
def test_worker_cleans_up_both_loops_and_database(monkeypatch, fail):
    dispose = AsyncMock()
    monkeypatch.setattr(worker, 'dispose_engine', dispose)

    async def check():
        started = [asyncio.Event(), asyncio.Event()]
        finished = []

        async def monitor(index):
            started[index].set()
            try:
                await started[1 - index].wait()
                if fail and index == 0:
                    raise RuntimeError('Monitor failed')
                await asyncio.Event().wait()
            finally:
                finished.append(index)

        monkeypatch.setattr(worker, 'run_monitor_loop', lambda: monitor(0))
        monkeypatch.setattr(worker, 'run_traffic_loop', lambda: monitor(1))
        task = asyncio.create_task(worker.run())
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
        if fail:
            with pytest.raises(ExceptionGroup):
                await asyncio.wait_for(task, 1)
        else:
            task.cancel()
            await asyncio.wait_for(task, 1)
        assert sorted(finished) == [0, 1]

    asyncio.run(check())
    dispose.assert_awaited_once()
