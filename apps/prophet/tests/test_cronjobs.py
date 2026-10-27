"""Deployment cron entries invoke the supported one-shot CLI commands."""
from pathlib import Path
import shlex
from unittest.mock import AsyncMock

from prophet import cli
from prophet.commands import generation, verification
from prophet.services.generation.products import PRODUCTS


def test_cron_jobs_cover_products_and_verification_with_the_expected_cadence(monkeypatch):
    generate, verify = AsyncMock(return_value=None), AsyncMock(return_value=None)
    monkeypatch.setattr(generation, 'run', generate)
    monkeypatch.setattr(verification, 'run', verify)
    from common import runtime
    monkeypatch.setattr(runtime, 'run_command', lambda run: run())
    entries = [shlex.split(line) for line in (Path(__file__).parents[1] / 'cronjobs.txt').read_text().splitlines()
               if line.strip() and not line.startswith('#')]
    products = {}
    for entry in entries:
        assert entry[5:7] == ['@ROOT@/scripts/cron-run', 'prophet']
        cli.main(entry[8:])
        if entry[8] == 'generation':
            products[entry[9]] = entry[:5]
        else:
            assert entry[:5] == ['30', '*/6', '*', '*', '*']
            assert entry[8:] == ['verification', 'all']
    assert set(products) == set(PRODUCTS)
    assert len(entries) == len(PRODUCTS)+1
    assert products['solar-wind-events'] == ['*/5', '*', '*', '*', '*']
    assert all(schedule == ['10', '*', '*', '*', '*'] for product, schedule in products.items()
               if product != 'solar-wind-events')
    assert generate.await_count == len(PRODUCTS)
    verify.assert_awaited_once()
