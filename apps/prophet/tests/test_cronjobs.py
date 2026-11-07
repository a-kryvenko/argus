"""Deployment cron entries invoke the supported one-shot CLI commands."""
from pathlib import Path
import shlex
from unittest.mock import AsyncMock

from prophet import cli
from prophet.commands import summary, cleanup, generation, verification
from prophet.services.generation.products import PRODUCTS


def test_cron_jobs_cover_products_verification_and_detail_retention(monkeypatch):
    generate, verify, clean, draw = (AsyncMock(return_value=None) for _ in range(4))
    monkeypatch.setattr(generation, 'run', generate)
    monkeypatch.setattr(verification, 'run', verify)
    monkeypatch.setattr(cleanup, 'run', clean)
    monkeypatch.setattr(summary, 'run', draw)
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
        elif entry[8] == 'summary':
            assert entry[:5] == ['*/5', '*', '*', '*', '*']
            assert entry[8:] == ['summary']
        elif entry[8] == 'verification':
            assert entry[:5] == ['30', '*/6', '*', '*', '*']
            assert entry[8:] == ['verification', 'all']
        else:
            assert entry[:5] == ['45', '*', '*', '*', '*']
            assert entry[8:] == ['cleanup', '--apply']
    assert set(products) == set(PRODUCTS)
    assert len(entries) == len(PRODUCTS)+3
    assert products['solar-wind-events'] == ['*/5', '*', '*', '*', '*']
    assert all(schedule == ['10', '*', '*', '*', '*'] for product, schedule in products.items()
               if product != 'solar-wind-events')
    assert generate.await_count == len(PRODUCTS)
    draw.assert_awaited_once()
    verify.assert_awaited_once()
    clean.assert_awaited_once()
    assert clean.await_args.args[0].apply
