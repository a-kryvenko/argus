"""Publish the first available Kp/ap forecast for a detected L1 front."""
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import os

import httpx
from sqlalchemy import text

from common.schemas.forecast_release import FAST_L1_ARTIFACT, FAST_L1_COLUMNS, FAST_L1_PRODUCT
from prophet.db.locks import GENERATION_LOCK, GenerationBusy, transaction, writer_lock
from prophet.services.generation.fast_l1_model import FastL1Forecast, FastL1GeomagneticForecaster, find_trigger
from prophet.services.generation.models import load_model
from prophet.services.inputs.l1_geomagnetic import read_l1_geomagnetic_indices, read_l1_wind_history
from prophet.services.releases.recording import RunRecorder


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingUpdate:
    issue_time: datetime
    forecast: FastL1Forecast
    inputs: dict
    model_info: dict


def input_rows(frame):
    return json.loads(frame.reset_index().to_json(
        orient='records', date_format='iso', date_unit='us', double_precision=15,
    ))


async def published_trigger(writer):
    async with transaction(writer) as session:
        value = await session.scalar(text("""SELECT a.model_info->>'trigger_time' FROM forecast_release r
            JOIN forecast_artifact a ON a.run_id = r.run_id AND a.name = :artifact
            WHERE r.product = :product
            ORDER BY r.issue_time DESC, r.published_at DESC, r.id DESC LIMIT 1"""),
            {'product': FAST_L1_PRODUCT, 'artifact': FAST_L1_ARTIFACT})
    return datetime.fromisoformat(value) if value else None


class FastL1Update:
    def __init__(self):
        self.model = None
        self.model_info = None
        self.last_trigger = None
        self.pending = None

    def calculate(self, runtime, issue_time):
        url, token = os.getenv('OBSERVATIONS_URL'), os.getenv('OBSERVATIONS_SERVICE_TOKEN')
        if not url or not token:
            raise RuntimeError('OBSERVATIONS_URL and OBSERVATIONS_SERVICE_TOKEN are required')
        with httpx.Client(timeout=httpx.Timeout(10, connect=3), follow_redirects=False,
                          trust_env=False) as client:
            wind = read_l1_wind_history(client, url, token, issue_time)
            trigger = find_trigger(wind, issue_time=issue_time)
            if trigger is None or (self.last_trigger is not None and trigger.trigger_time <= self.last_trigger):
                return None
            kp, dst = read_l1_geomagnetic_indices(client, url, token, issue_time)
        if self.model is None:
            self.model, self.model_info = load_model(
                FastL1GeomagneticForecaster, workdir=runtime.workdir, registry=runtime.models_registry['models'],
            )
        forecast = self.model.forecast_frames(wind, kp, dst, issue_time=issue_time)
        if forecast is None:
            return None
        inputs = {'as_of': issue_time.isoformat(), 'wind': input_rows(wind),
                  'kp': input_rows(kp.rename('kp')), 'dst': input_rows(dst.rename('dst'))}
        metadata = dict(self.model_info, trigger_time=forecast.trigger_time.isoformat(),
                        estimated_arrival=forecast.estimated_arrival.isoformat(),
                        input_cutoff=issue_time.isoformat())
        return PendingUpdate(issue_time, forecast, inputs, metadata)

    async def publish(self, runtime, update):
        trigger_time = update.forecast.trigger_time
        async with writer_lock(GENERATION_LOCK) as writer:
            previous = await published_trigger(writer)
            if previous is not None and previous >= trigger_time:
                self.last_trigger = previous
                return
            joined = update.forecast.kp.merge(update.forecast.ap, on='valid_time', validate='one_to_one')
            if len(joined) != len(update.forecast.kp) or len(joined) != len(update.forecast.ap):
                raise ValueError('Fast Kp and ap must share the same targets')
            future = joined.loc[joined.valid_time.gt(datetime.now(UTC))]
            if future.empty:
                self.last_trigger = trigger_time
                return
            frame = future.assign(issue_time=update.issue_time)[['issue_time', 'valid_time', *FAST_L1_COLUMNS]]
            run = await RunRecorder.begin(FAST_L1_PRODUCT, 'scheduled', runtime, writer=writer)
            try:
                await run.snapshot_payload(update.inputs, {'trigger_time': trigger_time.isoformat(),
                                                           'input_cutoff': update.issue_time.isoformat()})
                await run.store(FAST_L1_ARTIFACT, frame.to_csv(index=False).encode('utf-8'),
                                update.model_info, len(frame), list(frame))
                publication = await run.finish()
            except BaseException as exc:
                await run.finish(error=exc)
                raise
            self.last_trigger = trigger_time
            if publication.status == 'published':
                logger.info('Prophet published fast L1 Kp/ap release %s', publication.release_id)

    async def poll(self):
        from common.config import get_config
        try:
            runtime = get_config()
            if FastL1GeomagneticForecaster.registry_name not in runtime.models_registry['models']:
                return
            if self.pending is None:
                self.pending = await asyncio.to_thread(self.calculate, runtime, datetime.now(UTC))
            if self.pending is not None:
                await self.publish(runtime, self.pending)
                self.pending = None
        except GenerationBusy:
            # The next minute can still publish the same first-front measurements.
            return
        except Exception:
            self.pending = None
            logger.exception('Fast L1 Kp/ap update unavailable')
