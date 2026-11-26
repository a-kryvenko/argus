"""Publish the first available fast forecasts for a detected L1 front."""
import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import logging
import os

import httpx
import pandas as pd
from sqlalchemy import text

from common.schemas.forecast_release import FAST_BT_PRODUCT, FAST_L1_PRODUCT, FAST_L1_RELEASES, PREDICTION_COLUMNS
from prophet.db.locks import GENERATION_LOCK, GenerationBusy, transaction, writer_lock
from prophet.services.generation.fast_l1_bt import FastL1BtForecaster
from prophet.services.generation.fast_l1_model import FastL1GeomagneticForecaster, find_trigger
from prophet.services.generation.models import load_model
from prophet.services.inputs.l1_geomagnetic import read_l1_geomagnetic_indices, read_l1_wind_history
from prophet.services.releases.recording import RunRecorder


logger = logging.getLogger(__name__)
MODELS = {FAST_BT_PRODUCT: FastL1BtForecaster, FAST_L1_PRODUCT: FastL1GeomagneticForecaster}


@dataclass(frozen=True)
class PendingUpdate:
    product: str
    issue_time: datetime
    trigger_time: datetime
    frame: pd.DataFrame
    inputs: dict
    model_info: dict


def input_rows(frame):
    return json.loads(frame.reset_index().to_json(
        orient='records', date_format='iso', date_unit='us', double_precision=15,
    ))


async def published_trigger(writer, product):
    async with transaction(writer) as session:
        value = await session.scalar(text("""SELECT a.model_info->>'trigger_time' FROM forecast_release r
            JOIN forecast_artifact a ON a.run_id = r.run_id AND a.name = :artifact
            WHERE r.product = :product
            ORDER BY r.issue_time DESC, r.published_at DESC, r.id DESC LIMIT 1"""),
            {'product': product, 'artifact': FAST_L1_RELEASES[product][0]})
    return datetime.fromisoformat(value) if value else None


class FastL1Update:
    def __init__(self):
        self.models = {}
        self.last_trigger = {}
        self.pending = []

    def calculate(self, runtime, issue_time):
        url, token = os.getenv('OBSERVATIONS_URL'), os.getenv('OBSERVATIONS_SERVICE_TOKEN')
        if not url or not token:
            raise RuntimeError('OBSERVATIONS_URL and OBSERVATIONS_SERVICE_TOKEN are required')
        with httpx.Client(timeout=httpx.Timeout(10, connect=3), follow_redirects=False,
                          trust_env=False) as client:
            wind = read_l1_wind_history(client, url, token, issue_time)
            trigger = find_trigger(wind, issue_time=issue_time)
            if trigger is None:
                return []
            updates = []
            registry = runtime.models_registry['models']
            for product, service in MODELS.items():
                previous = self.last_trigger.get(product)
                if service.registry_name not in registry or previous is not None and previous >= trigger.trigger_time:
                    continue
                try:
                    if product not in self.models:
                        self.models[product] = load_model(service, workdir=runtime.workdir, registry=registry)
                    model, model_info = self.models[product]
                    inputs = {'as_of': issue_time.isoformat(), 'wind': input_rows(wind)}
                    if product == FAST_BT_PRODUCT:
                        frame = model.forecast_frame(wind, trigger=trigger, issue_time=issue_time)
                    else:
                        kp, dst = read_l1_geomagnetic_indices(client, url, token, issue_time)
                        forecast = model.forecast_frames(wind, kp, dst, issue_time=issue_time)
                        if forecast is None:
                            continue
                        frame = forecast.kp.merge(forecast.ap, on='valid_time', validate='one_to_one')
                        if len(frame) != len(forecast.kp) or len(frame) != len(forecast.ap):
                            raise ValueError('Fast Kp and ap must share the same targets')
                        inputs.update(kp=input_rows(kp.rename('kp')), dst=input_rows(dst.rename('dst')))
                    if frame is None or frame.empty:
                        continue
                    metadata = dict(model_info, trigger_time=trigger.trigger_time.isoformat(),
                                    estimated_arrival=trigger.estimated_arrival.isoformat(),
                                    input_cutoff=issue_time.isoformat())
                    updates.append(PendingUpdate(product, issue_time, trigger.trigger_time, frame, inputs, metadata))
                except Exception:
                    logger.exception('Fast L1 calculation unavailable for %s', product)
            return updates

    async def publish(self, runtime, update):
        product, trigger_time = update.product, update.trigger_time
        artifact = FAST_L1_RELEASES[product][0]
        async with writer_lock(GENERATION_LOCK) as writer:
            previous = await published_trigger(writer, product)
            if previous is not None and previous >= trigger_time:
                self.last_trigger[product] = previous
                return
            future = update.frame.loc[update.frame.valid_time.gt(datetime.now(UTC))]
            if future.empty:
                self.last_trigger[product] = trigger_time
                return
            frame = future.assign(issue_time=update.issue_time)[['issue_time', 'valid_time', *PREDICTION_COLUMNS[artifact]]]
            run = await RunRecorder.begin(product, 'scheduled', runtime, writer=writer)
            try:
                await run.snapshot_payload(update.inputs, {'trigger_time': trigger_time.isoformat(),
                                                           'input_cutoff': update.issue_time.isoformat()})
                await run.store(artifact, frame.to_csv(index=False).encode('utf-8'),
                                update.model_info, len(frame), list(frame))
                publication = await run.finish()
            except BaseException as exc:
                await run.finish(error=exc)
                raise
            self.last_trigger[product] = trigger_time
            if publication.status == 'published':
                logger.info('Prophet published fast L1 %s release %s', product, publication.release_id)

    async def poll(self):
        from common.config import get_config
        try:
            runtime = get_config()
            if not any(model.registry_name in runtime.models_registry['models'] for model in MODELS.values()):
                return
            if not self.pending:
                self.pending = await asyncio.to_thread(self.calculate, runtime, datetime.now(UTC))
        except Exception:
            logger.exception('Fast L1 observations unavailable')
            return
        while self.pending:
            update = self.pending[0]
            try:
                await self.publish(runtime, update)
            except GenerationBusy:
                # Retry the original inputs on the next minute.
                return
            except Exception:
                logger.exception('Fast L1 publication unavailable for %s', update.product)
            self.pending.pop(0)
