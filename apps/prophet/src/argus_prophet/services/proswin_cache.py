"""Optional immutable native-horizon forecasts; failed reads leave DLinear usable."""
import json
import logging
import os
from pathlib import Path
from common.config import get_config
from common.schemas.forecast_inputs import ProswinPrediction
from datetime import timedelta

logger = logging.getLogger(__name__)


def cache_root():
    return Path(os.getenv('PROPHET_PROSWIN_CACHE',str(get_config().data_root/'prophet/proswin')))


def read_predictions(as_of, *, ready_at=None):
    issue=as_of.replace(minute=0, second=0, microsecond=0)
    ready_at = ready_at or as_of
    result=[]
    for lead in range(1, 97):
        valid = issue + timedelta(hours=lead)
        path=cache_root()/'predictions'/f'{valid:%Y%m%dT%H}.json'
        try:
            record=ProswinPrediction.model_validate(json.loads(path.read_text()))
            if (record.available_at<=ready_at and
                (record.available_at<=as_of or (record.source_cutoff is not None and record.source_cutoff<=as_of)) and record.valid_time==valid and
                record.image_slot+timedelta(hours=96)==valid): result.append(record)
        except FileNotFoundError:
            continue
        except (OSError,ValueError,TypeError):
            logger.warning('PROSWIN record unavailable: %s',path,exc_info=True)
    return result
