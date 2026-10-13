"""Optional HSS enrichment: consume existing PROSWIN cache, never start inference."""
import logging

import httpx

logger = logging.getLogger(__name__)


def enrich_hss_inputs(inputs, client, url, token):
    from argus_prophet.services.proswin_cache import read_predictions
    if not inputs.proswin_predictions:
        try:
            inputs.proswin_predictions = read_predictions(inputs.as_of)
        except (OSError, ValueError, RuntimeError):
            logger.exception('HSS PROSWIN cache unavailable')
            inputs.hss_input_status['proswin'] = {'state': 'unavailable'}
    try:
        from argus_prophet.services.coronal_holes import prepare
        prepare(inputs, client, url, token)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, ImportError, httpx.HTTPError) as exc:
        logger.warning('HSS image features unavailable: %s', type(exc).__name__)
        inputs.hss_input_status['coronal_holes'] = {'state': 'unavailable', 'reason': type(exc).__name__}
    return inputs
