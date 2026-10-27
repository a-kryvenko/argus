"""JSON catalog of individual observations; image bytes stay on shared disk."""
from datetime import UTC, datetime, timedelta

from common.schemas.response import ApiResponse, success_response
from common.schemas.sdo_images import SDOImageCatalog
from common.sdo_images import OBSERVED_CHANNELS, RETENTION_HOURS, utc
from fastapi import APIRouter, HTTPException
from pydantic import AwareDatetime

from clio.observations.errors import ObservationUnavailable
from clio.observations.file_catalog import read_image_catalog

router = APIRouter(prefix='/internal/v1/observations')


@router.get('/sdo-images', response_model=ApiResponse[SDOImageCatalog])
def sdo_images(start: AwareDatetime | None = None, end: AwareDatetime | None = None,
               as_of: AwareDatetime | None = None, channel: str | None = None):
    """List a bounded hourly range, without selecting model moments or filling gaps.

    Defaults to 45 days of hourly slots ending in the as_of hour. End is exclusive.
    Paths are references at read time, not leases against subsequent retention.
    """
    now = datetime.now(UTC)
    as_of = utc(as_of) if as_of is not None else now
    if as_of > now:
        raise HTTPException(422, 'as_of must not be in the future')
    last_boundary = as_of.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    end = utc(end) if end is not None else last_boundary
    start = utc(start) if start is not None else end - timedelta(hours=RETENTION_HOURS)
    if (any(t.minute or t.second or t.microsecond for t in (start, end))
            or not timedelta(0) < end - start <= timedelta(hours=RETENTION_HOURS)
            or end > last_boundary):
        raise HTTPException(422, f'Use whole UTC hours, start < end, at most {RETENTION_HOURS} hours and no slots after as_of')
    if channel is not None and channel not in OBSERVED_CHANNELS:
        raise HTTPException(422, 'Unknown SDO observation channel')
    channels = (channel,) if channel is not None else OBSERVED_CHANNELS
    try:
        catalog = read_image_catalog(start, end, as_of, now, channels)
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    return success_response(catalog)
