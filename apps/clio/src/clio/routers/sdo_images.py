"""JSON catalog of individual observations; image bytes stay on shared disk."""
from datetime import UTC, datetime, timedelta
import logging
from zipfile import BadZipFile

from fastapi import APIRouter, Depends, HTTPException
from pydantic import AwareDatetime

from clio.routers.forecast_inputs import require_service_token
from common.schemas.response import ApiResponse, success_response
from common.schemas.sdo_images import MissingSDOImage, SDOImageCatalog, SDOImageReference
from common.sdo_images import OBSERVED_CHANNELS, RETENTION_HOURS, archive_root, image_path, read_metadata, utc

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/internal/v1/observations', dependencies=[Depends(require_service_token)])


@router.get('/sdo-images', response_model=ApiResponse[SDOImageCatalog])
def sdo_images(start: AwareDatetime | None = None, end: AwareDatetime | None = None,
               as_of: AwareDatetime | None = None, channel: str | None = None):
    """List a bounded hourly range, without selecting model moments or filling gaps.

    Defaults to the 144 slots ending in the as_of hour. End is exclusive.
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
        raise HTTPException(422, 'Use whole UTC hours, start < end, at most 144 hours and no slots after as_of')
    if channel is not None and channel not in OBSERVED_CHANNELS:
        raise HTTPException(422, 'Unknown SDO observation channel')
    channels = (channel,) if channel is not None else OBSERVED_CHANNELS
    root = archive_root().resolve()
    items, missing = [], []
    for offset in range(int((end - start).total_seconds() // 3600)):
        slot = start + timedelta(hours=offset)
        for name in channels:
            try:
                metadata = read_metadata(root, slot, name)
            except FileNotFoundError:
                missing.append(MissingSDOImage(slot_at=slot, channel=name))
                continue
            except (OSError, ValueError, KeyError, TypeError, EOFError, BadZipFile):
                logger.exception('Cannot read SDO observation %s at %s', name, slot)
                raise HTTPException(503, 'SDO observation storage is not readable') from None
            if utc(metadata['available_at']) > as_of or utc(metadata['observed_at']) > as_of:
                missing.append(MissingSDOImage(slot_at=slot, channel=name))
                continue
            items.append(SDOImageReference(path=str(image_path(root, slot, name)), metadata=metadata))
    return success_response(SDOImageCatalog(start=start, end=end, as_of=as_of, read_at=now,
                                           items=items, missing=missing))
