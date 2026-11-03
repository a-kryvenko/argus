"""Display-only SDO images; scientific FITS have a separate contract."""
from fastapi import APIRouter, HTTPException
from common.schemas.response import success_response
from clio.observations import sdo_browse
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy.exc import SQLAlchemyError

router = APIRouter(prefix='/internal/v1/observations/sdo')


@router.get('')
def catalog():
    try:
        return success_response(sdo_browse.catalog())
    except (OSError, ValueError, SQLAlchemyError, BotoCoreError, ClientError):
        raise HTTPException(503, 'SDO image storage is unavailable') from None
