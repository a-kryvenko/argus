"""HTTP access to stored observation originals."""
from datetime import timedelta

from common.schemas.observation_files import RawObservationFile
from common.schemas.response import success_response
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import AwareDatetime
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from clio.db.models.file import File
from clio.db.models.file_observation import FileObservation
from clio.db.session import get_request_db_session
from clio.routers.errors import ObservationUnavailable

router = APIRouter(prefix='/internal/v1')
MAX_FILES = 10_000


@router.get('/files')
async def list_files(
    response: Response,
    observation: str,
    start: AwareDatetime,
    end: AwareDatetime,
    session: AsyncSession = Depends(get_request_db_session),
):
    if not timedelta(0) < end - start <= timedelta(days=366):
        raise HTTPException(422, 'Invalid observation file interval')
    response.headers['Cache-Control'] = 'no-store'
    try:
        files = await load_observation_files(session, observation=observation, start=start, end=end)
    except ObservationUnavailable as exc:
        raise HTTPException(503, str(exc)) from None
    return success_response(files)


async def load_observation_files(session: AsyncSession, *, observation, start, end):
    statement = (select(FileObservation.observation, FileObservation.observed_at, FileObservation.sha256, FileObservation.source_product,
        File.id.label('file_id'), File.timestamp.label('file_timestamp'), File.bucket, File.key)
        .join(File, File.id == FileObservation.file_id)
        .where(FileObservation.observation == observation,
               (FileObservation.source_id.is_(None) | (FileObservation.source_id == '')),
               FileObservation.observed_at >= start, FileObservation.observed_at < end)
        .order_by(FileObservation.observed_at, FileObservation.source_product, File.key).limit(MAX_FILES + 1))

    try:
        rows = (await session.execute(statement)).mappings().all()
        if len(rows) > MAX_FILES:
            raise ObservationUnavailable('Observation file range exceeds the supported batch size')

        return [RawObservationFile(
            observation=row['observation'],
            observed_at=row['observed_at'],
            sha256=row['sha256'],
            source_product=row['source_product'],
            file=dict(
                id=row['file_id'],
                timestamp=row['file_timestamp'],
                bucket=row['bucket'],
                key=row['key']
                )) for row in rows]
    except (SQLAlchemyError, ValueError, KeyError, TypeError) as exc:
        raise ObservationUnavailable('Observation metadata is unavailable') from exc
