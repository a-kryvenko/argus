from fastapi import APIRouter, Response
from app.services.observations_client import read_observations

router = APIRouter(prefix='/public/observations', tags=['observations'])


@router.get('/status')
async def collection_status(response: Response):
    """Source diagnostics: attempts, successful responses, storage outcomes and data age.

    Status priority: stalled/overdue collection, errors, then delayed/missing data.
    No recent attempts indicates missing progress; it does not inspect processes.
    A historical last error is retained after recovery; consecutive_failures resets.
    """
    response.headers['Cache-Control'] = 'no-store'
    return await read_observations('status')
