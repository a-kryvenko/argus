from fastapi import APIRouter, Response
from app.services.observations_client import read_observations

router = APIRouter(prefix='/public/observations', tags=['observations'])


@router.get('/summary')
async def observation_summary(response: Response):
    """Current values and observed trends with coverage; no forecast or impact score.

    Changes compare five-minute means one hour apart with at least 80% coverage
    in both windows and the last hour. Source changes and stale inputs suppress them.
    Southward Bz duration requires consecutive, unflagged samples from one spacecraft.
    """
    response.headers['Cache-Control'] = 'no-store'
    return await read_observations('summary')
