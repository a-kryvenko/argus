from app.services.observations_client import read_observations
from fastapi import APIRouter, Query

router = APIRouter(prefix="/public/observations", tags=["observations"])


@router.get("/history")
async def observation_history(
    limit: int = Query(default=24, ge=1, le=168),
):
    """Return the most recent hourly observations, oldest first (up to 168)."""
    return await read_observations('history', {'limit': limit})


@router.get("/latest")
async def latest_observations():
    return await read_observations('latest')
