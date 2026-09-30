from fastapi import APIRouter, HTTPException
from common.schemas.leo_drag import LeoDragRequest
from app.schemas.leo_drag import LeoDragAssessment
from app.schemas.metadata import MetaQuery
from app.schemas.response import ApiResponse, success_response
from app.services.forecast_errors import ArtifactNotReadyError
from app.services.intelligence_client import assess_drag, DragDomainError

router = APIRouter(tags=['risks'])


@router.post('/public/risks/leo-drag', response_model=ApiResponse[LeoDragAssessment],
             summary='Estimate drag for a circular LEO orbit',
             description=('First-order drag assessment for 200–800 km using fresh JB2008 density. '
                          '24/48 hours start at the forecast issue time. Drivers persist at observed values. '
                          'Optional cumulative altitude-loss thresholds determine risk categories; no probabilities. '
                          'Returns 422 outside the grid or above 1000 m estimated decay, and 503 when unavailable.'),
             responses={503: {'model': ApiResponse[None], 'description': 'Impact service or density is not ready'}})
def leo_drag(request: LeoDragRequest, meta: MetaQuery = False):
    try:
        return success_response(LeoDragAssessment.from_internal(assess_drag(request), meta=meta))
    except ArtifactNotReadyError:
        raise HTTPException(503, 'Impact service or atmospheric density is not ready') from None
    except DragDomainError as exc:
        raise HTTPException(422, str(exc)) from None
