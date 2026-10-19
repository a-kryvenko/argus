from fastapi import APIRouter, HTTPException
from common.schemas.leo_drag import LeoDragRequest
from api.schemas.leo_drag import LeoDragAssessment
from api.schemas.metadata import MetaQuery
from common.schemas.response import ApiResponse, success_response
from api.services.errors import ArtifactNotReadyError
from api.services.intelligence.drag import assess_drag, DragDomainError

router = APIRouter(tags=['risks'])


@router.post('/risks/leo-drag', response_model=ApiResponse[LeoDragAssessment],
             summary='Estimate drag for a circular LEO orbit',
             description=('First-order drag assessment for 200–800 km using a fresh thermospheric density grid. '
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
