"""Public API adapter for the internal Intelligence HTTP contract."""
import os

import httpx
from common.schemas.leo_drag import LeoDragAssessment, LeoDragRequest
from api.services.errors import ArtifactNotReadyError

class DragDomainError(Exception):
    pass


def assess_drag(request: LeoDragRequest, *, transport=None) -> LeoDragAssessment:
    url, token = os.getenv('INTELLIGENCE_URL'), os.getenv('INTELLIGENCE_SERVICE_TOKEN')
    if not url or not token:
        raise ArtifactNotReadyError('Impact service is not configured')
    try:
        with httpx.Client(timeout=httpx.Timeout(90, connect=10), follow_redirects=False,
                          trust_env=False, transport=transport) as client:
            response = client.post(url.rstrip('/') + '/internal/v1/risks/leo-drag',
                                   headers={'Authorization': f'Bearer {token}'},
                                   json=request.model_dump(mode='json'))
            if response.status_code == 422:
                raise DragDomainError('Orbit is outside the available grid or the fixed-orbit approximation (maximum decay 1000 m)')
            response.raise_for_status()
        result = LeoDragAssessment.model_validate_json(response.content)
        if result.inputs != request:
            raise ValueError('Mismatched impact inputs')
        return result
    except (httpx.HTTPError, ValueError):
        raise ArtifactNotReadyError('Impact service is unavailable') from None
