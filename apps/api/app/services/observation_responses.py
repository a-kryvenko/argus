"""Project Clio responses at the public boundary; never alter internal payloads."""
from app.schemas.observations import ObservationMetadata, ObservationSummary, SummaryMetadata
from app.schemas.response import success_response
from fastapi import HTTPException
from pydantic import ValidationError


def public_observations(response, model, *, meta: bool):
    # Preserve upstream validation responses and not-ready error envelopes.
    if not isinstance(response, dict) or not response.get('success') or response.get('data') is None:
        return response
    data = response['data']
    try:
        result = model.model_validate(data)
        if meta:
            result.meta = (SummaryMetadata.model_validate(data) if model is ObservationSummary
                           else ObservationMetadata.model_validate(data))
    except ValidationError:
        raise HTTPException(503, 'Observation service returned an invalid response') from None
    return success_response(result)
