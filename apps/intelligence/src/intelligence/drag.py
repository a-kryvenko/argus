"""Adapt a validated Prophet release to the private impact backend."""
from datetime import UTC, datetime
import io

import pandas as pd
from common.config import get_config
from common.density_contract import parse_density_frame
from common.schemas.leo_drag import LeoDragAssessment, LeoDragRequest

from intelligence.cli import fetch_release


class DragNotReadyError(RuntimeError):
    pass


class DragDomainError(ValueError):
    pass


def assess_drag(request: LeoDragRequest, *, release=None) -> LeoDragAssessment:
    now = datetime.now(UTC)
    try:
        release = release if release is not None else fetch_release('atmospheric-density')
        frame = pd.read_csv(io.StringIO(release.artifacts[0].csv_text))
        entry = get_config().models_registry['models']['atmospheric_density']
        density = parse_density_frame(frame, max_age_hours=entry.get('max_age_hours', 6), now=now)
        if release.published_at > now:
            raise ValueError('Future publication')
        from intelligence_core.api import assess_density, ModelDomainError
    except Exception:
        raise DragNotReadyError('Atmospheric density or impact backend is not ready') from None
    try:
        return assess_density(request, density=density, release_id=release.release_id, computed_at=now)
    except ModelDomainError as exc:
        raise DragDomainError(str(exc)) from None
    except (ValueError, ArithmeticError):
        raise DragNotReadyError('Atmospheric density grid is not ready') from None
