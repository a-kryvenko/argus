"""Adapt a validated Prophet release to the private impact backend."""
from datetime import UTC, datetime
import io

import pandas as pd
from common.config import get_config
from common.density_contract import parse_density_frame
from common.schemas.leo_drag import DragSource, LeoDragAssessment, LeoDragRequest

from argus_intelligence.cli import fetch_release


class DragNotReadyError(RuntimeError):
    pass


class DragDomainError(ValueError):
    pass


def assess_drag(request: LeoDragRequest) -> LeoDragAssessment:
    now = datetime.now(UTC)
    try:
        release = fetch_release('atmospheric-density')
        frame = pd.read_csv(io.StringIO(release.artifacts[0].csv_text))
        entry = get_config().models_registry['models']['atmospheric_density']
        density = parse_density_frame(frame, max_age_hours=entry.get('max_age_hours', 6), now=now)
        if release.published_at > now:
            raise ValueError('Future publication')
        from intelligence_core.api import calculate_drag, ModelDomainError
    except Exception:
        raise DragNotReadyError('Atmospheric density or impact backend is not ready') from None
    points = density.predictions[:request.horizon_hours + 1]
    try:
        result = calculate_drag(**request.model_dump(exclude={'horizon_hours', 'thresholds'}),
                                grids=[[cell.model_dump() for cell in point.cells] for point in points])
    except ModelDomainError as exc:
        raise DragDomainError(str(exc)) from None
    except (ValueError, ArithmeticError):
        raise DragNotReadyError('Atmospheric density grid is not ready') from None
    loss = result['estimated_altitude_loss_m']
    risk = 'not_assessed'
    reason = 'No altitude-loss thresholds supplied; physical estimates only.'
    if request.thresholds is not None:
        thresholds = request.thresholds
        risk = ('high' if loss >= thresholds.high_altitude_loss_m else
                'elevated' if loss >= thresholds.elevated_altitude_loss_m else 'low')
        reason = (f'Estimated cumulative altitude loss {loss:.6g} m over {request.horizon_hours} h; '
                  f'user thresholds: elevated >= {thresholds.elevated_altitude_loss_m:g} m, '
                  f'high >= {thresholds.high_altitude_loss_m:g} m.')
    return LeoDragAssessment(
        computed_at=now, start_time=density.issue_time, end_time=points[-1].valid_time,
        inputs=request,
        source=DragSource(release_id=release.release_id, **density.model_dump(include={
            'issue_time', 'observed_at', 'dtc_observed_at', 'history_start',
            'background_method', 'background_interpolated_days', 'dtc_method'})),
        **{key: value for key, value in result.items() if key != 'predictions'},
        predictions=[dict(valid_time=point.valid_time, lead_hours=point.lead_hours, **values)
                     for point, values in zip(points, result['predictions'], strict=True)],
        drag_risk=risk, risk_reason=reason,
        assumptions=[
            'Horizon starts at the density release issue_time, not the request time.',
            'Observed solar and geomagnetic drivers are held constant; future storms are not predicted.',
            'Circular orbit at fixed mean altitude; spherical Earth and co-rotating atmosphere, no winds or maneuvers.',
            'Constant effective area and drag coefficient; longitude-averaged density and uniform orbital-phase averaging.',
            'First-order altitude loss; calculations exceeding 1000 m are outside the model domain.',
            'Delta-v loss is accumulated along-track drag impulse, not the change in orbital speed.',
            'Risk categories use user thresholds; no probability or calibrated uncertainty is estimated.',
        ])
