"""Horizon-dependent speed blend. PROSWIN records are optional causal inputs."""
import numpy as np
import pandas as pd
from forecast.inference.aia_wind import distribution
from forecast.inference.rotation_dlinear import RotationDLinearForecaster

KNOTS = [1, 6, 12, 24, 48, 72, 96]
WEIGHTS = [0.03933113755799481, 0.06723997380003054, 0.1843781075839117,
           0.39179176251397624, 0.8019248836236577, 0.8019248836236577, 0.8438318175641963]
VERSION = 'proswin-dlinear-mae-2025-v2'


class ProswinBlendForecaster:
    def __init__(self, bundle):
        self.bundle = bundle
        self.dlinear = RotationDLinearForecaster(bundle['feature_models']['dlinear_v']['bundle'])

    def frame(self, issue_time, speed_history, predictions=(), *, ready_at=None, source_cutoff=None):
        issue = pd.Timestamp(issue_time)
        if issue.tzinfo is None: raise ValueError('Timezone-aware issue required')
        issue = issue.tz_convert('UTC').floor('h')
        cutoff = pd.Timestamp(source_cutoff) if source_cutoff is not None else issue
        ready_limit = pd.Timestamp(ready_at) if ready_at is not None else cutoff
        if cutoff.tzinfo is None or ready_limit.tzinfo is None or cutoff < issue or ready_limit < cutoff:
            raise ValueError('Invalid forecast preparation times')
        leads = np.arange(1, self.dlinear.horizon + 1)
        request = pd.DataFrame({'issue_time':issue, 'lead_hours':leads})
        base = self.dlinear.add_rotation_v(request, speed_history, column='dlinear_v', require_history=True)
        base['valid_time'] = issue + pd.to_timedelta(leads, unit='h')
        chosen = {}
        for item in predictions:
            r = item.model_dump() if hasattr(item, 'model_dump') else item
            try:
                valid, image, ready = (pd.Timestamp(r[k]) for k in ['valid_time','image_slot','available_at'])
                value = float(r['value'])
                if ready > cutoff:
                    source = pd.Timestamp(r.get('source_cutoff'))
                    if pd.isna(source) or source.tzinfo is None or source > cutoff:
                        continue
                if (any(t.tzinfo is None for t in [valid,image,ready]) or
                    valid-image != pd.Timedelta(hours=96) or ready > ready_limit or ready < image or
                    image > issue or not np.isfinite(value) or value <= 0 or
                    r.get('model_version') != 'proswin-fold1-nrt-v1'):
                    continue
                if valid not in chosen or ready < chosen[valid][0]: chosen[valid] = (ready,value)
            except (KeyError, TypeError, ValueError):
                continue
        p = np.array([chosen.get(t,(None,np.nan))[1] for t in base.valid_time])
        active = np.isfinite(p) & (leads <= 96)
        w = np.where(active,np.interp(leads,KNOTS,WEIGHTS),0.)
        point = base.dlinear_v.to_numpy().copy()
        point[active] = (1-w[active])*point[active] + w[active]*p[active]
        # Preserve DLinear's empirical uncertainty offsets until blend-specific
        # intervals are calibrated. Never reuse the old AIA Ridge correction.
        result = distribution(base,self.bundle,point,np.zeros(len(base),dtype=bool))
        result['proswin_weight'] = w
        result['proswin_v'] = p
        return result
