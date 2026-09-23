"""Frozen DLinear + fixed-arrival-window AIA Ridge and residual uncertainty."""
from datetime import UTC,datetime
import numpy as np
import pandas as pd
from common.adapters import observations_to_dataframe
from forecast.aia_alignment import align
from forecast.inference.rotation_dlinear import RotationDLinearForecaster

FORMAT='dlinear_aia_ridge_v1'


def ridge_prediction(frame,bundle):
    result=frame.dlinear_v.to_numpy(dtype=float).copy();corrected=np.zeros(len(frame),bool)
    for entry,scale in zip(bundle['ridge']['models'],bundle['ridge']['scales']):
        if (entry['lo'],entry['hi'])!=(scale['lo'],scale['hi']):raise ValueError('Mismatched Ridge buckets')
        rows=np.flatnonzero(frame.lead_hours.between(entry['lo'],entry['hi']).to_numpy())
        if not len(rows) or not scale['scale']:continue
        model=entry['ridge'];x=frame.iloc[rows][model['columns']].to_numpy(dtype=float)
        required=np.ones(len(model['columns']),bool);required[model['indicators']]=False
        active=frame.aia_available.to_numpy()[rows].astype(bool)&np.isfinite(x[:,required]).all(axis=1)
        if np.isinf(x).any():raise ValueError('Infinite AIA features')
        values=np.column_stack([np.nan_to_num(x,nan=0),np.isnan(x[:,model['indicators']]).astype(float)])
        delta=values@model['coefficient']+model['intercept']
        result[rows[active]]+=scale['scale']*delta[active];corrected[rows[active]]=True
    if not np.isfinite(result).all():raise ValueError('Nonfinite wind prediction')
    return result,corrected


def distribution(frame,bundle,point,corrected):
    result=frame.copy();result['v_q50']=point
    for entry in bundle['uncertainty']:
        bucket=frame.lead_hours.between(entry['lo'],entry['hi']).to_numpy()
        for mode,mask in [('aia',bucket&corrected),('dlinear',bucket&~corrected)]:
            ids=np.flatnonzero(mask)
            if not len(ids):continue
            samples=np.asarray(entry[mode],dtype=float)
            if len(samples)<100 or not np.isfinite(samples).all() or (np.diff(samples)<0).any():raise ValueError('Invalid frozen error distribution')
            for quantile,name in [(.1,'v_q10'),(.9,'v_q90')]:result.loc[result.index[ids],name]=point[ids]+np.quantile(samples,quantile)
            for threshold in bundle['thresholds']:
                # P(V >= threshold), empirical survivor, including equality.
                probability=1-np.searchsorted(samples,threshold-point[ids],side='left')/len(samples)
                result.loc[result.index[ids],f'p_v_ge_{threshold}']=probability
    return result


class AIAWindForecaster:
    def __init__(self,bundle):
        if bundle.get('format')!=FORMAT:raise ValueError('Unsupported AIA wind bundle')
        self.bundle=bundle;self.dlinear=RotationDLinearForecaster(bundle['feature_models']['dlinear_v']['bundle'])

    def frame(self,issue_time,speed_history,aia_features):
        issue=pd.Timestamp(issue_time)
        if issue.tzinfo is None:raise ValueError('Timezone-aware issue_time required')
        issue=issue.tz_convert('UTC').floor('h');lead=np.arange(1,121)
        request=pd.DataFrame(dict(issue_time=issue,lead_hours=lead))
        base=self.dlinear.add_rotation_v(request,speed_history,column='dlinear_v')
        if not np.isfinite(base.dlinear_v).all():raise ValueError('Insufficient observed hourly speed history for DLinear')
        base['valid_time']=issue+pd.to_timedelta(lead,unit='h');base['target_v']=np.nan
        phase=2*np.pi*(issue.dayofyear-1)/365.25;base['calendar_sin']=np.sin(phase);base['calendar_cos']=np.cos(phase)
        solar=pd.DataFrame() if aia_features is None else aia_features.copy()
        if not solar.empty:
            for c in ['slot_at','observed_at','available_at']:solar[c]=pd.to_datetime(solar[c],utc=True)
            solar=solar[(solar.available_at<=issue)&(solar.observed_at<=issue)&solar.slot_at.dt.hour.mod(6).eq(0)&solar.slot_at.dt.minute.eq(0)]
            solar=solar.sort_values('observed_at').drop_duplicates('observed_at',keep='first')
        if solar.empty:
            aligned=base.copy()
            for c in self.bundle['feature_columns']:
                if c not in aligned:aligned[c]=np.nan
            aligned['aia_available']=False
        else:
            # Availability has already been enforced for this single issue.
            # Receipt order may differ from observation order during backfill.
            solar=solar.copy();solar['available_at']=solar.observed_at
            aligned=align(base,solar,self.bundle['plan'])
        point,corrected=ridge_prediction(aligned,self.bundle)
        return distribution(aligned,self.bundle,point,corrected)


class AIAWindServiceMixin:
    def __init__(self,models_bundle):
        super().__init__(models_bundle)
        self.uses_aia=models_bundle.get('format')==FORMAT
        self._aia=AIAWindForecaster(models_bundle) if self.uses_aia else None
        if self.uses_aia:self.thresholds=models_bundle['thresholds']

    def forecast(self,observations,*,issue_time=None,speed_history=None,aia_features=None):
        if not self.uses_aia:return super().forecast(observations,issue_time=issue_time,speed_history=speed_history)
        issue_time=issue_time or datetime.now(UTC)
        history=observations_to_dataframe(observations) if speed_history is None else speed_history
        return self.forecast_from_df(self._aia.frame(issue_time,history,aia_features))
