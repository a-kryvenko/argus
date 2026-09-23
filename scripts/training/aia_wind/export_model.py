"""Package selected Ridge with pre-2025 residual uncertainty and audit2025."""
import json,sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import brier_score_loss,roc_auc_score,average_precision_score
import run_experiment as selected
from utils import digest
from forecast.inference.aia_wind import FORMAT,ridge_prediction,distribution

DEST=Path('data/models/argus-plasma-speed-aia-ridge-v1.joblib')
METRICS=Path('data/metrics/plasma')


def export():
    selected.verify();plan=json.loads(selected.PLAN.read_text());solar=selected.solar_data()
    ridge=joblib.load(selected.OUT/'model.joblib');columns=ridge['models'][0]['ridge']['columns']
    history=pd.concat([pd.read_parquet(selected.ROOT/f'frames/year={y}.parquet') for y in [2023,2024]],ignore_index=True)
    uncertainty_bounds=[(h,h) for h in range(1,121)]
    samples=[dict(aia=[],dlinear=[]) for _ in uncertainty_bounds];audits=[]
    for q in plan['validation_quarters']:
        start=pd.Timestamp(year=2024,month=1+3*(q-1),day=1,tz='UTC');end=start+pd.DateOffset(months=3)
        train=selected.training(history,plan['train_start'],start);val=selected.training(history,start,end)
        models=selected.fit_variant(train,solar,columns,True,plan)
        point=selected.prediction(val,solar,dict(models=models,scales=ridge['scales']))
        for i,(lo,hi) in enumerate(uncertainty_bounds):
            mask=val.lead_hours.between(lo,hi)
            samples[i]['aia'].extend((val.target_v.to_numpy()-point)[mask].tolist())
            samples[i]['dlinear'].extend((val.target_v-val.dlinear_v)[mask].tolist())
        audits.append(dict(train_last_target=str(train.valid_time.max()),validation_start=str(start),validation_end_exclusive=str(end),rows=len(val)))
    uncertainty=[]
    for (lo,hi),data in zip(uncertainty_bounds,samples):
        row=dict(lo=lo,hi=hi,centering={})
        for mode,errors in data.items():
            values=np.asarray(errors,dtype=float);center=float(np.median(values))
            row[mode]=np.sort(values-center);row['centering'][mode]=center
        uncertainty.append(row)
    bundle=dict(format=FORMAT,version=1,lead_hours=120,buckets=plan['horizon_buckets'],models={},
        feature_columns=columns,feature_models={'dlinear_v':{'bundle':joblib.load(selected.DLINEAR)}},
        ridge=ridge,plan=plan,thresholds=[450,500,600],uncertainty=uncertainty,
        provenance=dict(ridge_sha256=digest(selected.OUT/'model.joblib'),dlinear_sha256=digest(selected.DLINEAR),
            exporter_sha256=digest(__file__),point_training_end=plan['refit_end'],uncertainty_folds=audits,
            uncertainty_method='Empirical held-out residuals separately for each target hour, centered to retain the frozen point prediction as q50. Separate DLinear fallback distribution. Validation quarters also used for Ridge-scale selection, so calibration is not independent. No2025 labels used.',
            historical_test=True))
    DEST.parent.mkdir(parents=True,exist_ok=True);temp=DEST.with_suffix('.part');joblib.dump(bundle,temp,compress=3);temp.replace(DEST)
    # The entire bundle, including uncertainty, is frozen BEFORE reading2025 labels.
    frozen=digest(DEST);METRICS.mkdir(parents=True,exist_ok=True)
    (METRICS/'frozen.json').write_text(json.dumps(dict(model_sha256=frozen,provenance=bundle['provenance']),indent=2))
    evaluate()


def evaluate():
    """Evaluate the deployed frozen artifact without fitting or overwriting it."""
    selected.verify()
    bundle=joblib.load(DEST);frozen=digest(DEST)
    METRICS.mkdir(parents=True,exist_ok=True)
    frame=pd.read_parquet(selected.ROOT/'frames/year=2025.parquet');point,active=ridge_prediction(frame,bundle)
    expected=pd.read_parquet(selected.OUT/'predictions.parquet')
    np.testing.assert_allclose(point,expected.aia_ridge,rtol=0,atol=1e-7)
    output=distribution(frame,bundle,point,active);output=output[output.common_available&output.target_v.notna()]
    regressions=[];classifications={t:[] for t in bundle['thresholds']}
    for lead,part in output.groupby('lead_hours'):
        y=part.target_v.to_numpy();err=part.v_q50.to_numpy()-y
        row=dict(lead_hours=int(lead),n=len(part),mae=float(abs(err).mean()),rmse=float(np.sqrt(np.square(err).mean())),
            coverage_80=float(((y>=part.v_q10)&(y<=part.v_q90)).mean()),lower_tail=float((y<part.v_q10).mean()),upper_tail=float((y>part.v_q90).mean()),interval_width_80=float((part.v_q90-part.v_q10).mean()))
        for name,alpha in [('q10',.1),('q50',.5),('q90',.9)]:
            e=y-part['v_'+name].to_numpy();row[name+'_pinball']=float(np.maximum(alpha*e,(alpha-1)*e).mean())
        regressions.append(row)
        for threshold in bundle['thresholds']:
            labels=y>=threshold;p=part[f'p_v_ge_{threshold}'].to_numpy();pred=p>=.5
            tp=int((labels&pred).sum());fp=int((~labels&pred).sum());fn=int((labels&~pred).sum());tn=int((~labels&~pred).sum())
            bins=np.minimum((p*10).astype(int),9);rel=[]
            for b in range(10):
                mask=bins==b
                if mask.any():rel.append(f'{p[mask].mean():.3f}_{labels[mask].mean():.3f}')
            denom=(tp+fn)*(fn+tn)+(tp+fp)*(fp+tn)
            classifications[threshold].append(dict(lead_hours=int(lead),n=len(p),brier=float(brier_score_loss(labels,p)),roc_auc=float(roc_auc_score(labels,p)) if np.unique(labels).size==2 else np.nan,avg_precision=float(average_precision_score(labels,p)) if labels.any() else np.nan,reliability=';'.join(rel),threat_score=tp/(tp+fp+fn) if tp+fp+fn else 0.,heidke=2*(tp*tn-fp*fn)/denom if denom else 0.))
    (METRICS/'speed_quantile').mkdir(exist_ok=True);(METRICS/'speed_proba').mkdir(exist_ok=True)
    pd.DataFrame(regressions).to_csv(METRICS/'speed_quantile/regression.csv',index=False)
    for t,rows in classifications.items():pd.DataFrame(rows).to_csv(METRICS/f'speed_proba/threshold_{t}.csv',index=False)
    (METRICS/'evaluation.json').write_text(json.dumps(dict(model_sha256=frozen,observed_rows=len(output),test_year=2025,issue_stride_hours=6,point_predictions_unchanged=True,calibration_coverage_80=float(((output.target_v>=output.v_q10)&(output.target_v<=output.v_q90)).mean())),indent=2))
    print(pd.DataFrame(regressions).query('lead_hours in [24,72,96,120]').to_string(index=False),flush=True)
    for t,rows in classifications.items():print('threshold',t,pd.DataFrame(rows).query('lead_hours==96').to_dict('records'),flush=True)
    print('Evaluated',DEST,'sha256',frozen,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['evaluate','export'],nargs='?',default='evaluate')
    args=parser.parse_args()
    with threadpool_limits(limits=4):globals()[args.command]()
