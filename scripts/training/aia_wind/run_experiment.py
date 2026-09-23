"""Selected DLinear + six-hour AIA193 fixed-arrival-window bucket Ridge."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from utils import digest
from aia_features import build_features
from alignment import align
from backtest import load_year,feature_columns
from stream_ridge import design
import stream_ridge
PLAN=Path(__file__).with_name('plan.json')
ROOT=Path('data/experiments/aia_wind/selected_fixed_window_ridge_v1')
OUT=Path('data/metrics/aia_wind/selected_fixed_window_ridge_v1')
RAW=Path('data/raw/aia193_6h_2023_2025')
OOF=Path('data/training/dlinear_oof/annual_expanding_vs_fixed_v1/fixed_initial')
DLINEAR=Path('data/models/dlinear_walk_forward/annual_expanding_vs_fixed_v1/fixed_initial/before_2000.joblib')
CODE=['run_experiment.py','alignment.py','backtest.py','aia_features.py','stream_ridge.py','utils.py','plan.json']

def window(frame, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return frame[(frame.issue_time >= start) & (frame.issue_time < end)
                 & (frame.valid_time < end)].sort_values(["issue_time", "lead_hours"]).reset_index(drop=True)

def training(frame,start,end):
    f=window(frame,start,end)
    return f[f.common_available & f.target_v.notna()]

def unsupported_missing(frame, solar, bundle):
    required = np.ones(len(bundle["columns"]), dtype=bool)
    required[bundle["indicators"]] = False
    result = np.zeros(len(frame), dtype=bool)
    for start in range(0, len(frame), 4096):
        x = design(frame.iloc[start:start+4096], solar, bundle["columns"])
        result[start:start+len(x)] = ~np.isfinite(x[:, required]).all(axis=1)
    return result

def residual(frame, solar, bundle):
    """Zero residual for stale images or missing inputs unsupported by training."""
    result = np.zeros(len(frame))
    active = frame.aia_available.eq(1).to_numpy() & ~unsupported_missing(frame, solar, bundle)
    if active.any():
        result[active] = stream_ridge.predict(frame.loc[active],solar,bundle)
    return result

def fit_variant(train, solar, columns, bucketed, plan):
    bounds = plan["horizon_buckets"] if bucketed else [[1,120]]
    return [dict(lo=lo, hi=hi, ridge=stream_ridge.fit(
        train[train.lead_hours.between(lo,hi)],solar,columns,alpha=plan["ridge_alpha"])) for lo,hi in bounds]

def raw_correction(frame, solar, models):
    result = np.zeros(len(frame))
    assigned = np.zeros(len(frame), dtype=int)
    for model in models:
        mask = frame.lead_hours.between(model["lo"],model["hi"]).to_numpy()
        result[mask] = residual(frame.loc[mask],solar,model["ridge"])
        assigned[mask] += 1
    if not (assigned == 1).all():
        raise ValueError("Horizon models must partition all forecast rows")
    return result

def prediction(frame, solar, bundle):
    delta = raw_correction(frame,solar,bundle["models"])
    scales = np.zeros(len(frame))
    for row in bundle["scales"]:
        scales[frame.lead_hours.between(row["lo"],row["hi"]).to_numpy()] = row["scale"]
    result = frame.dlinear_v.to_numpy(dtype=float) + scales*delta
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite prediction")
    return result

def select(blocks,plan):
    scales=[]
    for lo,hi in plan['horizon_buckets']:
        scores=[]
        for scale in plan['shrinkage_candidates']:
            quarters=[]
            for frame,delta in blocks:
                mask=frame.lead_hours.between(lo,hi)&frame.common_available&frame.target_v.notna()
                quarters.append(float(abs(frame.dlinear_v.to_numpy()[mask]+scale*delta[mask]-frame.target_v.to_numpy()[mask]).mean()))
            scores.append((scale,float(np.mean(quarters))))
        scale,mae=min(scores,key=lambda x:x[1])
        scales.append(dict(lo=lo,hi=hi,scale=scale,validation_mae=mae))
    quarter_mae=[]
    for f,delta in blocks:
        weights=np.zeros(len(f))
        for s in scales:weights[f.lead_hours.between(s['lo'],s['hi'])]=s['scale']
        mask=f.common_available&f.target_v.notna()
        quarter_mae.append(float(abs(f.dlinear_v.to_numpy()[mask]+weights[mask]*delta[mask]-f.target_v.to_numpy()[mask]).mean()))
    return dict(scales=scales,validation_mae=float(np.mean(quarter_mae)),quarter_mae=quarter_mae)


def solar_data():
    return pd.read_parquet(ROOT/'features/features.parquet')


def features():
    plan=json.loads(PLAN.read_text())
    build_features(RAW,ROOT/'features',assumed_latency_hours=plan['assumed_latency_hours'],pair_tolerance_hours=plan['pair_tolerance_hours'])


def make_frame(year,solar,plan):
    solar=solar.copy();solar['aia_frame_index']=np.arange(len(solar))
    metadata=solar[['observed_at','available_at','aia_valid_fraction','aia_b0_deg','aia_frame_index']]
    frame=load_year(year,OOF,'data/clean/omni_hourly',metadata,leads=list(range(1,121)),stride_hours=plan['issue_stride_hours'],max_age_hours=plan['max_latest_age_hours'])
    if not frame.model_sha256.eq(digest(DLINEAR)).all():raise ValueError('Changed DLinear baseline')
    result=align(frame,solar,plan);result['common_available']=result.aia_available
    return result


def prepare():
    plan=json.loads(PLAN.read_text());solar=solar_data();(ROOT/'frames').mkdir(parents=True,exist_ok=True)
    for year in plan['data_years']:
        frame=make_frame(year,solar,plan)
        if frame.aia_available.mean()<.9:raise ValueError('Insufficient coverage')
        frame.to_parquet(ROOT/f'frames/year={year}.parquet',index=False)
        print('Prepared',year,len(frame),flush=True)


def protocol():
    plan=json.loads(PLAN.read_text())
    paths=[ROOT/'features/features.parquet',ROOT/'features/manifest.parquet',ROOT/'features/extraction.json',DLINEAR]
    for year in plan['data_years']:
        paths += [ROOT/f'frames/year={year}.parquet',OOF/f'year={year}.parquet',Path(f'data/clean/omni_hourly/omni_{year}.parquet')]
    return dict(inputs={str(p):digest(p) for p in paths},code={n:digest(Path(__file__).with_name(n)) for n in CODE})


def fit_bundle():
    plan=json.loads(PLAN.read_text());solar=solar_data();columns=feature_columns(solar)
    frame=pd.concat([pd.read_parquet(ROOT/f'frames/year={y}.parquet') for y in [2023,2024]],ignore_index=True)
    blocks=[]
    for q in plan['validation_quarters']:
        start=pd.Timestamp(year=2024,month=1+3*(q-1),day=1,tz='UTC');end=start+pd.DateOffset(months=3)
        train=training(frame,plan['train_start'],start);val=window(frame,start,end)
        models=fit_variant(train,solar,columns,True,plan)
        blocks.append((val,raw_correction(val,solar,models)))
    selected=select(blocks,plan)
    models=fit_variant(training(frame,plan['train_start'],plan['refit_end']),solar,columns,True,plan)
    return dict(models=models,scales=selected['scales'],selection=selected)


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'protocol.json').write_text(json.dumps(protocol(),indent=2))
    (OUT/'frozen.json').write_text(json.dumps(dict(protocol_sha256=digest(OUT/'protocol.json'),model_sha256=digest(OUT/'model.joblib')),indent=2))


def fit():
    bundle=fit_bundle();OUT.mkdir(parents=True,exist_ok=True)
    joblib.dump(bundle,OUT/'model.joblib',compress=3);freeze()
    (OUT/'selection.json').write_text(json.dumps(bundle['selection'],indent=2))


def verify():
    frozen=json.loads((OUT/'frozen.json').read_text())
    if protocol()!=json.loads((OUT/'protocol.json').read_text()):raise ValueError('Changed inputs/code; explicitly prepare/refit a new snapshot')
    if frozen['protocol_sha256']!=digest(OUT/'protocol.json') or frozen['model_sha256']!=digest(OUT/'model.joblib'):raise ValueError('Changed frozen model')
    manifest=pd.read_parquet(ROOT/'features/manifest.parquet')
    for p in manifest.cache_path:
        if not Path(p).is_file():raise ValueError(f'Missing selected cache: {p}')
    print('Selected model, inputs and cache verified',flush=True)


def evaluate():
    verify();plan=json.loads(PLAN.read_text());solar=solar_data();frame=pd.read_parquet(ROOT/'frames/year=2025.parquet')
    bundle=joblib.load(OUT/'model.joblib');result=frame[['issue_time','valid_time','lead_hours','target_v','dlinear_v','common_available']].copy()
    result['aia_ridge']=prediction(frame,solar,bundle)
    reference=OUT/'reference_predictions.parquet'
    if reference.exists():
        prior=pd.read_parquet(reference)
        for c in ['issue_time','valid_time','lead_hours','target_v','dlinear_v','common_available']:pd.testing.assert_series_equal(prior[c],result[c])
        np.testing.assert_allclose(prior.aia_ridge,result.aia_ridge,atol=1e-7,rtol=0)
    rows=[]
    for scope in ['common','all_with_fallback']:
        eligible=result.target_v.notna()&(result.common_available if scope=='common' else True)
        for lo,hi in [(1,120),*map(tuple,plan['horizon_buckets']),*[(h,h) for h in range(1,121)]]:
            part=result[eligible&result.lead_hours.between(lo,hi)]
            for name in ['dlinear_v','aia_ridge']:
                error=part[name]-part.target_v
                rows.append(dict(scope=scope,model=name,lo=lo,hi=hi,n=len(error),mae=float(abs(error).mean()),rmse=float(np.sqrt(np.square(error).mean())),bias=float(error.mean())))
    metrics=pd.DataFrame(rows);metrics.to_csv(OUT/'metrics.csv',index=False);result.to_parquet(OUT/'predictions.parquet',index=False)
    (OUT/'evaluation.json').write_text(json.dumps(dict(model_sha256=digest(OUT/'model.joblib'),metrics_sha256=digest(OUT/'metrics.csv'),predictions_sha256=digest(OUT/'predictions.parquet'),historical_evaluation=True),indent=2))
    print(metrics.query('scope=="common" and ((lo==1 and hi==120) or (lo==96 and hi==96))').to_string(index=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',nargs='?',default='verify',choices=['features','prepare','fit','evaluate','verify']);args=parser.parse_args()
    with threadpool_limits(limits=4):globals()[args.command]()
