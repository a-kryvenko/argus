import copy
import numpy as np
import pandas as pd
import pytest
from forecast.inference.temperature_proswin import TemperatureProswinForecaster,FEATURES,hourly_histories
from .test_proswin_blend import record

@pytest.fixture(scope='module')
def bundle():
    import lightgbm as lgb
    heads={};offsets={}
    for j,(name,columns) in enumerate(FEATURES.items(),1):
        x=pd.DataFrame(np.zeros((30,len(columns))),columns=columns)
        model=lgb.LGBMRegressor(n_estimators=1,n_jobs=1,verbosity=-1).fit(x,np.full(30,j*10000.))
        heads[name]=[model.booster_.model_to_string()]*96
        offsets[name]=np.tile([-j*100.,j*200.],(96,1))
    return {'format':'temperature_proswin','version':1,'features':FEATURES,'heads':heads,'interval_offsets':offsets}


def inputs():
    issue=pd.Timestamp('2026-01-05T12:00Z');times=pd.date_range(issue-pd.Timedelta(hours=30),issue,freq='h')
    return issue,{t:pd.DataFrame({'issue_time':times,t:value}) for t,value in [('t',100000.),('n',5.),('v',400.)]}


def test_per_lead_fallback_and_own_intervals(bundle):
    issue,h=inputs();m=TemperatureProswinForecaster(bundle);f=m.frame(issue,h,[record(issue,lead=24)])
    np.testing.assert_allclose(f.loc[23,['t_q10','t_q50','t_q90']].to_numpy(float),[29700,30000,30600])
    np.testing.assert_allclose(f.loc[0,['t_q10','t_q50','t_q90']].to_numpy(float),[19800,20000,20400])
    assert m.last_status['candidate_leads']==1 and m.last_status['history_fallback_leads']==95
    h['n']=h['n'].iloc[:0];f=m.frame(issue,h,[record(issue,lead=24)])
    np.testing.assert_allclose(f[['t_q10','t_q50','t_q90']],np.tile([9900,10000,10200],(96,1)))
    assert m.last_status['temperature_only_leads']==96


@pytest.mark.parametrize('failure',['late','wrong_slot','nan','error'])
def test_candidate_failure_retains_history(bundle,failure):
    issue,h=inputs();m=TemperatureProswinForecaster(bundle);r=record(issue,lead=24)
    if failure=='late':r['available_at']=issue+pd.Timedelta(hours=1)
    if failure=='wrong_slot':r['image_slot']-=pd.Timedelta(hours=1)
    if failure in ['nan','error']:
        class Broken:
            def predict(self,*args,**kwargs):
                if failure=='error':raise RuntimeError('broken head')
                return [np.nan]
        m.heads['tnv_proswin'][23]=Broken()
    assert (m.frame(issue,h,[r]).t_q50==20000).all()


def test_missing_temperature_fails(bundle):
    issue,h=inputs();h['t']=h['t'].iloc[:-3]
    with pytest.raises(ValueError,match='temperature observation'):TemperatureProswinForecaster(bundle).frame(issue,h)


def payload(issue):
    return {'resolution_seconds':3600,'gap_filling':'none','series':{t:{'unit':unit,'aggregation':'mean_min_max',
        'location':'L1','time_basis':'measurement','propagated':False,'points':[{
        'observed_at':issue-pd.Timedelta(hours=1),'interval_end':issue,'received_at':issue,
        'quality':'unverified','count':20,'expected_count':60,'value':value}]} for t,unit,value in [('t','K',100000.),('n','cm⁻³',5.),('v','km/s',400.)]}}


def test_closed_hour_receipt_and_no_filling():
    issue,_=inputs();p=payload(issue)
    h=hourly_histories(p,issue_time=issue,as_of=issue)
    assert h['t'].iloc[0].issue_time==issue and h['t'].iloc[0].t==100000.
    p['series']['t']['points'][0]['received_at']=issue+pd.Timedelta(seconds=1)
    assert hourly_histories(p,issue_time=issue,as_of=issue)['t'].empty
    p=payload(issue);p['series']['t']['points'][0].update(observed_at=issue,interval_end=issue+pd.Timedelta(hours=1))
    assert hourly_histories(p,issue_time=issue,as_of=issue)['t'].empty


def test_snapshot_and_delayed_proswin_job(bundle):
    from common.schemas.forecast_inputs import ForecastInputs
    from forecast.api import calculate_snapshot
    issue,_=inputs();cutoff=issue+pd.Timedelta(minutes=1);ready=cutoff+pd.Timedelta(minutes=2)
    r=record(issue,lead=96);r.update(available_at=ready,source_cutoff=cutoff)
    i=ForecastInputs(as_of=cutoff,read_at=cutoff,observations={'points':[]},solar_wind_hourly=payload(issue),
        proswin_predictions=[r],proswin_ready_at=ready)
    i=ForecastInputs.model_validate_json(i.model_dump_json())
    result=calculate_snapshot(TemperatureProswinForecaster(bundle),i,issue_time=issue,model_info={})
    assert result.name=='plasma_temperature_quantile' and len(result.frame)==96
    assert result.frame.t_q50.iloc[-1]==30000


def test_artifact_validation(bundle):
    b=copy.deepcopy(bundle);b['interval_offsets']['tnv_proswin'][0]=[5,-5]
    with pytest.raises(ValueError,match='intervals'):TemperatureProswinForecaster(b)
