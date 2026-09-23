from datetime import UTC,datetime
import numpy as np
import pandas as pd
import pytest
from common.schemas.observation import Observation
from forecast.api import SWSpeedFS,SWSpeedProbaFS
from forecast.inference.aia_wind import AIAWindForecaster,FORMAT


def bundle():
    dependency=dict(format='rotation_dlinear',version=1,settings=dict(columns=['v'],segments={'rotations':[[-2,0]]},horizon=120,ffill_limit_hours=1,mean=[0.],std=[1.]),weights=np.tile([0.,0.,1.],(120,1)),bias=np.zeros(120))
    return dict(format=FORMAT,lead_hours=120,buckets=[[1,120]],models={},feature_columns=['dlinear_v','aia_area_sector'],feature_models={'dlinear_v':{'bundle':dependency}},
        ridge=dict(models=[dict(lo=1,hi=120,ridge=dict(columns=['dlinear_v','aia_area_sector'],indicators=np.array([],dtype=int),coefficient=np.array([0.,100.]),intercept=0.))],scales=[dict(lo=1,hi=120,scale=1.)]),
        plan=dict(fixed_delay_hours=96,max_latest_age_hours=36,latest_from_horizon=96,window_radius_hours=24,window_sigma_hours=12),thresholds=[450,500,600],
        uncertainty=[dict(lo=1,hi=120,aia=np.linspace(-100,100,201),dlinear=np.linspace(-150,150,301))])


def inputs():
    issue=pd.Timestamp('2025-02-10T12:00:00Z')
    history=pd.DataFrame(dict(issue_time=pd.date_range(issue-pd.Timedelta(hours=2),periods=3,freq='h'),v=400.))
    times=pd.date_range(issue-pd.Timedelta(days=6),issue-pd.Timedelta(hours=6),freq='6h')
    solar=pd.DataFrame(dict(slot_at=times,observed_at=times,available_at=times+pd.Timedelta(hours=2),aia_valid_fraction=1.,aia_b0_deg=0.,aia_area_sector=.4,aia_delta_24h_sector=0.))
    return issue,history,solar


def test_quantile_and_threshold_share_point_distribution_and_keep_contract():
    issue,history,solar=inputs();b=bundle()
    q=SWSpeedFS(b).forecast(Observation(points=[]),issue_time=issue,speed_history=history,aia_features=solar)
    p=SWSpeedProbaFS(b).forecast(Observation(points=[]),issue_time=issue,speed_history=history,aia_features=solar)
    assert len(q.points)==len(p.points)==120
    assert q.points[95].v_q50==pytest.approx(440.)
    for v,prob in zip(q.points,p.points):
        assert v.v_q10<=v.v_q50<=v.v_q90
        assert 0<=prob.p_v_ge_600<=prob.p_v_ge_500<=prob.p_v_ge_450<=1
        assert v.valid_time==prob.valid_time
    assert q.points[95].valid_time==issue+pd.Timedelta(hours=96)


def test_actual_receipt_hourly_sampling_and_future_speed_are_causal():
    issue,history,solar=inputs();model=AIAWindForecaster(bundle());original=model.frame(issue,history,solar)
    changed=solar.copy();changed.loc[changed.index[-1],'available_at']=issue+pd.Timedelta(hours=1)
    changed.loc[changed.index[-1],'aia_area_sector']=999.
    out=model.frame(issue,history,changed)
    assert out.v_q50.iloc[95]==pytest.approx(original.v_q50.iloc[95])
    hourly=solar.iloc[[-1]].copy();hourly.slot_at=issue-pd.Timedelta(hours=1);hourly.observed_at=hourly.slot_at;hourly.available_at=hourly.slot_at;hourly.aia_area_sector=999
    result=model.frame(issue,pd.concat([history,pd.DataFrame(dict(issue_time=[issue+pd.Timedelta(hours=1)],v=[9999.]))]),pd.concat([solar,hourly]))
    np.testing.assert_allclose(result.v_q50,original.v_q50)


def test_no_aia_falls_back_to_dlinear_with_fallback_uncertainty():
    issue,history,_=inputs();out=AIAWindForecaster(bundle()).frame(issue,history,None)
    np.testing.assert_allclose(out.v_q50,400.)
    np.testing.assert_allclose(out.v_q10,280.)
    with pytest.raises(ValueError,match='Insufficient'):
        AIAWindForecaster(bundle()).frame(issue,history.iloc[:0],None)


def test_shuffled_actual_receipts_do_not_require_observation_order():
    issue,history,solar=inputs();solar.available_at=issue-pd.Timedelta(minutes=1)
    out=AIAWindForecaster(bundle()).frame(issue,history,solar.sample(frac=1,random_state=2))
    np.testing.assert_allclose(out.v_q50,440.)
