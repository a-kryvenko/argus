import copy
import numpy as np
import pandas as pd
import pytest
from forecast.inference.density_proswin import DensityProswinForecaster, FEATURES, history_features
from .test_density_dlinear import bundle as baseline_bundle
from .test_proswin_blend import record


@pytest.fixture(scope='module')
def bundle():
    import lightgbm as lgb
    x=pd.DataFrame(np.zeros((30,len(FEATURES))),columns=FEATURES)
    model=lgb.LGBMRegressor(n_estimators=1,n_jobs=1,verbosity=-1).fit(x,np.full(30,9.))
    b=baseline_bundle();b['settings']['horizon']=120
    b['weights']=np.tile([0.,0.,1.],(120,1));b['bias']=np.zeros(120)
    b['residual_offsets']=np.tile([-2.,.2,3.],(120,1))
    return {'format':'density_proswin_blend','version':1,'dlinear':b,'features':FEATURES,
            'heads':[model.booster_.model_to_string()]*96,'blend_weights':np.full(96,.5),
            'interval_offsets':np.tile([-1.,2.],(96,1))}


def inputs():
    issue=pd.Timestamp('2026-01-05T12:00Z')
    times=pd.date_range(issue-pd.Timedelta(hours=30),issue,freq='h')
    return issue,pd.DataFrame({'issue_time':times,'n':5.}),pd.DataFrame({'issue_time':times,'v':400.})


def test_blend_quantiles_and_exact_per_horizon_fallback(bundle):
    issue,n,v=inputs();m=DensityProswinForecaster(bundle)
    expected=m.dlinear.frame(issue,n)
    actual=m.frame(issue,n,v,[record(issue,lead=24)])
    np.testing.assert_allclose(actual.loc[23,['n_q10','n_q50','n_q90']].to_numpy(float),[6.1,7.1,9.1])
    cols=['n_q10','n_q50','n_q90']
    pd.testing.assert_frame_equal(actual.drop(index=23)[cols],expected.drop(index=23)[cols])
    assert m.last_status=={'candidate_leads':1,'fallback_leads':119}
    assert not actual.loc[96:,'proswin_weight'].any()


@pytest.mark.parametrize('failure',['missing','stale_speed','future','wrong_slot','nan_result','head_error'])
def test_failures_retain_all_dlinear_quantiles(bundle,failure):
    issue,n,v=inputs();m=DensityProswinForecaster(bundle);records=[record(issue,lead=24)]
    if failure=='missing':records=[]
    if failure=='stale_speed':v=v.iloc[:-3]
    if failure=='future':records[0]['available_at']=issue+pd.Timedelta(seconds=1)
    if failure=='wrong_slot':records[0]['image_slot']-=pd.Timedelta(hours=1)
    if failure in ('nan_result','head_error'):
        class Broken:
            def predict(self,*a,**k):
                if failure=='head_error':raise RuntimeError('test prediction failure')
                return [np.nan]
        m.heads[23]=Broken()
    actual=m.frame(issue,n,v,records)
    pd.testing.assert_frame_equal(actual[['n_q10','n_q50','n_q90']],m.dlinear.frame(issue,n)[['n_q10','n_q50','n_q90']])


def test_history_availability_gaps_and_future_values():
    issue,n,v=inputs();v['v']=np.arange(len(v))+400.
    row=history_features(issue,v,'v')
    assert row['v_count_24h']==24 and row['v_change_24h']==24
    future=pd.DataFrame({'issue_time':[issue+pd.Timedelta(hours=1)],'v':[1e7]})
    assert row==history_features(issue,pd.concat([v,future]),'v')
    assert np.isnan(history_features(issue,v.iloc[:-3],'v')['v_speed'])
    assert np.isnan(history_features(issue,v.tail(2),'v')['v_std_24h'])


def test_snapshot_round_trip_and_delayed_frozen_job(bundle):
    from common.schemas.forecast_inputs import ForecastInputs
    from common.schemas.observation import Observation
    from forecast.api import SWDensityFS,calculate_snapshot
    issue,n,v=inputs();cutoff=issue+pd.Timedelta(minutes=1);ready=cutoff+pd.Timedelta(minutes=2)
    r=record(issue,lead=96);r.update(available_at=ready,source_cutoff=cutoff)
    source=ForecastInputs(as_of=cutoff,read_at=cutoff,observations=Observation(points=[]),
                          density_observations=n.to_dict('records'),speed_observations=v.to_dict('records'),
                          proswin_predictions=[r],proswin_ready_at=ready)
    source=ForecastInputs.model_validate_json(source.model_dump_json())
    result=calculate_snapshot(SWDensityFS(bundle),source,issue_time=issue,model_info={})
    assert result.frame.iloc[95].n_q50==pytest.approx(7.1)
    assert result.frame.iloc[94].n_q50==pytest.approx(5.2)


def test_invalid_model_configuration_fails_before_use(bundle):
    b=copy.deepcopy(bundle);b['blend_weights'][0]=2
    with pytest.raises(ValueError,match='configuration'):DensityProswinForecaster(b)
