import numpy as np
import pandas as pd
from .test_aia_wind_service import bundle, inputs
from forecast.inference.proswin_blend import ProswinBlendForecaster, WEIGHTS


def record(issue,lead=24,value=600,**kwargs):
    valid=issue+pd.Timedelta(hours=lead)
    return dict(valid_time=valid,image_slot=valid-pd.Timedelta(hours=96),
        available_at=issue-pd.Timedelta(hours=1),value=value,model_version='proswin-fold1-nrt-v1',**kwargs)


def test_mixture_and_pure_dlinear_fallback_per_valid_time():
    issue,history,_=inputs();model=ProswinBlendForecaster(bundle())
    out=model.frame(issue,history,[record(issue)])
    assert out.iloc[23].v_q50==400*(1-WEIGHTS[3])+600*WEIGHTS[3]
    assert out.iloc[22].v_q50==400
    assert out.iloc[96].v_q50==400
    assert np.all(out.v_q10<=out.v_q50) and np.all(out.v_q50<=out.v_q90)


def test_future_wrong_slot_and_invalid_records_do_not_contaminate_fallback():
    issue,history,_=inputs();model=ProswinBlendForecaster(bundle())
    future=record(issue);future['available_at']=issue+pd.Timedelta(seconds=1)
    shifted=record(issue);shifted['image_slot']-=pd.Timedelta(hours=1)
    invalid=record(issue);invalid['value']=float('nan')
    other=record(issue);other['model_version']='unknown'
    out=model.frame(issue,history,[future,shifted,invalid,other,{}])
    np.testing.assert_array_equal(out.v_q50,np.full(120,400.))
    assert (out.proswin_weight==0).all()


def test_snapshot_passes_proswin_into_main_forecast():
    from common.schemas.forecast_inputs import ForecastInputs
    from common.schemas.observation import Observation
    from forecast.api import SWSpeedFS
    issue, history, _ = inputs()
    source = ForecastInputs(as_of=issue, read_at=issue, observations=Observation(points=[]),
                           speed_observations=history.to_dict('records'), proswin_predictions=[record(issue)])
    service = SWSpeedFS(bundle())
    result = service.forecast(source.observations, issue_time=issue, **service.snapshot_options(source))
    assert result.points[23].v_q50 == 400*(1-WEIGHTS[3])+600*WEIGHTS[3]
    assert result.points[22].v_q50 == 400


def test_job_result_after_hour_boundary_uses_frozen_source_cutoff():
    issue, history, _ = inputs()
    cutoff = issue + pd.Timedelta(minutes=10)
    finished = cutoff + pd.Timedelta(minutes=2)
    item = record(issue)
    item.update(source_cutoff=cutoff, available_at=finished)
    model = ProswinBlendForecaster(bundle())
    out = model.frame(issue, history, [item], source_cutoff=cutoff, ready_at=finished)
    assert out.iloc[23].proswin_weight > 0
    item['source_cutoff'] = cutoff + pd.Timedelta(seconds=1)
    assert model.frame(issue, history, [item], source_cutoff=cutoff, ready_at=finished).iloc[23].proswin_weight == 0
    item.pop('source_cutoff')
    assert model.frame(issue, history, [item], source_cutoff=cutoff, ready_at=finished).iloc[23].proswin_weight == 0
