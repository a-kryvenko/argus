from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/training/aia_wind'))
from alignment import align
from run_experiment import PLAN,training,residual
import stream_ridge

def fixture():
 plan=json.loads(PLAN.read_text());times=pd.date_range('2024-01-01',periods=41,freq='6h',tz='UTC')
 solar=pd.DataFrame(dict(observed_at=times,available_at=times+pd.Timedelta(hours=2),aia_valid_fraction=1.,aia_b0_deg=0.,aia_area_sector=np.arange(len(times),dtype=float),aia_delta_24h_sector=0.,aia_24h_separation_h=24.))
 issue=pd.Timestamp('2024-01-10T00:00Z');leads=np.array([24,48,72,95,96,120])
 frame=pd.DataFrame(dict(issue_time=issue,valid_time=issue+pd.to_timedelta(leads,unit='h'),lead_hours=leads,dlinear_v=400.,target_v=500.,calendar_sin=0.,calendar_cos=1.))
 return frame,solar,plan

def test_fixed_window_and_latest_boundary():
 f,s,p=fixture();a=align(f,s,p)
 np.testing.assert_allclose(a.aia_age_hours.iloc[:2],[72,48])
 assert 24 < a.aia_age_hours.iloc[2] < 30  # Latest unavailable endpoint is excluded.
 np.testing.assert_allclose(a.aia_age_hours.iloc[-2:],[6,6])
 assert a.selected_frames.iloc[0]>1 and a.selected_frames.iloc[-1]==1

def test_unavailable_images_and_targets_do_not_affect_features():
 f,s,p=fixture();a=align(f,s,p);f.target_v=99999;s.loc[s.available_at>f.issue_time.iloc[0],'aia_area_sector']=99999
 pd.testing.assert_frame_equal(a.drop(columns='target_v'),align(f,s,p).drop(columns='target_v'))

def test_missing_history_is_not_backfilled():
 f,s,p=fixture();s=s[s.observed_at>=pd.Timestamp('2024-01-09T00:00Z')];a=align(f,s,p)
 assert not a.aia_available.iloc[0] and np.isnan(a.aia_age_hours.iloc[0]) and a.aia_available.iloc[-1]

def test_training_excludes_missing_and_boundary_targets():
 t=pd.Timestamp('2024-04-01',tz='UTC');f=pd.DataFrame(dict(issue_time=[t-pd.Timedelta(hours=6)]*4,valid_time=[t-pd.Timedelta(hours=1),t,t-pd.Timedelta(hours=1),t-pd.Timedelta(hours=1)],lead_hours=[5,6,5,5],target_v=[400.,400.,np.nan,400.],common_available=[True,True,True,False]))
 assert len(training(f,'2023-01-01T00:00:00Z',t))==1

def test_ridge_unsupported_inputs_fall_back():
 frame=pd.DataFrame(dict(aia_frame_index=[0,0,0],aia_available=[1,1,1],dlinear_v=[390.,410.,450.],target_v=[400.,425.,470.],aia_area_x=[.1,.2,.3]))
 bundle=stream_ridge.fit(frame,pd.DataFrame(),['dlinear_v','aia_area_x'])
 frame.loc[0,'aia_area_x']=np.nan;frame.loc[1,'aia_available']=0
 result=residual(frame,pd.DataFrame(),bundle)
 assert result[0]==result[1]==0 and np.isfinite(result[2])
