from datetime import UTC,datetime,timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock,Mock
import asyncio
import numpy as np
import pytest
from argus_clio.services.aia import feature_frames,load_aia_features


def test_six_hour_features_ignore_hourly_frames_and_late_receipts(tmp_path):
    start=datetime(2025,1,1,tzinfo=UTC);records=[]
    for i,value in [(0,.1),(1,.9),(24,.4),(25,.8),(30,.6)]:
        path=tmp_path/f'{i}.npz';np.savez(path,dark=np.full((61,61),value))
        slot=start+timedelta(hours=i)
        records.append(SimpleNamespace(slot_at=slot,observed_at=slot,available_at=slot+timedelta(hours=2),cache_path=str(path),sha256='a'*64,b0_deg=0.,valid_fraction=1.,carrington_lon=0.))
    as_of=start+timedelta(hours=31)
    out=feature_frames(records,as_of)
    assert len(out)==2
    assert out[-1].features['aia_delta_24h_lat1_lon2']==pytest.approx(.3)
    assert out[-1].available_at<=as_of


def test_absent_optional_cache_does_not_break_other_forecast_inputs(tmp_path):
    now=datetime(2025,1,1,tzinfo=UTC)
    row=SimpleNamespace(slot_at=now,observed_at=now,available_at=now,cache_path=str(tmp_path/'missing.npz'))
    assert feature_frames([row],now)==[]


def test_owner_query_is_bounded_by_receipt_and_observation():
    from sqlalchemy.dialects import postgresql
    now=datetime(2025,1,1,tzinfo=UTC);session=AsyncMock()
    session.execute.return_value=Mock(scalars=lambda:Mock(all=lambda:[]))
    assert asyncio.run(load_aia_features(session,now))==[]
    q=session.execute.call_args.args[0].compile(dialect=postgresql.dialect())
    assert q.params['available_at_1']==now and q.params['observed_at_2']==now
    assert (now-q.params['observed_at_1']).days==40
