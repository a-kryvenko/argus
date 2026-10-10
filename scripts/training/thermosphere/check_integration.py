import hashlib
import json
from pathlib import Path
import tempfile
import time
import io

import numpy as np
import pandas as pd
from forecast_core.models.dtm2020 import DTM2020
from forecast_core.models.dtm2020_drivers import DTM2020Drivers, ObservedPersistenceDrivers
from forecast_core.models.AtmosphericDensityForecastService import AtmosphericDensityForecastService
from common.density_contract import parse_density_frame
from common.schemas.forecast_inputs import ForecastInputs, SourceMeasurement
from common.schemas.observation import Observation

root=Path(__file__).resolve().parents[3]
base=root/'data/experiments/dtm2020_jb2008_v1'
output=root/'data/experiments/dtm2020_integration_v5'
output.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix='dtm-smoke-') as directory:
    directory=Path(directory)
    for name, source in [('libdtm2020.so',base/'libdtm2020.so'),('DTM_2020_F107_Kp.dat',base/'upstream/DTM_2020_F107_Kp.dat')]:
        (directory/name).symlink_to(source)
    identity={'model':'DTM2020 Operational','abi':'argus-dtm-v1','sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}}
    (directory/'manifest.json').write_text(json.dumps(identity))
    model=DTM2020(directory)
    actual=[model.evaluate(180.,a,0.,0.,3.1415,DTM2020Drivers(f,f,3.,3.))[0] for a,f in [(300,80),(300,180),(800,80),(800,180)]]
    np.testing.assert_allclose(actual,np.array([.94719e-14,.29053e-13,.26600e-17,.39104e-16])*1000,rtol=5e-5,atol=0)
    # Compare new driver's physical values against every archived issue snapshot.
    records=pd.read_parquet(base/'measurements.parquet')
    saved=json.loads((base/'driver_snapshots.json').read_text())
    provider=ObservedPersistenceDrivers()
    maximum=0.
    for row in saved:
        issue=pd.Timestamp(row['issue_time'])
        drivers=provider.prepare(records.loc[records.observed_at.ge(issue-pd.Timedelta(days=90))],issue).iloc[0]
        for name in ('f107','f107_mean','kp','kp_mean'):
            np.testing.assert_allclose(drivers[name],row['dtm2020'][name],rtol=1e-12,atol=0)
            maximum=max(maximum,abs(drivers[name]-row['dtm2020'][name]))
    print('Archived snapshots match:',len(saved),'; max absolute driver difference:',maximum,flush=True)
    issue=pd.Timestamp('2025-11-12T12:00Z')
    known=records.loc[records.observed_at.ge(issue-pd.Timedelta(days=90)) & records.available_at.le(issue)]
    inputs=ForecastInputs(as_of=issue,read_at=issue,observations=Observation(points=[]),
        measurements=[SourceMeasurement(**row) for row in known.to_dict('records')])
    service=AtmosphericDensityForecastService(density_model='DTM2020',runtime_path=directory)
    started=time.monotonic()
    frame=service.forecast_snapshot(inputs,issue_time=issue)
    elapsed=time.monotonic()-started
    frame.to_csv(output/'density.csv',index=False)
    parsed=parse_density_frame(pd.read_csv(output/'density.csv'),now=issue)
    assert parsed.model=='DTM2020' and len(parsed.predictions)==49
    assert len(frame)==49*25*19
    assert frame[['f107','f107_mean','kp','kp_mean']].nunique().eq(1).all()
    # Compare actual evaluator at measured LST/positions to persisted experiment.
    obs=pd.read_parquet(base/'observations.parquet')
    predictions=pd.read_parquet(base/'predictions.parquet')
    refs=predictions.loc[predictions.issue_time.eq(issue)].merge(obs[['valid_time','local_solar_time']],on='valid_time')
    snapshot=provider.prepare(known,issue).iloc[0]
    drivers=DTM2020Drivers(**{name:float(snapshot[name]) for name in DTM2020Drivers.__dataclass_fields__})
    for row in refs.itertuples():
        day=row.valid_time.dayofyear+(row.valid_time-row.valid_time.floor('D')).total_seconds()/86400
        rho=model.evaluate(day,row.altitude_km,np.deg2rad(row.latitude_deg),np.deg2rad(row.longitude_deg),row.local_solar_time*np.pi/12,drivers)[0]
        np.testing.assert_allclose(rho,row.dtm2020,rtol=1e-12,atol=0)
    result={'benchmark_passed':True,'matched_archived_snapshots':len(saved),'matched_density_points':len(refs),
        'grid_cells':len(frame),'grid_seconds':elapsed,'model':parsed.model,'issue_time':issue.isoformat(),
        'runtime':identity,'deployed':False}
    (output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
