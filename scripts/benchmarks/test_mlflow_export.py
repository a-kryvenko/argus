import json

import mlflow
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from scripts.benchmarks.benchmark import digest, write_json
from scripts.benchmarks.mlflow_export import export_report


def test_export_preserves_horizons_missing_metrics_and_files(tmp_path):
    report = tmp_path/'report'
    report.mkdir()
    pd.DataFrame([
        dict(fold='a', target='v', baseline='persistence', lead_hours=1, mae=2., n=5, status='ok'),
        dict(fold='a', target='v', baseline='persistence', lead_hours=24, mae=4., n=5, status='ok'),
        dict(fold='a', target='s10', baseline='climatology', lead_hours=24, mae=None, n=0, status='no_matched_data'),
    ]).to_csv(report/'metrics.csv', index=False)
    write_json(report/'protocol.json', {'train_start': '2011', 'folds': {'a': {'train_end': '2025'}}})
    pd.DataFrame({'value': [1]}).to_parquet(report/'observations.parquet')
    write_json(report/'manifest.json', {'artifacts': {p.name: digest(p) for p in report.iterdir()}})
    before = {p.name: digest(p) for p in report.iterdir()}
    uri = f'sqlite:///{tmp_path}/tracking.db'
    client = MlflowClient(tracking_uri=uri)
    experiment_id = client.create_experiment('test-export', artifact_location=(tmp_path/'artifacts').as_uri())
    old_uri = mlflow.get_tracking_uri()
    parent = export_report(report, experiment='test-export', tracking_uri=uri)
    assert mlflow.get_tracking_uri() == old_uri
    runs = client.search_runs([experiment_id], filter_string=f"tags.`mlflow.parentRunId` = '{parent}'")
    assert len(runs) == 2
    speed = next(r for r in runs if r.data.params['target'] == 'v')
    assert [(m.step, m.value) for m in client.get_metric_history(speed.info.run_id, 'mae')] == [(1, 2.), (24, 4.)]
    missing = next(r for r in runs if r.data.params['target'] == 's10')
    assert 'mae' not in missing.data.metrics
    assert missing.data.metrics['n'] == 0
    assert client.get_run(parent).info.status == 'FINISHED'
    assert 'report/observations.parquet' not in {a.path for a in client.list_artifacts(parent, 'report')}
    assert before == {p.name: digest(p) for p in report.iterdir()}
    (report/'metrics.csv').write_text('changed')
    with pytest.raises(ValueError, match='Changed report artifact'):
        export_report(report, experiment='test-export', tracking_uri=uri)
