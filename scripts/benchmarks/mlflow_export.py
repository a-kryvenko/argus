"""Export saved evaluation reports to MLflow without recomputing metrics."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import digest


def export_report(report, *, experiment='forecast-benchmarks', tracking_uri=None, include_pairs=False):
    import mlflow

    report = Path(report)
    manifest = json.loads((report/'manifest.json').read_text())
    if (report/'status.json').exists():
        if json.loads((report/'status.json').read_text())['status'] != 'complete':
            raise ValueError('Report is incomplete')
    files = []
    for name, expected in manifest['artifacts'].items():
        path = (report/name).resolve()
        if not path.is_relative_to(report.resolve()):
            raise ValueError('Artifact outside report directory')
        if digest(path) != expected:
            raise ValueError(f'Changed report artifact: {name}')
        if include_pairs or path.suffix != '.parquet':
            files.append(path)
    if 'metrics.csv' not in manifest['artifacts']:
        raise ValueError('Metrics are not included in manifest')
    metrics = pd.read_csv(report/'metrics.csv')
    if metrics.empty or metrics.duplicated(['fold', 'target', 'baseline', 'lead_hours']).any():
        raise ValueError('Empty or duplicate metric rows')
    if mlflow.active_run() is not None:
        raise ValueError('Finish the active MLflow run before exporting')
    protocol = json.loads((report/'protocol.json').read_text()) if 'protocol.json' in manifest['artifacts'] else {}
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri or os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=report.name) as parent:
            parent_id = parent.info.run_id
            mlflow.set_tags({'report_sha256': digest(report/'manifest.json'), 'kind': 'forecast_evaluation'})
            mlflow.log_params({'report': report.name, 'lead_unit': 'hour', 'include_pairs': include_pairs})
            mlflow.log_artifact(str(report/'manifest.json'), artifact_path='report')
            for path in files:
                folder = path.relative_to(report.resolve()).parent
                mlflow.log_artifact(str(path), artifact_path=str(Path('report')/folder))
            for (fold, target, baseline), group in metrics.groupby(['fold', 'target', 'baseline']):
                with mlflow.start_run(run_name=f'{fold}-{target}-{baseline}', nested=True):
                    mlflow.log_params({'fold': fold, 'target': target, 'baseline': baseline,
                        'lead_unit': 'hour', **protocol.get('folds', {}).get(fold, {}),
                        **({'train_start': protocol['train_start']} if 'train_start' in protocol else {}),
                        **{k: v for k, v in manifest.get('metadata', {}).items()
                           if isinstance(v, (str, int, float, bool))}})
                    mlflow.set_tags({'report_sha256': digest(report/'manifest.json'),
                                     'evaluation_status': ','.join(sorted(group.status.unique()))})
                    mlflow.log_table(group.reset_index(drop=True), artifact_file='metrics.json')
                    for _, row in group.sort_values('lead_hours').iterrows():
                        values = {key: float(value) for key, value in row.items()
                                  if key != 'lead_hours' and isinstance(value, (int, float, np.number))
                                  and np.isfinite(value)}
                        mlflow.log_metrics(values, step=int(row.lead_hours))
        return parent_id
    finally:
        mlflow.set_tracking_uri(previous_uri)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--experiment', default='forecast-benchmarks')
    parser.add_argument('--tracking-uri')
    parser.add_argument('--include-pairs', action='store_true', help='Also upload observations and forecast pairs')
    args = parser.parse_args()
    print(export_report(args.report, experiment=args.experiment, tracking_uri=args.tracking_uri,
                        include_pairs=args.include_pairs))
