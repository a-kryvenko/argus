"""Offline perfect-future-ap diagnostic; same adapter and issue-time history as v2."""
import json
import time

import numpy as np
import pandas as pd

from prepare import ROOT, HERE, digest
from train_ap import OUT as PREVIOUS, read_omni
from run_three_way import BASE, check_baseline, common_rows
from run import paired_metrics
from ap_forecast import forecast_dtm_snapshot
from models import DTM2020

OUT = ROOT/'data/experiments/dtm2020_ap_oracle_v4'
MODELS = ('dtm2020', 'dtm2020_ap_forecast', 'dtm2020_ap_oracle')


def oracle_trajectory(forecasts, raw_ap):
    """Replace forecast values only, retaining its exact hourly/half-hourly clock."""
    result = forecasts[['issue_time', 'valid_time', 'lead_hours']].copy()
    result['ap_q50'] = raw_ap.reindex(pd.DatetimeIndex(result.valid_time).floor('h')).to_numpy()
    if not np.isfinite(result.ap_q50).all() or not result.ap_q50.between(0, 400).all():
        raise ValueError('Missing or invalid future observed ap; no imputation permitted')
    return result


def main():
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    check_baseline()
    provenance = json.loads((PREVIOUS/'results.json').read_text())
    for name, sha in provenance['outputs'].items():
        if digest(PREVIOUS/name) != sha:
            raise ValueError(f'Previous output changed: {name}')
    for name, sha in provenance['baseline_artifact_hashes'].items():
        if digest(BASE/name) != sha:
            raise ValueError(f'Baseline changed: {name}')
    raw, paths = read_omni()
    for path in paths:
        if digest(path) != provenance['training']['source_files'][str(path.relative_to(ROOT))]:
            raise ValueError('OMNI source changed')
    forecasts = pd.read_parquet(PREVIOUS/'ap_forecasts.parquet')
    oracle = oracle_trajectory(forecasts, raw.ap)
    oracle.to_parquet(OUT/'oracle_ap.parquet', index=False)
    groups = {t: f for t, f in oracle.groupby('issue_time')}
    predictions = common_rows(pd.read_parquet(PREVIOUS/'predictions.parquet'))
    measurements = pd.read_parquet(BASE/'measurements.parquet')
    fixed = {pd.Timestamp(r['issue_time']): r['dtm2020']
             for r in json.loads((BASE/'driver_snapshots.json').read_text())}
    model = DTM2020(BASE)
    cached, values, snapshots = {}, [], []
    for i, row in enumerate(predictions.itertuples(index=False)):
        if row.lead_hours == 0:
            values.append(row.dtm2020)
            continue
        if row.issue_time not in cached:
            cached[row.issue_time] = measurements.loc[measurements.metric.eq('ap')
                & measurements.available_at.le(row.issue_time)
                & measurements.observed_at.ge(row.issue_time-pd.Timedelta(hours=30))]
        drivers = forecast_dtm_snapshot(cached[row.issue_time], groups[row.issue_time],
            row.issue_time, row.valid_time, baseline=fixed[row.issue_time])
        assert drivers['f107'] == fixed[row.issue_time]['f107']
        assert drivers['f107_mean'] == fixed[row.issue_time]['f107_mean']
        day = row.valid_time.dayofyear+(row.valid_time-row.valid_time.floor('D')).total_seconds()/86400
        rho = model.evaluate(day, row.altitude_km, np.deg2rad(row.latitude_deg),
            np.deg2rad(row.longitude_deg), row.local_solar_time*np.pi/12,
            drivers['f107'], drivers['f107_mean'], drivers['kp'], drivers['kp_mean'])[0]
        values.append(float(rho))
        snapshots.append({'issue_time': row.issue_time, 'valid_time': row.valid_time, **drivers})
        if i % 1000 == 0:
            print(f'{i}/{len(predictions)} rows, {time.monotonic()-started:.0f}s', flush=True)
    predictions['dtm2020_ap_oracle'] = values
    if not np.isfinite(values).all() or not (np.array(values) > 0).all():
        raise ValueError('Invalid oracle density')
    predictions.to_parquet(OUT/'predictions.parquet', index=False)
    pd.DataFrame(snapshots).to_parquet(OUT/'driver_snapshots.parquet', index=False)
    scores = []
    for grouping in [('lead_hours',), ('lead_hours', 'activity'), ('lead_hours', 'window')]:
        for key, frame in predictions.groupby(list(grouping)):
            labels = dict(zip(grouping, key if isinstance(key, tuple) else (key,)))
            for name in MODELS:
                adapted = frame.copy()
                adapted['dtm2020'] = frame[name]
                metric = paired_metrics(adapted)[1]
                metric['model'] = name
                scores.append({'grouping': '+'.join(grouping), **labels, **metric})
    metrics = pd.DataFrame(scores)
    metrics.to_csv(OUT/'metrics.csv', index=False)
    predictions.groupby('lead_hours').size().rename('common_n').to_csv(OUT/'coverage.csv')
    metadata = {'experiment': OUT.name, 'elapsed_seconds': time.monotonic()-started,
        'models': MODELS, 'input_hashes': {str(p.relative_to(ROOT)): digest(p) for p in
            [PREVIOUS/'predictions.parquet', PREVIOUS/'ap_forecasts.parquet', PREVIOUS/'results.json', *paths]},
        'source_hashes': {p.name: digest(p) for p in HERE.glob('*.py')},
        'limitations': ['Diagnostic with deliberate future ap access; impossible as a live forecast.',
            'Only future forecast ap values replaced with observed 3h ap from hourly OMNI records; same forecast timestamps.',
            'Issue-time ap bridge, interpolation, 3h Kp lag and 24h Kp mean unchanged; unfinished current bin not revealed.',
            'F107 and its mean fixed; future satellite positions observed, same as v2.',
            'Common v2 rows only; no additional missing rows permitted.',
            'One satellite, same previously examined windows; density validation, not satellite drag-risk validation.',
            'Oracle is not a guaranteed error lower bound: model/input biases may compensate each other.']}
    (OUT/'results.json').write_text(json.dumps(metadata, indent=2)+'\n')
    report(metrics, metadata)
    print(metrics.loc[metrics.lead_hours.isin([24, 48]) & (metrics.grouping.eq('lead_hours') | metrics.activity.eq('storm')),
        ['lead_hours', 'activity', 'model', 'mape_pct', 'bias_pct', 'underestimate_gt_30pct']].to_string(index=False))


def report(metrics, metadata):
    from report import plt
    import html
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = ['Fixed ap', 'Forecast ap q50', 'Observed future ap (oracle)']
    for name, label in zip(MODELS, labels):
        for ax, storm in zip(axes, [False, True]):
            selected = metrics.loc[metrics.model.eq(name) &
                (metrics.activity.eq('storm') if storm else metrics.grouping.eq('lead_hours'))]
            ax.plot(selected.lead_hours, selected.mape_pct, 'o-', label=label)
    for ax, title in zip(axes, ['All observations', 'Storm observations: Kp >= 5']):
        ax.set(title=title, xlabel='Lead (hours)', ylabel='Density MAPE (%)')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(OUT/'comparison.png', dpi=180)
    plt.close(fig)
    table = metrics.copy()
    table['underestimate_gt_30_pct'] = table.underestimate_gt_30pct*100
    table['activity'] = table.activity.fillna('all')
    selected = table.loc[table.grouping.isin(['lead_hours', 'lead_hours+activity']),
        ['lead_hours', 'activity', 'model', 'n', 'mape_pct', 'bias_pct', 'underestimate_gt_30_pct']]
    limits = ''.join(f'<li>{html.escape(s)}</li>' for s in metadata['limitations'])
    (OUT/'report.html').write_text('''<!doctype html><meta charset="utf-8"><title>DTM: observed future ap</title>
<style>body{max-width:1100px;margin:40px auto;font:16px/1.5 system-ui}img{width:100%}td,th{padding:6px;border-bottom:1px solid #ddd}table{border-collapse:collapse;font-size:13px}</style>
<h1>DTM с фактическим будущим ap</h1><p>Диагностический эксперимент: прогноз ap заменён фактическими будущими значениями.
Солнечные драйверы, координаты, доступная на момент выпуска история, преобразование ap → Kp и временные настройки сохранены.
Это проверка вклада ошибок прогноза ap, а не доступная в работе модель и не гарантированный минимум ошибки.</p>
<img src="comparison.png" alt="MAPE всей выборки и бурь">'''+selected.to_html(index=False, float_format=lambda x:f'{x:.2f}')+
        '<h2>Ограничения</h2><ul>'+limits+'</ul><p>Повторение: .venv/bin/python -u scripts/training/thermosphere/run_oracle.py</p>')


if __name__ == '__main__':
    main()
