"""Paired diagnostic of the published demo; no fitting or parameter selection."""
import argparse
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path


def timestamp(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def save_csv(path, rows):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def errors(cases, key):
    residuals = [r[key] - r['observed'] for r in cases]
    return dict(n=len(cases), mae=sum(map(abs, residuals))/len(cases),
                rmse=math.sqrt(sum(e*e for e in residuals)/len(cases)),
                bias=sum(residuals)/len(cases))


def alerts(cases, key, threshold):
    tp = fp = fn = tn = 0
    for r in cases:
        predicted, observed = r[key] >= threshold, r['observed'] >= threshold
        tp += predicted and observed
        fp += predicted and not observed
        fn += not predicted and observed
        tn += not predicted and not observed
    return dict(n=len(cases), tp=tp, fp=fp, fn=fn, tn=tn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


def evaluate(bundle):
    onset = timestamp(bundle['event']['starts_at']).replace(minute=0, second=0, microsecond=0)
    event_start, event_end = onset-timedelta(hours=12), onset+timedelta(hours=72)
    observations = {timestamp(r['time']): r['values']['v'] for r in bundle['observations']}
    speed = []
    for release in bundle['releases']['solar-wind-speed']:
        issue = timestamp(release['issue_time'])
        cme = release.get('cme_speed')
        assert cme and cme['integrated_into_speed']
        assert all(timestamp(e['available_at']) <= issue for e in cme['events'])
        outlook = {timestamp(p['valid_time']): p for p in cme['points']}
        for p in release['predictions']:
            valid = timestamp(p['valid_time'])
            observed = observations.get(valid)
            if observed is None or not math.isfinite(observed):
                continue
            point = outlook[valid]
            q = p['variables']['v']['continuous']
            corrected, background = q['q50'], point['speed_background']
            assert abs(corrected-point['speed_integrated']) < 1e-8
            delta = corrected-background
            assert delta >= 0
            speed.append(dict(issue_time=issue, valid_time=valid, lead_hours=p['lead_hours'],
                              observed=observed, background=background, corrected=corrected,
                              delta=delta, q10=q['q10'], q90=q['q90']))
    subsets = dict(all=speed,
        event=[r for r in speed if event_start <= r['valid_time'] < event_end],
        pre_event=[r for r in speed if r['issue_time'] < onset and event_start <= r['valid_time'] < event_end],
        corrected_hours=[r for r in speed if r['delta'] > 1e-8])
    metrics, thresholds = [], []
    for scope, rows in subsets.items():
        for lead in (None, 6, 12, 24, 48):
            cases = [r for r in rows if lead is None or r['lead_hours'] == lead]
            if not cases:
                continue
            for model in ('background', 'corrected'):
                metrics.append(dict(scope=scope, lead_hours=lead or 'all', model=model, **errors(cases, model)))
                for threshold in (500, 600, 700):
                    thresholds.append(dict(scope=scope, lead_hours=lead or 'all', model=model,
                                           threshold_km_s=threshold, **alerts(cases, model, threshold)))
    paired = []
    for scope, rows in subsets.items():
        changed = [r for r in rows if r['delta'] > 1e-8]
        improvements = [abs(r['background']-r['observed'])-abs(r['corrected']-r['observed']) for r in changed]
        paired.append(dict(scope=scope, cases=len(rows), changed=len(changed),
            improved=sum(e > 1e-8 for e in improvements), worsened=sum(e < -1e-8 for e in improvements),
            tied=sum(abs(e) <= 1e-8 for e in improvements)))
    # Fixed issue examples share the same 48-hour event window and observation support.
    peaks = []
    for offset in (-24, -12, -6, -1):
        issue = onset+timedelta(hours=offset)
        rows = [r for r in speed if r['issue_time'] == issue and onset <= r['valid_time'] < onset+timedelta(hours=48)]
        if len(rows) != 48:
            continue
        observed_peak = max(rows, key=lambda r: r['observed'])
        for model in ('background', 'corrected'):
            peak = max(rows, key=lambda r: r[model])
            peaks.append(dict(issue_time=issue.isoformat(), model=model,
                observed_peak=observed_peak['observed'], predicted_peak=peak[model],
                peak_magnitude_error=peak[model]-observed_peak['observed'],
                peak_timing_error_hours=(peak['valid_time']-observed_peak['valid_time']).total_seconds()/3600))
    scenario = bundle['density_scenario']
    truth = {timestamp(p['time']): p['density_kg_m3'] for p in scenario['observations']}
    density = []
    for variant, releases in scenario['releases'].items():
        for lead in (6, 12, 24, 48):
            cases = []
            for release in releases:
                issue = timestamp(release['issue_time'])
                point = next(p for p in release['predictions'] if timestamp(p['time']) == issue+timedelta(hours=lead))
                observed = truth.get(timestamp(point['time']))
                if observed is not None and point['density_kg_m3'] is not None:
                    cases.append(dict(observed=observed, predicted=point['density_kg_m3']))
            for threshold in (2, 4):
                calculated = alerts(cases, 'predicted', threshold*scenario['reference']['density_kg_m3'])
                published = next(r for r in scenario['scores'] if r['variant'] == variant and r['lead_hours'] == lead and r['threshold_ratio'] == threshold)
                assert [calculated[k] for k in ('tp', 'fp', 'fn', 'tn')] == [published[k] for k in ('true_positive', 'false_positive', 'false_negative', 'true_negative')]
                density.append(dict(variant=variant, lead_hours=lead, threshold_ratio=threshold,
                    mape_pct=100*sum(abs(r['predicted']/r['observed']-1) for r in cases)/len(cases), **calculated))
    return speed, metrics, thresholds, peaks, density, paired


def plot(speed, output, onset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), layout='constrained')
    for ax, offset in zip(axes, (-24, -12, -1)):
        issue = onset+timedelta(hours=offset)
        rows = [r for r in speed if r['issue_time'] == issue and onset-timedelta(hours=12) <= r['valid_time'] < onset+timedelta(hours=48)]
        for key, label, color in [('observed', 'Observed', '#202020'), ('background', 'Background', '#2878b5'), ('corrected', 'Background + CME', '#c87810')]:
            ax.plot([r['valid_time'] for r in rows], [r[key] for r in rows], label=label, color=color)
        ax.axvline(onset, color='gray', linestyle=':', label='T0 hour')
        ax.set(title=f'Issue {issue:%Y-%m-%d %H:%M} UTC', ylabel='Speed, km/s')
        if not any(r['delta'] > 1e-8 for r in rows):
            ax.text(.02, .9, 'No available CME correction at this issue', transform=ax.transAxes)
        ax.grid(alpha=.2)
        ax.legend(loc='upper right', ncol=4)
    fig.suptitle('January 2026 demo: paired speed diagnostic, same observed hours')
    fig.savefig(output/'speed-comparison.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path('data/demo/current.json'))
    args = parser.parse_args()
    raw = args.bundle.read_bytes()
    bundle = json.loads(raw)
    output = Path('data/demo/evaluation')/bundle['version']
    output.mkdir(parents=True, exist_ok=True)
    speed, metrics, thresholds, peaks, density, paired = evaluate(bundle)
    for name, rows in [('speed-cases', speed), ('speed-errors', metrics), ('speed-thresholds', thresholds),
                       ('speed-peaks', peaks), ('density-alerts', density)]:
        save_csv(output/f'{name}.csv', rows)
    summary = dict(bundle_version=bundle['version'], bundle_sha256=hashlib.sha256(raw).hexdigest(),
        event_window='T0 hour -12 h inclusive to T0 hour +72 h exclusive',
        fitting=False, paired=paired, speed_metrics=metrics, speed_peaks=peaks, density_alerts=density,
        limitations=['Previously inspected single event; diagnostic, not blind validation.',
            'Forecast cases overlap and are correlated; no independent-sample significance claims.',
            'Hourly means label hour starts; T0 denotes G4 onset, not CME shock arrival.',
            'Speed threshold counts use q50, not calibrated event probabilities.',
            'Density uses future measured GRACE-C positions and empirical 2x/4x reference rules.'])
    onset = timestamp(bundle['event']['starts_at']).replace(minute=0, second=0, microsecond=0)
    scenario = bundle['density_scenario']
    episodes = []
    for threshold in (2, 4):
        candidates = [r for r in scenario['events'] if r['threshold_ratio'] == threshold
            and r['variant'] == 'jb2008_fixed' and onset <= timestamp(r['start']) < onset+timedelta(hours=48)]
        if candidates:
            longest = max(candidates, key=lambda r: r['sample_count'])
            episodes.extend(r for r in scenario['events'] if r['threshold_ratio'] == threshold
                            and r['start'] == longest['start'])
    summary['longest_storm_density_episodes'] = episodes
    summary['warning_stability'] = [dict(variant=v,
        **{key:sum(r[key] for r in scenario['journal'] if r['variant'] == v)
           for key in ('changed_samples', 'removed_warning_samples', 'overlapping_samples')})
        for v in scenario['variants']]
    (output/'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    plot(speed, output, onset)
    print(json.dumps(dict(output=str(output), paired=paired,
        speed=[r for r in metrics if r['lead_hours'] in ('all', 24)],
        peaks=peaks, density_24h=[r for r in density if r['lead_hours'] == 24]), indent=2))


if __name__ == '__main__':
    main()
