"""Score actual published releases; never regenerate historical forecasts."""
import hashlib
import io
import json
import os
from datetime import UTC, datetime, timedelta
from threading import Lock
from time import monotonic

import httpx
import numpy as np
import pandas as pd

from forecast.evaluation import TARGETS, match_observations, score, validate_predictions
from common.schemas.forecast_release import PRODUCT_ARTIFACTS
from argus_prophet.db.session import connect, transaction
from argus_prophet.services.releases.publication import read_release

from argus_prophet.services.generation.products import VERIFIED_PRODUCTS as PRODUCTS
from argus_prophet.scheduling.execution import before_dispatch


MONTHLY_CACHE_SECONDS = 300
_monthly_cache = {}
_monthly_locks = {product: Lock() for product in PRODUCT_ARTIFACTS}


def cached_monthly_accuracy(product):
    """Reuse a rolling summary for five minutes; coalesce concurrent reads."""
    if product not in _monthly_locks:
        raise ValueError('Unknown forecast product')
    with _monthly_locks[product]:
        cached = _monthly_cache.get(product)
        if cached is not None and monotonic() < cached[0]:
            return cached[1]
        result = monthly_accuracy(product)
        _monthly_cache[product] = (monotonic() + MONTHLY_CACHE_SECONDS, result)
        return result


def targets(frame, start, end):
    """Prepare verification targets from raw observations, without filling gaps."""
    data = frame[frame.metric.isin(['v', 'n', 't', 'kp', 'ap', 'dst', 'bx', 'by', 'bz'])].copy()
    data['observed_at'] = pd.to_datetime(data.observed_at, utc=True)
    # browse includes its end boundary; verification uses [start, end).
    data = data[(data.observed_at >= start) & (data.observed_at < end)]
    if data.empty:
        return []
    wide = data.pivot(index='observed_at', columns='metric', values='value')
    if {'bx', 'by', 'bz'} <= set(wide):
        wide['bt'] = np.sqrt(wide[['bx', 'by', 'bz']].pow(2).sum(axis=1, min_count=3))
    if 'bz' in wide:
        wide['bs'] = (-wide.bz).clip(lower=0)
    rows = []
    for metric in ('v', 'n', 't', 'kp', 'ap', 'dst', 'bt', 'bs'):
        if metric not in wide:
            continue
        series = wide[metric].replace([np.inf, -np.inf], np.nan).dropna()
        for hour, values in series.groupby(series.index.floor('h')):
            rows.append({'valid_time': hour.isoformat(), 'metric': metric,
                         'value': float(values.mean()), 'sample_count': len(values)})
    return rows


def check_progress(deadline):
    before_dispatch()
    if deadline is not None and monotonic() >= deadline:
        raise TimeoutError('Verification exceeded its configured time budget')


def read_targets(start, end, *, deadline=None):
    url, token = os.getenv('OBSERVATIONS_URL'), os.getenv('OBSERVATIONS_SERVICE_TOKEN')
    if not url or not token:
        raise RuntimeError('OBSERVATIONS_URL and OBSERVATIONS_SERVICE_TOKEN are required')
    rows, total = [], None
    with httpx.Client(timeout=httpx.Timeout(180, connect=10), trust_env=False) as client:
        for page in range(1, 100_001):
            check_progress(deadline)
            response = client.get(url.rstrip('/') + '/internal/v1/observations/browse',
                                  params={'kind': 'raw', 'start': start.isoformat(), 'end': end.isoformat(),
                                          'order': 'asc', 'page': page, 'page_size': 200},
                                  headers={'Authorization': f'Bearer {token}'},
                                  timeout=httpx.Timeout(min(180, max(.1, deadline-monotonic())) if deadline else 180,
                                                        connect=10))
            response.raise_for_status()
            payload = response.json()
            if not payload['success']:
                raise ValueError('Observation read failed')
            data = payload['data']
            if total is not None and data['total'] != total:
                raise ValueError('Observations changed during pagination; repeat verification')
            total = data['total']
            rows.extend(data['items'])
            if len(rows) >= total:
                break
            if not data['items']:
                raise ValueError('Incomplete observation response')
        else:
            raise ValueError('Observation range exceeds browse pagination limit')
    frame = pd.DataFrame(rows, columns=['observed_at', 'metric', 'value'])
    return {'source': 'clio-measurement-hourly-mean-v1', 'status': 'provisional',
            'points': targets(frame, start, end)}


def verify(selection='solar-wind-speed', *, writer, days=7, now=None, deadline=None):
    from psycopg.types.json import Jsonb
    if selection not in (*PRODUCTS, 'all') or not 1 <= days <= 25:
        raise ValueError('Select a supported product and 1..25 days')
    now = now or datetime.now(UTC)
    if now.tzinfo is None:
        raise ValueError('now must include timezone')
    products = PRODUCTS if selection == 'all' else (selection,)
    start = (now - timedelta(days=days)).replace(minute=0, second=0, microsecond=0)
    end = now.replace(minute=0, second=0, microsecond=0)
    with connect() as conn:
        releases = conn.execute('''SELECT product,id FROM prophet.forecast_release
            WHERE product=ANY(%s) AND issue_time >= %s AND issue_time <= %s
            ORDER BY issue_time,id''', (list(products), start, now)).fetchall()
    if not releases:
        return {'status': 'no_releases', 'releases': 0, 'from': start.isoformat(), 'products': list(products)}
    observed = read_targets(start, end) if deadline is None else read_targets(start, end, deadline=deadline)
    truth = pd.DataFrame(observed['points'], columns=['valid_time', 'metric', 'value', 'sample_count'])
    summary = {'status': 'ok', 'releases': len(releases), 'from': start.isoformat(),
               'source': observed['source'], 'products': {}, 'evaluated_at': now.isoformat()}
    for product, release_id in releases:
        check_progress(deadline)
        release = read_release(product, release_id)
        for artifact in release.artifacts:
            check_progress(deadline)
            if artifact.name not in TARGETS:
                raise ValueError('No observed target protocol for ' + artifact.name)
            prediction = validate_predictions(pd.read_csv(io.StringIO(artifact.csv_text)), artifact.name)
            subset = truth[truth.metric.eq(TARGETS[artifact.name])]
            matched = match_observations(prediction, subset, as_of=now)
            # Exact target values (including gaps) are retained for replay and revisions.
            records = json.loads(matched.to_json(orient='records', date_format='iso'))
            evidence = json.dumps(records, sort_keys=True, allow_nan=False).encode()
            report = {'protocol': 'published-hourly-v1', 'release_id': str(release_id),
                      'source': observed['source'], 'observation_status': observed['status'],
                      'forecast_sha256': artifact.sha256, 'model_info': artifact.model_info,
                      'evidence_sha256': hashlib.sha256(evidence).hexdigest(),
                      **score(matched, artifact.name), 'pairs': records}
            with transaction(writer) as conn:
                conn.execute('''INSERT INTO prophet.forecast_verification
                    (release_id,artifact,evaluated_at,report) VALUES (%s,%s,%s,%s)
                    ON CONFLICT (release_id,artifact) DO UPDATE
                    SET evaluated_at=EXCLUDED.evaluated_at,report=EXCLUDED.report''',
                    (release_id, artifact.name, now, Jsonb(report)))
            totals = summary['products'].setdefault(product, {}).setdefault(
                artifact.name, dict.fromkeys(('total', 'verified', 'missing', 'pending'), 0))
            for key, value in report['counts'].items():
                totals[key] += value
    return summary


def verification_report(product, *, days=7):
    """Separate model hashes: an operational change must not blend model scores."""
    from psycopg.rows import dict_row
    if product not in (*PRODUCTS, 'all') or not 1 <= days <= 25:
        raise ValueError('Select a supported product and 1..25 days')
    products = PRODUCTS if product == 'all' else (product,)
    with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('''SELECT r.product,v.artifact,v.evaluated_at,v.report
            FROM prophet.forecast_verification v JOIN prophet.forecast_release r ON r.id=v.release_id
            WHERE r.product=ANY(%s) AND r.issue_time >= %s ORDER BY r.issue_time,r.id''',
            (list(products), datetime.now(UTC) - timedelta(days=days)))
        rows = cursor.fetchall()
    groups = {}
    for row in rows:
        info = row['report']['model_info']
        key = (row['product'], row['artifact'], info.get('sha256', 'unknown'))
        groups.setdefault(key, []).append(row)
    result = []
    for (name, artifact, digest), group in groups.items():
        frame = pd.concat([pd.DataFrame(row['report']['pairs']) for row in group], ignore_index=True)
        result.append({'product': name, 'artifact': artifact, 'model_sha256': digest,
                       'releases': len(group), 'last_evaluated_at': max(row['evaluated_at'] for row in group),
                       **score(frame, artifact)})
    return {'protocol': 'published-hourly-v1', 'days': days, 'groups': result}


def monthly_accuracy(product, *, now=None):
    """Pool matched pairs across all models, over valid times in the trailing 30 days."""
    from psycopg.rows import dict_row
    from common.schemas.forecast_verification import ForecastVerification, VerificationGroup, VerificationLead
    if product not in PRODUCT_ARTIFACTS:
        raise ValueError('Unknown forecast product')
    now = now or datetime.now(UTC)
    start = now - timedelta(days=30)
    with connect() as conn, conn.cursor(row_factory=dict_row) as cursor:
        cursor.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        cursor.execute('SET LOCAL statement_timeout=10000')
        # Include releases before the window whose forecast horizons enter it.
        cursor.execute("""SELECT v.artifact,v.evaluated_at,v.report
            FROM prophet.forecast_verification v JOIN prophet.forecast_release r ON r.id=v.release_id
            WHERE r.product=%s AND r.issue_time >= %s AND r.issue_time <= %s
            ORDER BY r.issue_time,r.id""", (product, start-timedelta(hours=96), now))
        rows = cursor.fetchall()
    groups = {}
    for row in rows:
        groups.setdefault(row['artifact'], []).append(row)
    result = []
    for artifact, entries in groups.items():
        # Build and filter one frame per artifact instead of one per release.
        pairs = [pair for entry in entries for pair in entry['report']['pairs']]
        if not pairs:
            continue
        frame = pd.DataFrame(pairs)
        frame['_release'] = [index for index, entry in enumerate(entries)
                             for _ in entry['report']['pairs']]
        times = pd.to_datetime(frame.valid_time, utc=True)
        frame = frame[(times >= start) & (times < now)].copy()
        if frame.empty:
            continue
        included = frame['_release'].unique()
        scored = score(frame, artifact)
        leads = {}
        for lead, part in frame.groupby('lead_hours', sort=True):
            counts = {state: int(part.state.eq(state).sum())
                      for state in ('verified', 'missing', 'pending')}
            counts['total'] = len(part)
            leads[int(lead)] = VerificationLead(lead_hours=int(lead), counts=counts)
        for table, records in scored['tables'].items():
            for record in records:
                values = {key: value for key, value in record.items()
                          if key not in ('lead_hours', 'n', 'scheduled', 'reliability')}
                point = leads[record['lead_hours']]
                if table == 'regression.csv':
                    point.continuous = values
                else:
                    point.binary[table.removeprefix('threshold_').removesuffix('.csv')] = values
        result.append(VerificationGroup(artifact=artifact,
            releases=len(included), evaluated_at=max(entries[index]['evaluated_at'] for index in included),
            counts=scored['counts'], by_lead_hour=list(leads.values())))
    return ForecastVerification(product=product, start=start, end=now, groups=result)
