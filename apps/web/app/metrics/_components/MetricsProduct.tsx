'use client';
import { useId, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, ArrowUpRight, ChevronDown, Crosshair, RefreshCw } from 'lucide-react';
import { useResource } from '../../_utils/useResource';
import ResourceState from '../../_components/ResourceState';
import WorkspaceShell from '../../_components/WorkspaceShell';
import { productApiPath, type ProductConfig } from '../../_config/products';
import type { ForecastMetrics } from '../../_utils/api';
import MetricChart from './MetricChart';
import ReliabilityChart from './MetricReliability';
import { binaryRows, continuousKeys, leadHours, metricLabel, metricNumber, metricUnit, scoreLabels, scoreNotes, type ScoreKey } from '../_utils/transform';
import styles from '../../_components/forecast.module.css';
import local from './metrics.module.css';

export default function MetricsProduct({ product }: { product: ProductConfig }) {
  const { data, error, retry } = useResource<ForecastMetrics>(productApiPath(product, '/metrics'));
  return <WorkspaceShell section="Model performance" contentId="metrics-content" status={error ? 'Metrics unavailable' : data ? 'Model evaluation' : 'Loading metrics'} statusTone={error ? 'warning' : 'neutral'}>
    <main id="metrics-content" className={styles.page}>
      <Link href="/metrics" className={styles.backLink}><ArrowLeft size={12} aria-hidden="true" />Model performance</Link>
      <div className={styles.pageHeading}><div><div className={styles.eyebrow}>MODEL EVALUATION <span>/ {product.variables.map(variable => variable.key.toUpperCase()).join(' · ')}</span></div><h1>{product.title} metrics</h1><p>Forecast quality by lead hour. Explore errors, threshold scores and probability calibration.</p></div>
        <Link className={styles.headingLink} href={`/products/${product.slug}`}>Open forecast<ArrowUpRight size={13} aria-hidden="true" /></Link>
      </div>
      {!data ? <ResourceState error={error} retry={retry} label="metrics" /> : <MetricsBoard key={product.slug} product={product} data={data} retry={retry} />}
      <div className={styles.pageFooter}><span>Evaluation scores · not forecast values</span><a href={`/api/v1${productApiPath(product, '/metrics')}`}>Metrics data · JSON<ArrowUpRight size={12} aria-hidden="true" /></a></div>
    </main>
  </WorkspaceShell>;
}

function MetricsBoard({ product, data, retry }: { product: ProductConfig; data: ForecastMetrics; retry: () => void }) {
  const [variableKey, setVariableKey] = useState(product.variables[0].key);
  const [continuousKey, setContinuousKey] = useState('mae');
  const [score, setScore] = useState<ScoreKey>('brier_score');
  const [selectedHour, setSelectedHour] = useState<number>();
  const [horizon, setHorizon] = useState<number | null>(null);
  const [expanded, setExpanded] = useState(false);
  const id = useId();
  const variable = product.variables.find(item => item.key === variableKey)!;
  const metrics = data.variables[variableKey];
  const allHours = leadHours(metrics);
  const hours = allHours.filter(hour => horizon == null || hour <= horizon);
  const hour = selectedHour != null && hours.includes(selectedHour) ? selectedHour : hours[0];
  const keys = continuousKeys(metrics);
  const chartKeys = keys.filter(key => key !== 'n');
  const selectedMetric = chartKeys.includes(continuousKey) ? continuousKey : chartKeys[0];
  const continuous = metrics?.continuous?.by_lead_hour.find(row => row.lead_hours === hour)?.values;
  const labels = Object.fromEntries(metrics?.binary.map(series => [String(series.threshold), variable.thresholds.find(item => item.value === series.threshold)?.label ?? `≥ ${series.threshold} ${variable.unit}`]) ?? []);
  const binary = metrics ? binaryRows(metrics, score).filter(row => horizon == null || row.lead_hours <= horizon) : [];
  const cards = ['mae', 'rmse', 'coverage_80', 'n'].filter(key => keys.includes(key));
  const summaryKeys = cards.length ? cards : chartKeys.slice(0, 4);
  const summaryScores = Object.keys(scoreLabels).slice(0, 4) as ScoreKey[];
  const firstThreshold = metrics?.binary[0];
  const firstScores = firstThreshold?.by_lead_hour.find(row => row.lead_hours === hour);
  return <>
    <div className={styles.toolbar}>
      <div className={styles.variableTabs} role="group" aria-label="Metrics variable">{product.variables.map(item => <button key={item.key} aria-pressed={item.key === variableKey} onClick={() => setVariableKey(item.key)}><span>{item.key.toUpperCase()}</span>{item.label}</button>)}</div>
      <div className={styles.horizonControl}><span>LEAD HORIZON</span><div className={styles.periods} role="group" aria-label="Metrics horizon">{[24, 48, null].map(value => <button key={value ?? 'full'} aria-label={value == null ? 'Full evaluation horizon' : `First ${value} hours`} aria-pressed={horizon === value} onClick={() => setHorizon(value)}>{value == null ? 'Full' : `${value}h`}</button>)}</div></div>
    </div>
    <div className={local.context}><span>{hours.length ? `${hours.length} evaluated lead times · +${hours[0]}h to +${hours.at(-1)}h` : 'No evaluated lead times in this horizon'}</span><div className={styles.actions}><button onClick={retry}><RefreshCw size={12} aria-hidden="true" />Refresh</button></div></div>
    {!hours.length ? <p className={styles.notice} role="status">{variable.label} metrics are not available in this horizon.</p> : <>
      <section className={local.summary} aria-label="Selected lead metrics">
        {summaryKeys.length ? summaryKeys.map(key => <div key={key}><span>{metricLabel(key)}</span><strong>{metricNumber(continuous?.[key])}<small>{metricUnit(key, variable.unit)}</small></strong><span>Lead +{hour}h{key === 'coverage_80' ? ' · nominal 0.8' : ''}</span></div>) : summaryScores.map(key => <div key={key}><span>{scoreLabels[key]}</span><strong>{metricNumber(firstScores?.[key])}</strong><span>Lead +{hour}h · {firstThreshold ? labels[String(firstThreshold.threshold)] : 'Unavailable'}</span></div>)}
      </section>
      <div className={styles.boardGrid}>
        <div className={styles.plots}>
          {selectedMetric && <>
            <div className={local.selector}><label htmlFor={`${id}-continuous`}>Continuous metric</label><select id={`${id}-continuous`} value={selectedMetric} onChange={event => setContinuousKey(event.target.value)}>{chartKeys.map(key => <option value={key} key={key}>{metricLabel(key)}</option>)}</select></div>
            <MetricChart data={metrics.continuous!.by_lead_hour.filter(row => horizon == null || row.lead_hours <= horizon)} title={metricLabel(selectedMetric)} labels={{ [selectedMetric]: variable.label }} unit={metricUnit(selectedMetric, variable.unit)} selectedHour={hour} onSelectHour={setSelectedHour} note={selectedMetric === 'coverage_80' ? 'Observed coverage of the nominal 80% prediction interval.' : 'Evaluation of the continuous forecast.'} />
          </>}
          {metrics.binary.length > 0 && <>
            <div className={local.selector}><label htmlFor={`${id}-score`}>Threshold metric</label><select id={`${id}-score`} value={score} onChange={event => setScore(event.target.value as ScoreKey)}>{Object.entries(scoreLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></div>
            <MetricChart data={binary} title={scoreLabels[score]} labels={labels} selectedHour={hour} onSelectHour={setSelectedHour} note={scoreNotes[score]} />
            <ReliabilityChart data={metrics} labels={labels} hour={hour} />
          </>}
          <details className={styles.dataTable}><summary>Values by lead hour<span>{hours.length} evaluated times</span></summary><div className={styles.tableScroll}><table>
            <caption>Continuous values and {scoreLabels[score].toLowerCase()} by threshold. A dash means unavailable.</caption>
            <thead><tr><th scope="col">Lead hour</th>{keys.map(key => <th key={key} scope="col">{metricLabel(key)}{metricUnit(key, variable.unit) && ` · ${metricUnit(key, variable.unit)}`}</th>)}{Object.entries(labels).map(([key, label]) => <th scope="col" key={key}>{label}</th>)}</tr></thead>
            <tbody>{hours.map(lead => { const row = metrics.continuous?.by_lead_hour.find(item => item.lead_hours === lead); const scores = binary.find(item => item.lead_hours === lead); return <tr key={lead} data-selected={lead === hour}><th scope="row"><button aria-label={`Inspect lead ${lead} hours`} onClick={() => setSelectedHour(lead)}>+{lead}h</button></th>{keys.map(key => <td key={key}>{metricNumber(row?.values[key])}</td>)}{Object.keys(labels).map(key => <td key={key}>{metricNumber(scores?.values[key])}</td>)}</tr>; })}</tbody>
          </table></div></details>
        </div>
        <aside className={styles.inspector} aria-label="Metrics inspector" data-expanded={expanded}>
          <div className={styles.inspectorHeading}><Crosshair size={13} aria-hidden="true" /><span>METRICS INSPECTOR</span><button className={styles.inspectorToggle} aria-expanded={expanded} aria-controls={`${id}-inspector`} onClick={() => setExpanded(!expanded)}>Details<ChevronDown size={13} aria-hidden="true" /></button></div>
          <div className={styles.inspectorBody} id={`${id}-inspector`}>
            <label htmlFor={`${id}-lead`}>Evaluation lead hour</label><select id={`${id}-lead`} value={hour} onChange={event => setSelectedHour(Number(event.target.value))}>{hours.map(lead => <option value={lead} key={lead}>+{lead} hours</option>)}</select>
            <div className={styles.inspectorReading}><span>{variable.label}</span><p>+{hour}<small> hours</small></p><span>Forecast lead time</span></div>
            {keys.length > 0 && <dl className={local.values} aria-label="Continuous scores">{keys.map(key => <div key={key}><dt>{metricLabel(key)}<small>{metricUnit(key, variable.unit)}</small></dt><dd>{metricNumber(continuous?.[key])}</dd></div>)}</dl>}
            {metrics.binary.length > 0 && <div className={styles.thresholdValues}><h3>{scoreLabels[score]}</h3>{Object.entries(labels).map(([key, label]) => <div key={key}><p><span>{label}</span><strong>{metricNumber(binary.find(row => row.lead_hours === hour)?.values[key])}</strong></p></div>)}</div>}
            <dl className={styles.facts}><div><dt>Product</dt><dd>{product.title}</dd></div><div><dt>Evaluated lead range</dt><dd>+{allHours[0]}h – +{allHours.at(-1)}h</dd></div>{metrics.continuous && <div><dt>Evaluated quantiles</dt><dd>{metrics.continuous.quantiles.join(' · ')}</dd></div>}</dl>
            <details className={styles.methodDetails}><summary>Reading these metrics</summary><p>Scores describe model evaluation at each lead hour. Missing values are unavailable, not zero. Calibration uses the same selected lead hour.</p><p>Sample counts, where supplied, apply to the continuous evaluation. The metrics response does not identify the evaluation period.</p></details>
            <Link className={styles.inspectorLink} href={`/products/${product.slug}`}>Open forecast<ArrowUpRight size={12} aria-hidden="true" /></Link>
            <a className={styles.inspectorLink} href={`/api/v1${productApiPath(product, '/metrics')}`}>Metrics data · JSON<ArrowUpRight size={12} aria-hidden="true" /></a>
          </div>
        </aside>
      </div>
    </>}
  </>;
}
