'use client';
import { useState } from 'react';
import Link from 'next/link';
import { ArrowUpRight, ChevronDown, Clock3, Crosshair, RefreshCw } from 'lucide-react';
import type { Forecast, ForecastPoint } from '../_utils/api';
import { formatForecastTime, formatProbability, forecastWindow, probabilityData, quantileData } from '../_utils/forecast';
import { productApiPath, type ProductConfig } from '../_config/products';
import HeatMap from './HeatMap';
import WindChart from './WindChart';
import ResourceState from './ResourceState';
import styles from './forecast.module.css';

export type ForecastResource = { data: Forecast | null; error: string | null; retry: () => void };
export const forecastNumber = (value: number | null | undefined) => value == null ? '—' : value.toLocaleString('en-US', { maximumFractionDigits: 3 });
function probability(point: ForecastPoint | undefined, key: string, threshold: number) {
  return point?.variables[key]?.binary.find(item => item.threshold === threshold && (!item.operator || item.operator === 'gte'))?.probability;
}

export default function ForecastBoard({ product, resource, overview = false }: {
  product: ProductConfig; resource: ForecastResource; overview?: boolean;
}) {
  const { data: forecast, error, retry } = resource;
  const [variableKey, setVariableKey] = useState(product.variables[0].key);
  const [horizon, setHorizon] = useState<number | null>(null);
  const [selectedTime, setSelectedTime] = useState<string>();
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const variable = product.variables.find(item => item.key === variableKey) ?? product.variables[0];
  const view = forecast ? forecastWindow(forecast, horizon) : null;
  const point = view?.predictions.find(item => item.valid_time === selectedTime) ?? view?.predictions[0];
  const values = point?.variables[variable.key];
  const selectTime = (time: string) => { setSelectedTime(time); setInspectorOpen(true); };
  const primaryThreshold = variable.thresholds[0];
  const lastPoint = view?.predictions.at(-1);
  return <section aria-label={`${product.title} forecast workspace`}>
    <div className={styles.boardHeading}>
      <div><h2>{overview ? product.title : 'Forecast timeline'}</h2><p>{overview ? product.description : 'Select a variable and forecast time to inspect its values.'}</p></div>
      <div className={styles.actions}>
        {overview && <Link href={`/products/${product.slug}`}>Open product<ArrowUpRight size={13} aria-hidden="true" /></Link>}
        <button onClick={retry} disabled={!forecast && !error} aria-label={`Refresh ${product.title}`}><RefreshCw size={13} aria-hidden="true" />Refresh</button>
      </div>
    </div>
    <div className={styles.toolbar}>
      <div className={styles.variableTabs} role="group" aria-label="Forecast variable">
        {product.variables.map(item => <button key={item.key} aria-pressed={variable.key === item.key} onClick={() => { setVariableKey(item.key); setInspectorOpen(true); }}>
          <span>{item.key.toUpperCase()}</span>{item.label}
        </button>)}
      </div>
      <div className={styles.horizonControl}><span>LEAD HORIZON</span><div className={styles.periods} role="group" aria-label="Forecast horizon">
        {[24, 48].map(hours => <button key={hours} aria-label={`First ${hours} hours`} aria-pressed={horizon === hours} disabled={!forecast || forecast.horizon_hours < hours} onClick={() => setHorizon(hours)}>{hours}h</button>)}
        <button aria-label="Full forecast" aria-pressed={horizon === null} onClick={() => setHorizon(null)}>Full{forecast && ` · ${forecast.horizon_hours}h`}</button>
      </div></div>
    </div>
    {forecast && <div className={styles.releaseStrip}>
      <span><Clock3 size={12} aria-hidden="true" />ISSUED <time dateTime={forecast.issue_time}>{formatForecastTime(forecast.issue_time)}</time></span>
      <span>VISIBLE VALID TIMES <span>{point && view?.predictions[0] ? `${formatForecastTime(view.predictions[0].valid_time)} — ${formatForecastTime(lastPoint!.valid_time)}` : 'None in this horizon'}</span></span>
    </div>}
    <div className={styles.boardGrid}>
      <div className={styles.plots}>
        {!forecast && <ResourceState error={error} retry={retry} label={`${product.title.toLowerCase()} forecast`} />}
        {forecast && !forecast.available_variables.includes(variable.key) && <p className={styles.notice} role="status">{variable.label} is not available in this release.</p>}
        {view && <>
          {variable.quantile && <WindChart title={variable.label} unit={variable.unit} data={quantileData(view, variable.key)} selectedTime={point?.valid_time} onSelectTime={selectTime} />}
          {variable.thresholds.length > 0 && <HeatMap title={`${variable.label} threshold probability`} yLabels={variable.thresholds.map(item => item.label)}
            data={probabilityData(view, variable.key, variable.thresholds.map(item => item.value))} times={view.predictions.map(item => item.valid_time)} selectedTime={point?.valid_time} onSelectTime={selectTime} />}
          <details className={styles.dataTable}>
            <summary>Hourly forecast values <span>{view.predictions.length} forecast times · UTC</span></summary>
            <div className={styles.tableScroll} tabIndex={0} role="region" aria-label="Hourly forecast values, scroll horizontally">
              <table><caption>{variable.label} · {variable.unit}. Probabilities refer to meeting or exceeding a threshold.</caption>
                <thead><tr><th>Valid time · UTC</th><th>Lead</th>{variable.quantile && <><th>q10</th><th>q50</th><th>q90</th></>}{variable.thresholds.map(item => <th key={item.value}>{item.label}</th>)}</tr></thead>
                <tbody>{view.predictions.map(item => <tr key={item.valid_time} data-selected={point?.valid_time === item.valid_time}>
                  <th scope="row"><button aria-label={`Inspect forecast at ${formatForecastTime(item.valid_time)}`} onClick={() => selectTime(item.valid_time)}>{item.valid_time.slice(0, 16).replace('T', ' ')}</button></th>
                  <td>+{item.lead_hours}h</td>{variable.quantile && <>{(['q10', 'q50', 'q90'] as const).map(q => <td key={q}>{forecastNumber(item.variables[variable.key]?.continuous?.[q])}</td>)}</>}
                  {variable.thresholds.map(threshold => <td key={threshold.value}>{formatProbability(probability(item, variable.key, threshold.value))}</td>)}
                </tr>)}</tbody>
              </table>
            </div>
          </details>
        </>}
        <p className={styles.footerNote}>Forecast times are relative to the stated issue time. Products and variables can have different horizons.</p>
      </div>
      <aside className={styles.inspector} aria-label="Forecast inspector" data-expanded={inspectorOpen}>
        <div className={styles.inspectorHeading}><Crosshair size={14} aria-hidden="true" /><span>FORECAST INSPECTOR</span>
          <button className={styles.inspectorToggle} onClick={() => setInspectorOpen(open => !open)} aria-expanded={inspectorOpen} aria-controls={`forecast-inspector-${product.slug}`}>Details<ChevronDown size={14} aria-hidden="true" /></button>
        </div>
        <div className={styles.inspectorBody} id={`forecast-inspector-${product.slug}`}>
          <label htmlFor={`forecast-time-${product.slug}`}>Forecast time · UTC</label>
          <select id={`forecast-time-${product.slug}`} value={point?.valid_time ?? ''} disabled={!view?.predictions.length} onChange={event => selectTime(event.target.value)}>
            {!view?.predictions.length && <option value="">No forecast times</option>}
            {view?.predictions.map(item => <option key={item.valid_time} value={item.valid_time}>+{item.lead_hours}h · {item.valid_time.slice(5, 16).replace('T', ' ')}</option>)}
          </select>
          <div className={styles.inspectorReading}>
            <span>{variable.quantile ? 'MEDIAN · q50' : primaryThreshold?.label ?? variable.label}</span>
            <p>{variable.quantile ? forecastNumber(values?.continuous?.q50) : formatProbability(primaryThreshold ? probability(point, variable.key, primaryThreshold.value) : null)} <small>{variable.quantile ? variable.unit : ''}</small></p>
            <span>{point ? `Lead +${point.lead_hours} hours` : 'Awaiting forecast'}</span>
          </div>
          {point && !values && <p className={styles.notice}>No values for {variable.label} at this forecast time.</p>}
          {variable.quantile && <div className={styles.quantileValues}><div><span>q10</span><strong>{forecastNumber(values?.continuous?.q10)}</strong></div><div><span>q90</span><strong>{forecastNumber(values?.continuous?.q90)}</strong></div></div>}
          {variable.thresholds.length > 0 && <div className={styles.thresholdValues}><h3>Threshold probabilities</h3>{variable.thresholds.map(threshold => {
            const value = probability(point, variable.key, threshold.value);
            return <div key={threshold.value}><p><span>{threshold.label}</span><strong>{formatProbability(value)}</strong></p><div className={styles.probabilityTrack} aria-hidden="true"><span style={{ width: `${value == null ? 0 : value * 100}%` }} /></div></div>;
          })}</div>}
          <dl className={styles.facts}>
            <div><dt>Product</dt><dd>{product.title}</dd></div>
            <div><dt>Issued · UTC</dt><dd>{forecast ? formatForecastTime(forecast.issue_time) : '—'}</dd></div>
            <div><dt>Valid · UTC</dt><dd>{point ? formatForecastTime(point.valid_time) : '—'}</dd></div>
            <div><dt>Release horizon</dt><dd>{forecast ? `${forecast.horizon_hours} hours` : '—'}</dd></div>
          </dl>
          <details className={styles.methodDetails}><summary>Reading this forecast</summary>
            {variable.quantile && <p>q50 is the median. q10–q90 is the model’s central 80% prediction interval; actual coverage depends on calibration. It is not a guaranteed minimum and maximum.</p>}
            {variable.thresholds.length > 0 && <p>Each probability describes whether a value meets or exceeds a threshold at one forecast time. It is not the probability of an event occurring at any time during the full horizon or an impact score.</p>}
            <p>A dash or an empty cell means unavailable, not zero. Selecting a horizon does not change the model’s issue time.</p>
          </details>
          <Link className={styles.inspectorLink} href={`/metrics/${product.slug}`}>Model performance<ArrowUpRight size={13} aria-hidden="true" /></Link>
          <a className={styles.inspectorLink} href={`/api/v1${productApiPath(product)}`}>Forecast data · JSON<ArrowUpRight size={13} aria-hidden="true" /></a>
        </div>
      </aside>
    </div>
  </section>;
}
