'use client';
import { useId, useState, type KeyboardEvent } from 'react';
import MetricValue from './MetricValue';
import type { ProductConfig } from '../../../forecast/products';
import { productApiPath } from '../../../forecast/products';
import { useResource } from '../../../hooks/useResource';
import ResourceState from '../../../components/ResourceState';
import local from './metrics.module.css';
import styles from '../../../components/forecast.module.css';

type Verification = {
  start: string;
  end: string;
  groups: Array<{
    artifact: string;
    model?: string | null;
    model_sha256?: string | null;
    last_issue_time?: string | null;
    evaluated_at: string;
    releases: number;
    counts: Record<string, number>;
    by_lead_hour: Array<{
      lead_hours: number;
      counts: Record<string, number>;
      continuous: Record<string, number> | null;
      binary: Record<string, Record<string, number | null>>;
    }>;
  }>;
};
const variables: Record<string, string> = {
  plasma_speed_quantile: 'v', plasma_speed_threshold: 'v', plasma_density_quantile: 'n', plasma_temperature_quantile: 't',
  kp_threshold: 'kp', ap_quantile: 'ap', dst_quantile: 'dst',
  hmf_total_threshold: 'bt', hmf_southward_threshold: 'bs',
  f10_7_quantile: 'f10_7', s10_quantile: 's10', m10_quantile: 'm10', y10_quantile: 'y10',
};
function utc(value: string) { return `${new Date(value).toISOString().slice(0, 16).replace('T', ' ')} UTC`; }

export default function MonthlyAccuracy({ product }: { product: ProductConfig }) {
  const { data, error, retry } = useResource<Verification>(productApiPath(product, '/verification'));
  return <section className={local.monthly} aria-label="Last 30 days average accuracy">
    <h2>Last 30 days average accuracy</h2>
    <p>Published forecasts compared with hourly observations from Clio, separately by model version and forecast lead time. Lower MAE, RMSE and Brier scores are better.</p>
    {!data ? <ResourceState error={error} retry={retry} label="observed forecast accuracy" /> : <>
      <p>{utc(data.start)} – {utc(data.end)} · by forecast valid time</p>
      {!data.groups.length && <p role="status">No verified forecast data available for this period.</p>}
      <ModelTabs key={product.slug} product={product} groups={data.groups} />
      <details><summary>How this average is calculated</summary><p>Scores are recomputed from individual forecast–observation pairs in the rolling 30-day window, separately for each model name and SHA-256. Each row uses only forecasts made exactly that many hours before valid time; +24h does not include shorter leads. Missing observations and pending hours are excluded from scores. Records without version metadata are shown separately. Available history may cover only part of the window; versions may cover different dates.</p></details>
    </>}
  </section>;
}

function ModelTabs({ product, groups }: { product: ProductConfig; groups: Verification['groups'] }) {
  const id = useId();
  const [selected, setSelected] = useState<string>();
  const models = new Map<string, { key: string; name: string; latest: number; groups: Verification['groups'] }>();
  for (const group of groups) {
    const key = JSON.stringify([group.model ?? null, group.model_sha256 ?? null]);
    const version = models.get(key) ?? { key, name: group.model ?? 'Unknown model', latest: 0, groups: [] };
    version.latest = Math.max(version.latest, group.last_issue_time ? Date.parse(group.last_issue_time) : 0);
    version.groups.push(group);
    models.set(key, version);
  }
  const versions = [...models.values()].sort((a, b) => b.latest - a.latest || a.key.localeCompare(b.key));
  const variableOrder = new Map(product.variables.map((variable, index) => [variable.key, index]));
  for (const version of versions) {
    version.groups.sort((a, b) => Number(a.artifact.endsWith('_threshold')) - Number(b.artifact.endsWith('_threshold'))
      || (variableOrder.get(variables[a.artifact]) ?? variableOrder.size) - (variableOrder.get(variables[b.artifact]) ?? variableOrder.size)
      || a.artifact.localeCompare(b.artifact));
  }
  const active = versions.some(version => version.key === selected) ? selected : versions[0]?.key;
  function navigate(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const next = event.key === 'ArrowRight' ? (index + 1) % versions.length
      : event.key === 'ArrowLeft' ? (index + versions.length - 1) % versions.length
      : event.key === 'Home' ? 0 : event.key === 'End' ? versions.length - 1 : null;
    if (next == null) return;
    event.preventDefault();
    setSelected(versions[next].key);
    document.getElementById(`${id}-tab-${next}`)?.focus();
  }
  if (!versions.length) return null;
  return <>
    <div role="tablist" aria-label="Model versions" className={local.modelTabs}>
      {versions.map((version, index) => <button key={version.key} role="tab" type="button"
        id={`${id}-tab-${index}`} aria-controls={`${id}-panel-${index}`} aria-selected={active === version.key}
        tabIndex={active === version.key ? 0 : -1} onClick={() => setSelected(version.key)}
        onKeyDown={event => navigate(event, index)}>{version.name}</button>)}
    </div>
    {versions.map((version, index) => <div key={version.key} role="tabpanel" tabIndex={0}
      id={`${id}-panel-${index}`} aria-labelledby={`${id}-tab-${index}`} hidden={active !== version.key}>
      {version.groups.map(group => {
        const variable = product.variables.find(item => item.key === variables[group.artifact]);
        return <div key={JSON.stringify([group.artifact, group.model, group.model_sha256])} className={local.monthlyGroup}>
          <h3>{variable?.label ?? group.artifact} · {group.artifact.endsWith('_threshold') ? 'Threshold probabilities' : 'Quantile forecast'}</h3>
          <p>Model: {group.model ?? 'Unknown model'} · {group.model_sha256 ? <span title={group.model_sha256}>SHA-256: {group.model_sha256.slice(0, 12)}</span> : 'Version unavailable'}</p>
          <p>{group.counts.verified.toLocaleString('en-US')} verified pairs · {group.releases} releases</p>
          {!group.counts.verified && <p role="status">No matched observations available yet.</p>}
          <div className={styles.tableScroll}>
            <table className={local.monthlyTable}>
              <caption>Accuracy by forecast horizon · a dash means no verified data</caption>
              <thead><tr><th scope="col">Horizon</th><th scope="col">Verified</th>
                {group.artifact.endsWith('_threshold') ? variable?.thresholds.map(({ value }) => <th scope="col" key={value}>Brier score · ≥ {value} {variable.unit}</th>) : <><th scope="col">MAE · {variable?.unit}</th><th scope="col">RMSE · {variable?.unit}</th></>}
              </tr></thead>
              <tbody>{[3, 6, 12, 24, 48, 96].map(lead => {
                const point = group.by_lead_hour.find(item => item.lead_hours === lead);
                return <tr key={lead}><th scope="row">+{lead}h</th>
                  <td>{point ? point.counts.verified.toLocaleString('en-US') : '—'}</td>
                  {group.artifact.endsWith('_threshold') ? variable?.thresholds.map(({ value }) => <td key={value}><MetricValue value={point?.binary[String(value)]?.brier} /></td>) : <><td><MetricValue value={point?.continuous?.mae} /></td><td><MetricValue value={point?.continuous?.rmse} /></td></>}
                </tr>;
              })}</tbody>
            </table>
          </div>
          <p>Updated {utc(group.evaluated_at)}</p>
        </div>;
      })}
    </div>)}
  </>;
}
