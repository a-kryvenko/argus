'use client';
import MetricValue from './MetricValue';
import type { ProductConfig } from '../../_config/products';
import { productApiPath } from '../../_config/products';
import { useResource } from '../../_utils/useResource';
import ResourceState from '../../_components/ResourceState';
import local from './metrics.module.css';
import styles from '../../_components/forecast.module.css';

type Verification = {
  start: string;
  end: string;
  groups: Array<{
    artifact: string;
    model_sha256: string;
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
  plasma_speed_quantile: 'v', plasma_speed_threshold: 'v', plasma_density_quantile: 'n',
  kp_threshold: 'kp', ap_quantile: 'ap', dst_quantile: 'dst',
  hmf_total_threshold: 'bt', hmf_southward_threshold: 'bs',
  f10_7_quantile: 'f10_7', s10_quantile: 's10', m10_quantile: 'm10', y10_quantile: 'y10',
};
function utc(value: string) { return `${new Date(value).toISOString().slice(0, 16).replace('T', ' ')} UTC`; }

export default function MonthlyAccuracy({ product }: { product: ProductConfig }) {
  const { data, error, retry } = useResource<Verification>(productApiPath(product, '/verification'));
  return <section className={local.monthly} aria-label="Last 30 days average accuracy">
    <h2>Last 30 days average accuracy</h2>
    <p>Published forecasts compared with hourly observations from Clio. Evaluated separately at each forecast lead time; lower MAE, RMSE and Brier scores are better.</p>
    {!data ? <ResourceState error={error} retry={retry} label="observed forecast accuracy" /> : <>
      <p>{utc(data.start)} – {utc(data.end)} · by forecast valid time</p>
      {!data.groups.length && <p role="status">No verified forecast data available for this period.</p>}
      {data.groups.map(group => {
        const variable = product.variables.find(item => item.key === variables[group.artifact]);
        return <div key={`${group.artifact}-${group.model_sha256}`} className={local.monthlyGroup}>
          <h3>{variable?.label ?? group.artifact} · {group.artifact.endsWith('_threshold') ? 'Threshold probabilities' : 'Quantile forecast'}</h3>
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
          <p>Updated {utc(group.evaluated_at)} · Model {group.model_sha256 === 'unknown' ? 'version unavailable' : group.model_sha256.slice(0, 12)}</p>
        </div>;
      })}
      <details><summary>How this average is calculated</summary><p>Scores are recomputed from individual forecast–observation pairs in the rolling 30-day window. Each row uses only forecasts made exactly that many hours before valid time; +24h does not include shorter leads. Missing observations and pending hours are excluded from scores. Model versions are evaluated separately. Available history may cover only part of the window.</p></details>
    </>}
  </section>;
}
