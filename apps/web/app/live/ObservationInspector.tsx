'use client';
import { ArrowUpRight, ChevronDown, Crosshair } from 'lucide-react';
import { definition, isWind, measurement, metricColors, number, selections, stamp, type Selection } from './observations';
import type { LiveSnapshot, QueryResult } from './useLiveQuery';
import type { History } from './solarWind';
import type { IndexHistory } from './geomagnetic';
import HistoryCoverage from './HistoryCoverage';
import styles from './page.module.css';

const explanations: Record<Selection, string> = {
  v: 'Bulk solar wind speed measured upstream of Earth at L1.',
  n: 'Proton number density in the solar wind at L1.',
  bz: 'North–south magnetic field in GSM coordinates. Negative values indicate a southward field.',
  bt: 'Total magnitude of the interplanetary magnetic field at L1.',
  bx: 'Sunward component of the magnetic field in GSM coordinates.',
  by: 'Transverse component of the magnetic field in GSM coordinates.',
  t: 'Proton temperature in the solar wind at L1.',
  kp: 'Preliminary planetary geomagnetic activity index, preserved in native three-hour intervals.',
  dst: 'Provisional hourly index of the disturbance in the equatorial magnetic field.',
};

export default function ObservationInspector({ selected, onSelect, snapshot, windHistory, indexHistory, expanded, onToggle }: {
  selected: Selection; onSelect: (metric: Selection) => void; snapshot: LiveSnapshot;
  windHistory: QueryResult<History>; indexHistory: QueryResult<IndexHistory>;
  expanded: boolean; onToggle: () => void;
}) {
  const { label, unit, frame } = definition(selected);
  const { point, series, status, usable, age, outdated } = measurement(selected, snapshot);
  const wind = isWind(selected);
  const windPoint = wind ? snapshot.data?.solar_wind[selected]?.latest : undefined;
  const indexPoint = !wind ? snapshot.data?.geomagnetic[selected]?.latest : undefined;
  const history = wind ? windHistory.data?.series[selected] : indexHistory.data?.series[selected];
  const historyFailed = wind ? windHistory.failed : indexHistory.failed;
  const historyLoading = wind ? windHistory.loading : indexHistory.loading;
  const source = wind ? 'NOAA SWPC RTSW' : selected === 'kp' ? 'NOAA SWPC' : 'WDC Kyoto via NOAA SWPC';
  const path = wind ? 'solar-wind' : 'geomagnetic';
  return <aside id="observation-inspector" className={styles.inspector} aria-label="Selected observation" data-expanded={expanded}>
    <div className={styles.inspectorHeading}><Crosshair size={15} aria-hidden="true" /><span>INSPECTOR</span><span className={styles.metricCode}>LATEST SAMPLE</span><button className={styles.inspectorToggle} onClick={onToggle} aria-expanded={expanded} aria-controls="inspector-details">{label}<ChevronDown size={14} aria-hidden="true" /></button></div>
    <div id="inspector-details" className={styles.inspectorBody}>
      <label className={styles.fieldLabel} htmlFor="selected-observation">Selected measurement</label>
      <select id="selected-observation" className={styles.metricSelect} value={selected} onChange={event => onSelect(event.target.value as Selection)}>
        {selections.map(metric => <option key={metric} value={metric}>{definition(metric).label}</option>)}
      </select>
      <div className={styles.inspectorReading} style={{ '--metric-color': metricColors[selected] } as React.CSSProperties}>
        <h2>{label}{frame && <small> · {frame}</small>}</h2>
        <p>{number(point?.value, selected === 'kp' ? 2 : 1)} <span>{unit}</span></p>
        <span className={styles.metricState} data-warning={snapshot.data && !usable}><span className={styles.dot} />{status}{age != null && ` · ${Math.floor(age / 60)} min ${wind ? 'old' : 'since interval end'}`}</span>
      </div>
      {snapshot.failed && <p className={styles.notice}>Refresh failed. {point ? 'Showing the last received sample.' : 'No sample available.'}</p>}
      {outdated && <p className={styles.notice}>Snapshot is more than two minutes old.</p>}
      {point?.quality === 'flagged' && <p className={styles.notice}>Provider quality flag. This measurement is excluded from chart lines and derived trends.</p>}
      <p className={styles.inspectorDescription}>{explanations[selected]}</p>
      <dl className={styles.inspectorFacts}>
        <div><dt>Source</dt><dd>{source}</dd></div>
        {windPoint && <div><dt>Spacecraft</dt><dd>{windPoint.spacecraft}</dd></div>}
        <div><dt>{wind ? 'Observed · UTC' : 'Interval start · UTC'}</dt><dd className={styles.mono}>{stamp(windPoint?.observed_at ?? indexPoint?.interval_start)}</dd></div>
        {indexPoint && <div><dt>Interval end · UTC</dt><dd className={styles.mono}>{stamp(indexPoint.interval_end)}</dd></div>}
        {indexPoint && <div><dt>Interval state</dt><dd>{snapshot.data && Date.parse(indexPoint.interval_end) > Date.parse(snapshot.data.generated_at) + (snapshot.now && snapshot.receivedAt ? Math.max(0, snapshot.now - snapshot.receivedAt) : 0) ? 'In progress' : 'Completed · subject to revision'}</dd></div>}
        {indexPoint?.station_count != null && <div><dt>Contributing stations</dt><dd>{indexPoint.station_count}</dd></div>}
        <div><dt>Quality</dt><dd>{point?.quality === 'flagged' ? 'Provider quality flag' : point?.quality === 'unverified' ? 'Not independently validated' : 'No usable measurement'}</dd></div>
        {series && <div><dt>Delayed after</dt><dd>{series.stale_after_seconds / 60} min{!wind && ' from interval end'}</dd></div>}
      </dl>
      <div className={styles.inspectorSection}>
        <h3>Selected period</h3>
        {history ? <HistoryCoverage label={label} coverage={history.coverage} processing={'processing' in history ? history.processing : undefined} /> :
          <p className={styles.muted}>{historyFailed ? 'History unavailable for this period.' : historyLoading ? 'Loading coverage…' : 'No coverage available.'}</p>}
        {historyFailed && history && <p className={styles.warning}>History could not be refreshed.</p>}
      </div>
      <details className={styles.methodDetails}>
        <summary>How to read this measurement</summary>
        <p>{wind ? 'Native measurements are taken at L1 and are not shifted to Earth arrival time. NOAA selects the active spacecraft independently for plasma and magnetic data. Gaps and spacecraft changes break chart lines.' : 'Source intervals retain their original boundaries. A completed interval may still be revised. Freshness is measured from its end and allows for publication delay.'}</p>
        {wind && <p>One-hour changes compare five-minute means one hour apart, requiring at least 80% coverage in both windows and the last hour. Flags, stale data and spacecraft changes suppress trends.</p>}
      </details>
      <a className={styles.inspectorLink} href={`/api/v1/public/observations/${path}/latest?meta=true${wind ? `&metrics=${selected}` : ''}`}>Measurement data & metadata <ArrowUpRight size={14} aria-hidden="true" /></a>
      <a className={styles.inspectorLink} href={wind ? 'https://www.spaceweather.gov/products/solar-wind' : selected === 'kp' ? 'https://www.spaceweather.gov/products/planetary-k-index' : 'https://wdc.kugi.kyoto-u.ac.jp/dstdir/'}>Source documentation <ArrowUpRight size={14} aria-hidden="true" /></a>
    </div>
  </aside>;
}
