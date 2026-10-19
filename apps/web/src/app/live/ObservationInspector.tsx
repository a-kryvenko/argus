'use client';
import { ArrowUpRight } from 'lucide-react';
import { isWind, measurement, stamp, type Selection } from './observations';
import type { LiveSnapshot } from './useLiveQuery';
import styles from './page.module.css';

const explanations: Record<Selection, string> = {
  v: 'Solar wind speed measured upstream of Earth at L1.',
  n: 'Proton number density in the solar wind at L1.',
  bz: 'Bz is the north–south magnetic field in GSM coordinates; negative values indicate a southward field. Bt is the total magnitude of the interplanetary magnetic field at L1.',
  bt: 'Total magnitude of the interplanetary magnetic field at L1.',
  bx: 'Sunward component of the magnetic field in GSM coordinates.',
  by: 'Transverse component of the magnetic field in GSM coordinates.',
  t: 'Proton temperature in the solar wind at L1.',
  kp: 'Preliminary planetary geomagnetic activity index, preserved in native three-hour intervals.',
  dst: 'Provisional hourly index of the disturbance in the equatorial magnetic field.',
  f10_7: 'Solar radio flux at a wavelength of 10.7 cm, measured in solar flux units (sfu).',
};

export default function ObservationInspector({ selected, snapshot }: {
  selected: Selection; snapshot: LiveSnapshot;
}) {
  const { point, series, outdated } = measurement(selected, snapshot);
  const bt = selected === 'bz' ? measurement('bt', snapshot) : undefined;
  const wind = isWind(selected);
  const windPoint = wind ? snapshot.data?.[selected] : undefined;
  const indexPoint = selected === 'kp' || selected === 'dst' ? snapshot.data?.[selected] : undefined;
  const solarPoint = selected === 'f10_7' ? snapshot.data?.f10_7 : windPoint;
  const source = wind ? 'NOAA SWPC RTSW' : selected === 'kp' || selected === 'f10_7' ? 'NOAA SWPC' : 'WDC Kyoto via NOAA SWPC';
  return <section id="observation-inspector" className={styles.inspector} aria-label="Selected observation">
    <div id="inspector-details" className={styles.inspectorBody}>
      <div className={styles.inspectorDescription}><p>{explanations[selected]}</p></div>
        <dl className={styles.inspectorFacts}>
          <div><dt>Source</dt><dd>{source}</dd></div>
          <div><dt>{wind || selected === 'f10_7' ? selected === 'bz' ? 'Bz observed · UTC' : 'Observed · UTC' : 'Interval start · UTC'}</dt><dd className={styles.mono}>{stamp(solarPoint?.observed_at ?? indexPoint?.interval_start)}</dd></div>
          {selected === 'bz' && <div><dt>Bt observed · UTC</dt><dd className={styles.mono}>{stamp(snapshot.data?.bt?.observed_at)}</dd></div>}
          {indexPoint && <div><dt>Interval end · UTC</dt><dd className={styles.mono}>{stamp(indexPoint.interval_end)}</dd></div>}
          {indexPoint && <div><dt>Interval state</dt><dd>{snapshot.data && Date.parse(indexPoint.interval_end) > (snapshot.now ?? snapshot.receivedAt ?? 0) ? 'In progress' : 'Completed · subject to revision'}</dd></div>}
          {(point?.quality || point?.value == null) && <div><dt>Quality</dt><dd>{point?.quality === 'flagged' ? 'Provider quality flag' : 'No usable measurement'}</dd></div>}
          {bt && (bt.flagged || bt.missing) && <div><dt>Bt quality</dt><dd>{bt.flagged ? 'Provider quality flag' : 'No usable measurement'}</dd></div>}
          {series && <div><dt>Delayed after</dt><dd>{series.stale_after_seconds / (selected === 'f10_7' ? 3600 : 60)} {selected === 'f10_7' ? 'h' : 'min'}{indexPoint && ' from interval end'}</dd></div>}
        </dl>
      <div className={styles.inspectorHelp}>
        {snapshot.failed && <p className={styles.notice}>Refresh failed. {point ? 'Showing the last received sample.' : 'No sample available.'}</p>}
        {outdated && <p className={styles.notice}>Snapshot is more than two minutes old.</p>}
        {point?.quality === 'flagged' && <p className={styles.notice}>Provider quality flag. This measurement is excluded from chart lines.</p>}
        <div className={styles.methodDetails}>
          <p>{selected === 'f10_7' ? 'Daily source observations retain their original timestamps. The latest available observation is shown.' : wind ? 'Native measurements are taken at L1 and are not shifted to Earth arrival time. NOAA selects the active spacecraft independently for plasma and magnetic data. Gaps and spacecraft changes break chart lines.' : 'Source intervals retain their original boundaries. A completed interval may still be revised. Freshness is measured from its end and allows for publication delay.'}</p>
          {wind && <p>One-hour changes compare five-minute means one hour apart, using available samples in each window. Flags, stale data and spacecraft changes suppress trends.</p>}
        </div>
        <a className={styles.inspectorLink} href={`/api/v1/observations?meta=true&observations=${selected === 'bz' ? 'bz,bt' : selected}`}>Measurement data & metadata <ArrowUpRight size={14} aria-hidden="true" /></a>
        <a className={styles.inspectorLink} href={wind ? 'https://www.spaceweather.gov/products/solar-wind' : selected === 'f10_7' ? 'https://www.spaceweather.gov/phenomena/f107-cm-radio-emissions' : selected === 'kp' ? 'https://www.spaceweather.gov/products/planetary-k-index' : 'https://wdc.kugi.kyoto-u.ac.jp/dstdir/'}>Source documentation <ArrowUpRight size={14} aria-hidden="true" /></a>
      </div>
    </div>
  </section>;
}
