'use client';
import { useEffect, useRef, useState, type CSSProperties } from 'react';
import Link from 'next/link';
import { usePathname, useSearchParams } from 'next/navigation';
import WorkspaceShell from '../../_components/WorkspaceShell';
import ForecastBoard from '../../_components/ForecastBoard';
import WindChart from '../../_components/WindChart';
import { products } from '../../_config/products';
import { formatForecastTime } from '../../_utils/forecast';
import { comparisonRows, countdown, demoHref, parseOffset, replayTime, selectRelease, type DemoBundle } from '../_lib/replay';
import { DemoContext } from './DemoContext';
import styles from './demo.module.css';
import forecastStyles from '../../_components/forecast.module.css';

export default function DemoWorkspace({ bundle, sectionPath }: { bundle: DemoBundle | null; sectionPath: string }) {
  const pathname = usePathname();
  const search = useSearchParams();
  const offset = parseOffset(search.get('offset'));
  const [playing, setPlaying] = useState(false);
  const [showFuture, setShowFuture] = useState(true);
  const [panelHeight, setPanelHeight] = useState(150);
  const panel = useRef<HTMLElement>(null);
  const now = bundle ? replayTime(bundle, offset) : undefined;
  const href = (path: string) => demoHref(path, offset);
  const setOffset = (value: number) => {
    const params = new URLSearchParams(search.toString());
    params.set('offset', String(value));
    window.history.replaceState(null, '', `${pathname}?${params}`);
  };
  useEffect(() => {
    if (!panel.current) return;
    const observer = new ResizeObserver(([entry]) => setPanelHeight(Math.ceil(entry.target.getBoundingClientRect().height)));
    observer.observe(panel.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    if (!playing || offset >= 0 || !bundle) return;
    const timer = window.setTimeout(() => {
      const params = new URLSearchParams(window.location.search);
      params.set('offset', String(offset + 1));
      window.history.replaceState(null, '', `${pathname}?${params}`);
      if (offset === -1) setPlaying(false);
    }, 1500);
    return () => window.clearTimeout(timer);
  }, [playing, offset, pathname, bundle]);
  const availableProducts = products.filter(product => bundle?.releases[product.slug]?.length);
  const selectedProduct = sectionPath.startsWith('/products/') ? products.find(product => sectionPath === `/products/${product.slug}`) : availableProducts[0];
  const catalog = sectionPath === '/products';
  const observations = sectionPath === '/live';
  const forecast = bundle && now != null && selectedProduct ? selectRelease(bundle, selectedProduct.slug, now) : null;
  const title = observations ? 'Historical observations' : catalog ? 'Demo forecast products' : sectionPath === '/' ? 'Forecast overview' : selectedProduct?.title ?? 'Forecast unavailable';
  const content = bundle && now != null ? <WorkspaceShell section={title}>
    <main id="forecast-content" className={forecastStyles.page}>
      <div className={forecastStyles.pageHeading}><div><div className={forecastStyles.eyebrow}>HISTORICAL REPLAY · 2026</div><h1>{title}</h1><p>{bundle.event.description}</p></div></div>
      <details className={styles.details}>
        <summary>About this event and the demo forecasts</summary>
        <p>Event begins: {formatForecastTime(bundle.event.starts_at)}. <a href={bundle.event.source_url} target="_blank" rel="noreferrer">Event source</a></p>
        <p>Retrospective forecasts generated specifically for this demo with current models. These are not operational releases.</p>
        <p>{bundle.input_source}</p>
        <p>Dataset {bundle.version} · Generated {formatForecastTime(bundle.generated_at)}. Observations: {bundle.observation_source}.</p>
        {Object.entries(bundle.models).map(([name, model]) => <p key={name}>{name} · SHA256 {model.sha256} · Training ends {formatForecastTime(model.training_end)} · {model.training_evidence}</p>)}
      </details>
      {catalog ? <div className={forecastStyles.catalogGrid}>{availableProducts.map(product => <Link key={product.slug} href={href(`/products/${product.slug}`)} className={forecastStyles.productCard}><h2>{product.title}</h2><p>{product.description}</p><span>Explore demo forecast →</span></Link>)}</div> : observations ? <div className={styles.observations}>
        <p>Completed observation hours before the simulated present · Last 24 hours · UTC</p>
        {availableProducts.flatMap(product => product.variables.filter(variable => bundle.releases[product.slug]?.some(release => release.available_variables.includes(variable.key)))).map(variable => <WindChart key={variable.key} title={variable.label} unit={variable.unit} comparison now={now} eventTime={Date.parse(bundle.event.starts_at)} data={comparisonRows(null, bundle.observations, variable.key, now, false)} />)}
      </div> : <>
        <nav className={forecastStyles.variableTabs} aria-label="Demo forecast products">{availableProducts.map(product => <Link className={forecastStyles.headingLink} key={product.slug} href={href(`/products/${product.slug}`)} aria-current={selectedProduct?.slug === product.slug ? 'page' : undefined}>{product.title}</Link>)}</nav>
        <label className={styles.toggle}><input type="checkbox" checked={showFuture} onChange={event => setShowFuture(event.target.checked)} />Show actual future values</label>
        {selectedProduct && <ForecastBoard key={selectedProduct.slug} product={selectedProduct} resource={{ data: forecast, error: forecast ? null : 'No demo release is available at this time.', retry: () => undefined }} />}
      </>}
    </main>
  </WorkspaceShell> : <main className={styles.unavailable}><h1>Historical demo is not published yet</h1><p>The event, historical observations and model forecasts are being prepared. Return to the live site for current conditions.</p></main>;
  return <div className={styles.root} style={{ '--demo-panel-height': `${panelHeight}px` } as CSSProperties}>
    <section ref={panel} className={styles.panel} aria-label="Demo time controls">
      <div className={styles.top}><div className={styles.event}><strong className={styles.badge}>DEMO</strong><strong>{bundle?.event.name ?? 'Historical event replay'}</strong>{bundle && <span>Event begins: {formatForecastTime(bundle.event.starts_at)}</span>}</div><Link className={styles.exit} href={sectionPath}>Exit demo</Link></div>
      {bundle && now != null ? <>
        <div className={styles.time}><time dateTime={new Date(now).toISOString()}>{formatForecastTime(new Date(now).toISOString())} · Simulated now</time><output htmlFor="demo-time">{countdown(offset)}</output></div>
        <div className={styles.range}><span>T−96 h</span><input id="demo-time" aria-label="Hours before event" aria-valuetext={`${countdown(offset)}, ${formatForecastTime(new Date(now).toISOString())}`} type="range" min={-96} max={0} step={1} value={offset} onChange={event => { setPlaying(false); setOffset(Number(event.target.value)); }} /><span>T0</span></div>
        <div className={styles.controls}><button aria-label={playing ? 'Pause replay' : 'Play replay'} onClick={() => { if (offset === 0) setOffset(-96); setPlaying(value => !value); }}>{playing ? 'Pause' : 'Play'}</button>{[-96, -72, -48, -24, -12, 0].map(value => <button key={value} aria-pressed={offset === value} onClick={() => { setPlaying(false); setOffset(value); }}>{value === 0 ? 'T0' : `−${Math.abs(value)}h`}</button>)}<span className={styles.method}>Retrospective forecasts · Current models</span></div>
      </> : <p>Demo data unavailable · Live forecasts are available on the normal site.</p>}
    </section>
    {bundle && now != null ? <DemoContext.Provider value={{ bundle, now, offset, showFuture, href }}>{content}</DemoContext.Provider> : content}
  </div>;
}
