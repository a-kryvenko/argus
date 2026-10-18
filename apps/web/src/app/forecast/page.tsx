import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';
import { products } from '../../forecast/products';
import WorkspaceShell from '../../components/WorkspaceShell';
import styles from '../../components/forecast.module.css';

export const metadata = { title: 'Forecast' };

export default function Forecast() {
  return <WorkspaceShell section="Forecast" status="Forecast catalog">
    <main id="forecast-content" className={styles.page}>
      <div className={styles.pageHeading}><div><h1>Forecast</h1><p>Explore solar wind, magnetic field, solar radiation and geomagnetic outlooks.</p></div>
      </div>
      <div className={styles.catalogGrid}><Link className={styles.productCard} href="/forecast/solar-events"><h2>Solar events</h2><p>CME arrivals at Earth with expected solar-wind speed and early/late arrival scenarios.</p><div className={styles.capabilities}><span>Arrival forecast</span><span>Speed scenarios</span></div><div className={styles.productCardFooter}><span>Explore forecast</span><ArrowUpRight size={15} aria-hidden="true" /></div></Link>{products.map(product => <Link className={styles.productCard} href={`/forecast/${product.slug}`} key={product.slug}>

        <h2>{product.title}</h2><p>{product.description}</p>
        <div className={styles.capabilities}>{product.variables.some(variable => variable.quantile) && <span>q10 · q50 · q90</span>}{product.variables.some(variable => variable.thresholds.length > 0) && <span>Threshold probability</span>}</div>
        <dl>{product.variables.map(variable => <div key={variable.key}><dt>{variable.label}</dt><dd>{variable.unit}</dd></div>)}</dl>
        <div className={styles.productCardFooter}><span>Explore forecast</span><ArrowUpRight size={15} aria-hidden="true" /></div>
      </Link>)}</div>
      <p className={styles.footerNote}>Data availability and forecast horizon depend on the published release. Units and supported variables are fixed for each product.</p>
    </main>
  </WorkspaceShell>;
}
