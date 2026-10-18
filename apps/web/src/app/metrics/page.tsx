import Link from 'next/link';
import { ArrowUpRight, ChartNoAxesCombined } from 'lucide-react';
import { products } from '../../forecast/products';
import WorkspaceShell from '../../components/WorkspaceShell';
import styles from '../../components/forecast.module.css';

export const metadata = { title: 'Model performance' };
export default function Metrics() {
  return <WorkspaceShell section="Model performance" contentId="metrics-content" status="Evaluation catalog">
    <main id="metrics-content" className={styles.page}>
      <div className={styles.pageHeading}><div><h1>Model performance</h1><p>Explore forecast quality across lead times, variables and event thresholds.</p></div><Link className={styles.headingLink} href="/forecast">Forecast<ArrowUpRight size={13} aria-hidden="true" /></Link></div>
      <div className={styles.catalogHeading}><span><ChartNoAxesCombined size={14} aria-hidden="true" />Evaluation by product</span><span>Errors · threshold scores · calibration</span></div>
      <div className={styles.catalogGrid}>{products.map(product => <Link className={styles.productCard} href={`/metrics/${product.slug}`} key={product.slug}>

        <h2>{product.title}</h2><p>{product.variables.map(variable => variable.label).join(' · ')}</p>
        <div className={styles.capabilities}>{product.variables.some(variable => variable.quantile) && <span>Continuous errors</span>}{product.variables.some(variable => variable.thresholds.length > 0) && <><span>Threshold scores</span><span>Reliability</span></>}</div>
        <div className={styles.productCardFooter}><span>Explore metrics</span><ArrowUpRight size={15} aria-hidden="true" /></div>
      </Link>)}</div>
      <p className={styles.footerNote}>Select a product to inspect published evaluation results. Available metrics and evaluated lead times depend on the model output.</p>
    </main>
  </WorkspaceShell>;
}
