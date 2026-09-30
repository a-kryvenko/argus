import Link from 'next/link';
import { ArrowUpRight, Layers3 } from 'lucide-react';
import { products } from '../_config/products';
import WorkspaceShell from '../_components/WorkspaceShell';
import styles from '../_components/forecast.module.css';

export const metadata = { title: 'Forecast products' };

export default function Products() {
  return <WorkspaceShell section="Forecast products" status="Forecast catalog">
    <main id="forecast-content" className={styles.page}>
      <div className={styles.pageHeading}><div><div className={styles.eyebrow}>MODEL CATALOG <span>/ {String(products.length).padStart(2, '0')} PRODUCTS</span></div><h1>Forecast products</h1><p>Explore solar wind, magnetic field, solar radiation and geomagnetic outlooks.</p></div>
        <Link className={styles.headingLink} href="/">Forecast overview<ArrowUpRight size={13} aria-hidden="true" /></Link>
      </div>
      <div className={styles.catalogHeading}><span><Layers3 size={14} aria-hidden="true" />Available product definitions</span><span>Quantiles & threshold probabilities</span></div>
      <div className={styles.catalogGrid}>{products.map((product, index) => <Link className={styles.productCard} href={`/products/${product.slug}`} key={product.slug}>
        <div className={styles.productCardTop}><span>{String(index + 1).padStart(2, '0')}</span><span>{product.variables.map(variable => variable.key.toUpperCase()).join(' / ')}</span></div>
        <h2>{product.title}</h2><p>{product.description}</p>
        <div className={styles.capabilities}>{product.variables.some(variable => variable.quantile) && <span>q10 · q50 · q90</span>}{product.variables.some(variable => variable.thresholds.length > 0) && <span>Threshold probability</span>}</div>
        <dl>{product.variables.map(variable => <div key={variable.key}><dt>{variable.label}</dt><dd>{variable.unit}</dd></div>)}</dl>
        <div className={styles.productCardFooter}><span>Explore forecast</span><ArrowUpRight size={15} aria-hidden="true" /></div>
      </Link>)}</div>
      <p className={styles.footerNote}>Data availability and forecast horizon depend on the published release. Units and supported variables are fixed for each product.</p>
    </main>
  </WorkspaceShell>;
}
