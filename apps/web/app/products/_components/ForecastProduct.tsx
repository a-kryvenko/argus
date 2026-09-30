'use client';
import Link from 'next/link';
import { ArrowLeft, ArrowUpRight } from 'lucide-react';
import { useResource } from '../../_utils/useResource';
import type { Forecast } from '../../_utils/api';
import { productApiPath, type ProductConfig } from '../../_config/products';
import WorkspaceShell from '../../_components/WorkspaceShell';
import ForecastBoard from '../../_components/ForecastBoard';
import styles from '../../_components/forecast.module.css';

export default function ForecastProduct({ product }: { product: ProductConfig }) {
  const resource = useResource<Forecast>(productApiPath(product));
  return <WorkspaceShell section="Forecast products">
    <main id="forecast-content" className={styles.page}>
      <Link href="/products" className={styles.backLink}><ArrowLeft size={12} aria-hidden="true" />Forecast products</Link>
      <div className={styles.pageHeading}><div><div className={styles.eyebrow}>MODEL OUTLOOK <span>/ {product.variables.map(variable => variable.key.toUpperCase()).join(' · ')}</span></div><h1>{product.title}</h1><p>{product.description}</p></div>
        <Link className={styles.headingLink} href={`/metrics/${product.slug}`}>Model performance<ArrowUpRight size={13} aria-hidden="true" /></Link>
      </div>
      <ForecastBoard key={product.slug} product={product} resource={resource} />
      <div className={styles.pageFooter}><span>Published forecast · all times UTC</span><Link href="/">Forecast overview<ArrowUpRight size={12} aria-hidden="true" /></Link></div>
    </main>
  </WorkspaceShell>;
}
