'use client';
import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';
import { useResource } from '../../../hooks/useResource';
import type { Forecast } from '@/forecast/types';
import { productApiPath, type ProductConfig } from '../../../forecast/products';
import WorkspaceShell from '../../../components/WorkspaceShell';
import ForecastBoard from '../../../components/ForecastBoard';
import styles from '../../../components/forecast.module.css';

export default function ForecastProduct({ product }: { product: ProductConfig }) {
  const resource = useResource<Forecast>(productApiPath(product));
  return <WorkspaceShell section="Forecast">
    <main id="forecast-content" className={styles.page}>
      <div className={styles.pageHeading}><div><h1>{product.title}</h1><p>{product.description}</p></div>
        <Link className={styles.headingLink} href={`/metrics/${product.slug}`}>Model performance<ArrowUpRight size={13} aria-hidden="true" /></Link>
      </div>
      <ForecastBoard key={product.slug} product={product} resource={resource} />
      <div className={styles.pageFooter}><span>Published forecast · all times UTC</span></div>
    </main>
  </WorkspaceShell>;
}
