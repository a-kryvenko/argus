import { productsBySlug } from '../_config/products';

export type Breadcrumb = { label: string; href?: string };

export function breadcrumbs(path: string, section: string): Breadcrumb[] {
  if (path === '/') return [{ label: 'Argus' }];
  const result: Breadcrumb[] = [{ label: 'Argus', href: '/' }];
  const [group, slug] = path.split('/').filter(Boolean);
  const groupLabel = { metrics: 'Metrics', forecast: 'Forecast', dashboard: 'Dashboard' }[group];
  if (!groupLabel) return [...result, { label: section }];
  if (!slug) return [...result, { label: groupLabel }];
  result.push({ label: groupLabel, href: `/${group}` });
  const product = (group === 'metrics' || group === 'forecast') ? productsBySlug[slug] : undefined;
  const label = product ? (slug === 'solar-wind-speed' ? 'Plasma speed' : product.title) : section;
  return [...result, { label }];
}
