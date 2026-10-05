import { notFound } from 'next/navigation';
import { readDemoBundle } from '../_lib/bundle';
import DemoWorkspace from '../_components/DemoWorkspace';
import { productsBySlug } from '../../_config/products';

export const dynamic = 'force-dynamic';
export const metadata = { title: 'Historical demo', robots: { index: false, follow: false } };

export default async function DemoPage({ params }: { params: Promise<{ path?: string[] }> }) {
  const { path = [] } = await params;
  if (!(path.length === 0 || (path.length === 1 && ['products', 'live'].includes(path[0])) ||
    (path.length === 2 && path[0] === 'products' && !!productsBySlug[path[1]]))) notFound();
  return <DemoWorkspace bundle={await readDemoBundle()} sectionPath={`/${path.join('/')}`} />;
}
