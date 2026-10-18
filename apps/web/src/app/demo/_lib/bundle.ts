import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { products } from '@/forecast/products';
import { demoPageBundle, type DemoBundle, type DemoPageBundle } from './replay';

function defaultBundlePath() {
  let root = process.cwd();
  while (!existsSync(path.join(root, 'configs/project.yaml')) && path.dirname(root) !== root) root = path.dirname(root);
  return path.join(root, 'data/demo/current.json');
}

// Only a server-controlled path is read. URL segments never become filesystem paths.
export async function readDemoBundle(sectionPath: string): Promise<DemoPageBundle | null> {
  try {
    const bundle = JSON.parse(await readFile(process.env.DEMO_BUNDLE_PATH ?? defaultBundlePath(), 'utf8')) as DemoBundle;
    if (bundle.schema_version !== 1 || !bundle.version || !Number.isFinite(Date.parse(bundle.event?.starts_at)) ||
      new Date(bundle.event.starts_at).getUTCFullYear() !== 2026 || !bundle.releases || !Array.isArray(bundle.observations)) {
      throw new Error('Invalid demo bundle');
    }
    const product = sectionPath === '/'
      ? products.find(product => bundle.releases[product.slug]?.length)?.slug
      : sectionPath.startsWith('/forecast/') ? sectionPath.slice('/forecast/'.length) : undefined;
    return demoPageBundle(bundle, product, sectionPath === '/leo');
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') console.error('Demo bundle unavailable:', error);
    return null;
  }
}
