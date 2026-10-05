import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import type { DemoBundle } from './replay';

function defaultBundlePath() {
  let root = process.cwd();
  while (!existsSync(path.join(root, 'configs/project.yaml')) && path.dirname(root) !== root) root = path.dirname(root);
  return path.join(root, 'data/demo/current.json');
}

// Only a server-controlled path is read. URL segments never become filesystem paths.
export async function readDemoBundle(): Promise<DemoBundle | null> {
  try {
    const bundle = JSON.parse(await readFile(process.env.DEMO_BUNDLE_PATH ?? defaultBundlePath(), 'utf8')) as DemoBundle;
    if (bundle.schema_version !== 1 || !bundle.version || !Number.isFinite(Date.parse(bundle.event?.starts_at)) ||
      new Date(bundle.event.starts_at).getUTCFullYear() !== 2026 || !bundle.releases || !Array.isArray(bundle.observations)) {
      throw new Error('Invalid demo bundle');
    }
    return bundle;
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== 'ENOENT') console.error('Demo bundle unavailable:', error);
    return null;
  }
}
