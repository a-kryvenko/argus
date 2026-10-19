import { copyFile, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

// Next emits different entrypoint formats across supported Node versions.
const source = resolve(process.argv[2]);
const code = await readFile(source, 'utf8');
const extension = /^import\s/m.test(code) ? 'mjs' : 'cjs';
const entrypoint = join(dirname(source), `server.${extension}`);
await copyFile(source, entrypoint);
await import(pathToFileURL(entrypoint).href);
