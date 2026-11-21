import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { build } from 'esbuild';

const { outputFiles } = await build({
  entryPoints: [fileURLToPath(new URL('../src/core/smartScoutEngine.ts', import.meta.url))],
  bundle: true,
  format: 'esm',
  platform: 'node',
  write: false,
});
const bundledSource = Buffer.from(outputFiles[0].contents).toString('base64');
const { estimatePitchFlight } = await import('data:text/javascript;base64,' + bundledSource);
const input = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(estimatePitchFlight(input)));
