// Run the browser feature code (docs/js/features.js) in Node for tools/verify_features.py.
//
//   node tools/features_cli.mjs <dir>
//
// <dir>/manifest.json lists signals as raw little-endian float32 files at 22,050 Hz:
//   [{"name": "...", "file": "x.f32", "hop_seconds": 1.0}, ...]
// For each signal it writes <name>.out.f32 (all windows' features back to back) and
// <name>.starts.json (window start times in seconds).
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DEFAULT_CONFIG, LogMel } from '../docs/js/features.js';

const here = dirname(fileURLToPath(import.meta.url));
const dir = process.argv[2];
const filters = JSON.parse(readFileSync(join(here, '..', 'docs', 'model', 'mel_filters.json'), 'utf8'));
const logmel = new LogMel(DEFAULT_CONFIG, filters);
const manifest = JSON.parse(readFileSync(join(dir, 'manifest.json'), 'utf8'));

for (const item of manifest) {
  const buf = readFileSync(join(dir, item.file));
  const y = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  const t0 = performance.now();
  const { features, starts } = logmel.windows(y, item.hop_seconds ?? 1.0);
  const ms = (performance.now() - t0) / features.length;
  const out = new Float32Array(features.length * features[0].length);
  features.forEach((f, i) => out.set(f, i * f.length));
  writeFileSync(join(dir, `${item.name}.out.f32`), Buffer.from(out.buffer));
  writeFileSync(join(dir, `${item.name}.starts.json`), JSON.stringify({ starts, ms_per_window: ms }));
}
