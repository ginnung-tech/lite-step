/**
 * Offline vendoring build for the headless Vision-render page.
 *
 * The render page (`index.html`) drives ThatOpen Components — the SAME
 * loader stack the browser app migrated to in the 2026-06 3D-viewer
 * migration (IfcImporter -> fragments -> FragmentsManager). Those packages
 * have a deep dependency graph (three, web-ifc, flatbuffers, …) that cannot
 * be curled as loose ESM files the way the legacy three.js + web-ifc pair
 * was; they have to be bundled.
 *
 * This script produces a fully self-contained `web_render/vendor/` tree so
 * the page has ZERO network dependency at render time (it loads from a
 * `file://` document inside the sandbox — see worker/render.py). It runs at
 * image-build time (Dockerfile) where npm registry access is available;
 * the output tree is what the sandbox loads at render time.
 *
 * Outputs (all under the chosen out dir, default ./vendor):
 *
 *   vendor/render-app.js          — bundled ESM: three + @thatopen/components
 *                                   + @thatopen/fragments + web-ifc glue +
 *                                   the render logic (src/render-app.mjs).
 *   vendor/fragments-worker.mjs   — the prebuilt @thatopen/fragments geometry
 *                                   worker (self-contained; no external fetch).
 *   vendor/web-ifc/web-ifc.wasm   — web-ifc single-thread WASM (IfcImporter).
 *   vendor/web-ifc/web-ifc-mt.wasm— multi-thread WASM (best-effort copy).
 *
 * Usage:  node build-vendor.mjs [--out <dir>]
 */
import { build } from 'esbuild';
import {
  mkdirSync,
  copyFileSync,
  existsSync,
  statSync,
  readdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(fileURLToPath(import.meta.url));

// Provenance stamp written next to the bundle. `worker/render.py`
// reads it back to refuse rendering a vendor bundle built from stale
// source — the failure mode that silently dropped geometry for days
// (a dev edits src/render-app.mjs but never re-runs this script, so the
// gitignored vendor/render-app.js next to it is a pre-fix build). The
// stamp is a CONTENT hash, not an mtime: `cp -r vendor` clobbers mtimes
// but carries the stamp verbatim, so a copied stale bundle is still
// caught. See BUILD_INFO_NAME + _verify_vendor_freshness in render.py.
const BUILD_INFO_NAME = 'BUILD_INFO.json';

// Deterministic hash over every *.mjs under src/, sorted by POSIX
// relative path: `sha256( join("<relpath>:<sha256(bytes)>\n") )`.
// worker/render.py:_source_hash reimplements this byte-for-byte — keep
// the two in lockstep (same sort key, same per-file hex digest, same
// "rel:hash\n" framing) or the freshness check false-positives.
function sourceFilesSorted(srcDir) {
  const out = [];
  const walk = (dir, rel) => {
    const entries = readdirSync(dir, { withFileTypes: true });
    for (const ent of entries) {
      const abs = join(dir, ent.name);
      const r = rel ? `${rel}/${ent.name}` : ent.name;
      if (ent.isDirectory()) walk(abs, r);
      else if (ent.name.endsWith('.mjs')) out.push([r, abs]);
    }
  };
  walk(srcDir, '');
  out.sort((a, b) => {
    if (a[0] < b[0]) return -1;
    return a[0] > b[0] ? 1 : 0;
  });
  return out;
}

function sourceHash(srcDir) {
  const h = createHash('sha256');
  for (const [rel, abs] of sourceFilesSorted(srcDir)) {
    const fileHash = createHash('sha256').update(readFileSync(abs)).digest('hex');
    h.update(`${rel}:${fileHash}\n`);
  }
  return h.digest('hex');
}

function argOut() {
  const i = process.argv.indexOf('--out');
  return i >= 0 && process.argv[i + 1]
    ? resolve(process.argv[i + 1])
    : join(ROOT, 'vendor');
}

const OUT = argOut();
const NM = join(ROOT, 'node_modules');

function req(path, label) {
  if (!existsSync(path)) {
    throw new Error(`${label} missing at ${path} — run \`bun install\` first`);
  }
  return path;
}

async function main() {
  mkdirSync(OUT, { recursive: true });
  mkdirSync(join(OUT, 'web-ifc'), { recursive: true });

  // 1. Bundle the render module + the whole ThatOpen/three/web-ifc graph
  //    into one self-contained ESM file. `web-ifc.wasm` is NOT bundled —
  //    web-ifc resolves it at runtime from the path we set explicitly
  //    (importer.wasm.path), so it stays a loose asset under vendor/web-ifc/.
  await build({
    entryPoints: [join(ROOT, 'src', 'render-app.mjs')],
    bundle: true,
    format: 'esm',
    target: 'es2022',
    minify: true,
    legalComments: 'none',
    outfile: join(OUT, 'render-app.js'),
    logLevel: 'info',
    // web-ifc ships a Node-flavoured entry that references `fs`/`path`;
    // the browser build guards them behind feature checks, so mark them
    // external and let the guards no-op in the browser.
    external: ['fs', 'path', 'perf_hooks', 'worker_threads'],
  });

  // 2. Copy the prebuilt fragments geometry worker (self-contained module
  //    worker — no external fetch/import at runtime, verified against the
  //    package). The page loads it via a Blob URL so the file:// document
  //    can spawn it without the same-origin worker restriction.
  copyFileSync(
    req(join(NM, '@thatopen', 'fragments', 'dist', 'Worker', 'worker.mjs'), 'fragments worker'),
    join(OUT, 'fragments-worker.mjs'),
  );

  // 3. Copy web-ifc WASM. Single-thread is required; multi-thread is a
  //    best-effort extra (only used under cross-origin isolation, which the
  //    file:// render document is not — the single-thread path is what runs).
  copyFileSync(
    req(join(NM, 'web-ifc', 'web-ifc.wasm'), 'web-ifc.wasm'),
    join(OUT, 'web-ifc', 'web-ifc.wasm'),
  );
  const mt = join(NM, 'web-ifc', 'web-ifc-mt.wasm');
  if (existsSync(mt)) copyFileSync(mt, join(OUT, 'web-ifc', 'web-ifc-mt.wasm'));

  // 4. Stamp the bundle with the source content hash so the render
  //    entry point can prove the vendored artifact matches the source
  //    it was built from (see BUILD_INFO_NAME comment above).
  const srcDir = join(ROOT, 'src');
  const buildInfo = {
    sourceHash: sourceHash(srcDir),
    sources: sourceFilesSorted(srcDir).map(([rel]) => rel),
    builtAt: new Date().toISOString(),
  };
  writeFileSync(join(OUT, BUILD_INFO_NAME), `${JSON.stringify(buildInfo, null, 2)}\n`);

  // Summary so the image build log shows what landed.
  for (const f of ['render-app.js', 'fragments-worker.mjs', 'web-ifc/web-ifc.wasm', BUILD_INFO_NAME]) {
    const p = join(OUT, f);
    console.log(`  vendored ${f} (${statSync(p).size} bytes)`);
  }
  console.log(`  source hash ${buildInfo.sourceHash.slice(0, 12)}… over ${buildInfo.sources.length} src file(s)`);
  console.log(`vendor build complete -> ${OUT}`);
}

try {
  await main();
} catch (e) {
  console.error('vendor build failed:', e);
  process.exit(1);
}
