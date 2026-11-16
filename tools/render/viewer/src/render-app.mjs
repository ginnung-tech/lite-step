/**
 * Headless Vision-render module — ThatOpen Components edition.
 *
 * This is the rendering internals for `web_render/index.html`. It loads an
 * IFC through the SAME stack the browser app uses (the 2026-06 ThatOpen 3D
 * viewer migration): `@thatopen/fragments` `IfcImporter` converts the IFC
 * STEP to the fragments binary, then `@thatopen/components`
 * `FragmentsManager` decodes it (off-thread, via the fragments worker) into a
 * `THREE.Object3D`. Because the loader is identical to the app's, the
 * validator sees exactly what the user sees — same skipped-element behaviour
 * (fragments drops geometry >100000 m from the origin), same coordinate
 * handling (`autoCoordinate = false`, local coords centred at origin), same
 * materials baked into the fragments. The legacy page reimplemented loading
 * with raw web-ifc `StreamAllMeshes`, which drifted from the app and let the
 * vision validator pass things the app rendered differently.
 *
 * The module keeps the EXACT external contract the Playwright driver
 * (worker/render.py) depends on, unchanged from the legacy page:
 *
 *   window.__viewStems  — the 6 filename stems, in order, for the Python loop.
 *   window.renderView(stem) — render one view onto #c and label #cap. May
 *                             return a Promise; page.evaluate awaits it.
 *   window.__ready      — set true once the model is built and renderable.
 *   window.__error      — set to an error string if the build failed.
 *   #wrap / #c / #cap   — the 1024x1024 capture element, canvas, caption.
 *
 * The 6 views + burned-in labels + elevation/mode query params are ported
 * verbatim from the legacy page so the filenames, on-image labels, and the
 * orchestrator's upload-key parity are all unchanged.
 */

import * as THREE from 'three';
import * as OBC from '@thatopen/components';
import { IfcImporter, LodMode } from '@thatopen/fragments';

const W = 1024, H = 1024;

// Sky-tone background — a neutral daylight backdrop for the massing read.
const BG = 0x87a5c8;

// Query params: ?mode=aerial|cardinal (default aerial), ?elev=<1..80> (22).
const params = new URLSearchParams(location.search);
const elevDeg = Math.min(80, Math.max(1, parseFloat(params.get('elev')) || 22));
const ELEV = elevDeg * Math.PI / 180;          // applies to BOTH view modes
const AZ = Math.sin(ELEV);                      // vertical component (shared)
const AC = Math.cos(ELEV) / Math.SQRT2;         // corner: horizontal split over 2 axes
const CH = Math.cos(ELEV);                      // axis: full horizontal on 1 axis
const mode = (params.get('mode') || 'aerial').toLowerCase();
const D = `${elevDeg}° up`;

// View tables: [filename-stem, burned-in label, camera-direction unit vector
// from the building centre to the camera]. Same 6 stems in both modes so the
// filenames (render_<stem>.png) stay the legacy upload keys; the label + the
// camera vector differ. Ported verbatim from the legacy page.
const AERIAL = [
  ["xpos",  `aerial  +X +Y  (${D})`,       [ AC,  AC, AZ]],
  ["xneg",  `aerial  +X −Y  (${D})`,       [ AC, -AC, AZ]],
  ["ypos",  `aerial  −X +Y  (${D})`,       [-AC,  AC, AZ]],
  ["yneg",  `aerial  −X −Y  (${D})`,       [-AC, -AC, AZ]],
  ["zpos",  "top-down  (from +Z)",          [ 0,   0,   1]],
  ["zneg",  "underside  (−Z, site hidden)", [ 0,   0,  -1]],
];
const CARDINAL = [
  ["xpos",  `axis  +X  (${D})`,             [ CH,  0,  AZ]],
  ["xneg",  `axis  −X  (${D})`,             [-CH,  0,  AZ]],
  ["ypos",  `axis  +Y  (${D})`,             [ 0,   CH, AZ]],
  ["yneg",  `axis  −Y  (${D})`,             [ 0,  -CH, AZ]],
  ["zpos",  "top-down  (from +Z)",          [ 0,   0,   1]],
  ["zneg",  "underside  (−Z, site hidden)", [ 0,   0,  -1]],
];
// ── local visual-debugger controls ────────────────────────────────────────
//
// All OPT-IN: with none of these params present the page behaves exactly as
// production does (same 6 stems, same labels, same framing), so the vision
// validator is unaffected. They exist because the ONLY way to chase a
// "geometry is correct in web-ifc but the element is missing in the app"
// class of bug is to drive this exact stack — IfcImporter + FragmentsManager,
// the app's loader — and look at it from an angle you choose.
//
//   views=az:elev,az:elev…  replace the 6 canned views with arbitrary cameras.
//                           az is degrees CCW from +X toward +Y, elev degrees
//                           above the horizon. Stems become a<az>_e<elev>, so
//                           worker/render.py's __viewStems loop picks them up
//                           with no driver change.
//   target=x,y,z            look-at point in IFC world metres (default: the
//                           building centre). Point it at a corner to inspect
//                           a miter joint.
//   ortho=1 & span=<m>      orthographic camera with the given half-width — a
//                           plan/elevation with no perspective skew, which is
//                           what you want when comparing two mitered faces.
//   dist=<mult>            perspective distance multiplier (default 1).
// (An only=/hide= isolation pass was built and REMOVED: `model.setVisible`
//  on this fragments version does not take effect for arbitrary item ids —
//  it reported "1 shown, 128 hidden" while rendering the whole building. Use
//  census= to ask what is present, and target=/span= to look at one corner,
//  until the visibility API is understood. Leaving a flag that silently lies
//  about what you are looking at would be worse than not having it.)
//   edges=1                overlay black edge lines — coplanar boolean seams
//                           and redundant carve faces show up as extra edges.
//   census=1               dump a per-element table (category, name, has
//                           geometry, bbox) to the console. This is the
//                           decisive check for "is the element in the scene
//                           at all?" — far more reliable than eyeballing.
const dbg = {
  views: params.get('views'),
  target: params.get('target'),
  ortho: params.get('ortho') === '1',
  span: parseFloat(params.get('span')) || 0,
  dist: parseFloat(params.get('dist')) || 1,
  edges: params.get('edges') === '1',
  census: params.get('census') === '1',
  winding: params.get('winding') === '1',
  doubleside: params.get("doubleside") === "1",
};

/** Parse `views=` into VIEW-table rows: [stem, label, unit direction]. */
function parseCustomViews(spec) {
  const out = [];
  for (const pair of spec.split(',')) {
    const [azRaw, elRaw] = pair.split(':');
    const az = parseFloat(azRaw), el = parseFloat(elRaw);
    if (!Number.isFinite(az) || !Number.isFinite(el)) {
      throw new Error(`views=: bad az:elev pair "${pair}" (expected e.g. 45:25)`);
    }
    const a = az * Math.PI / 180, e = el * Math.PI / 180;
    // Filename-safe stem; negatives become "m" so render_a-45_e25.png can't
    // collide with a flag-looking path on the Python side.
    const tag = (n) => (n < 0 ? 'm' : '') + Math.abs(Math.round(n));
    out.push([
      `a${tag(az)}_e${tag(el)}`,
      `az ${az}°  elev ${el}°${dbg.ortho ? '  (ortho)' : ''}`,
      [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)],
    ]);
  }
  return out;
}

const VIEWS = dbg.views ? parseCustomViews(dbg.views)
            : mode === 'cardinal' ? CARDINAL : AERIAL;
window.__viewStems = VIEWS.map(v => v[0]); // Python reads this to stay in sync

// ── three.js scene / camera / lights ──────────────────────────────────────
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setSize(W, H, false);
renderer.setClearColor(BG, 1);

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
const cam = new THREE.PerspectiveCamera(60, 1, 0.05, 100000);
cam.up.set(0, 0, 1); // Z-up world (IFC convention)

// Orthographic alternative for `ortho=1` — a plan/elevation with no
// perspective skew. Its half-width comes from `span=` (or the framed radius),
// set in renderView once `radius` is known.
const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, -100000, 100000);
ortho.up.set(0, 0, 1);

// Lighting mirrors the browser app's scene (ambient + key sun + cool fill);
// positioned in the Z-up world we rotate the model into below. No shadow
// maps — the app doesn't cast them on this path and SwiftShader shadow maps
// are slow + artefact-prone. The point is comparable shading, not a
// re-lit scene the user never sees.
scene.add(new THREE.AmbientLight(0x889aaa, 0.9));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
scene.add(sun);
scene.add(sun.target);
const fill = new THREE.DirectionalLight(0x5a8aae, 0.5);
scene.add(fill);

let center = new THREE.Vector3();
let radius = 5;
let fragments = null;            // OBC.FragmentsManager
let model = null;                // FragmentsModel
let contextLocalIds = [];        // IfcSite / IfcGeographicElement element ids
let contextHidden = false;       // current visibility state of context ids

/**
 * Fetch a vendored module and hand it back as a Blob URL. Creating a Worker
 * directly from a file:// URL is refused by Chromium even with
 * --allow-file-access-from-files (opaque origin); a same-document Blob URL is
 * not. The worker.mjs is self-contained (no external fetch/import at
 * runtime), so a Blob copy runs identically. The page's fetch shim routes the
 * file:// GET through XHR.
 */
async function blobModuleUrl(relUrl) {
  const abs = new URL(relUrl, document.baseURI).href;
  const res = await fetch(abs);
  const code = await res.text();
  return URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
}

async function build() {
  // 1. IFC bytes — same-origin file:// GET (fetch shim -> XHR).
  const ifcBuf = await (await fetch('./output.ifc')).arrayBuffer();
  const ifcBytes = new Uint8Array(ifcBuf);

  // 2. IFC -> fragments binary via IfcImporter (the app's converter). web-ifc
  //    resolves its WASM from the explicit absolute path we set; the fetch
  //    shim serves it from vendor/web-ifc/ under file://.
  const importer = new IfcImporter();
  importer.wasm = {
    absolute: true,
    path: new URL('./vendor/web-ifc/', document.baseURI).href,
  };
  const fragBytes = await importer.process({ bytes: ifcBytes });

  // 3. Components + FragmentsManager — the app's decode path. autoCoordinate
  //    off: the IFC uses local coordinates centred at the origin (mirrors the
  //    app's canvas/fragments.ts). Worker loaded via a Blob URL (see above).
  const components = new OBC.Components();
  fragments = components.get(OBC.FragmentsManager);
  fragments.init(await blobModuleUrl('./vendor/fragments-worker.mjs'));
  fragments.core.settings.autoCoordinate = false;

  // 4. Load + force a full update so all geometry is materialised (no camera
  //    passed to load -> no frustum culling -> every element is present for
  //    every static view).
  model = await fragments.core.load(fragBytes.buffer, { modelId: 'validator' });
  // Disable the fragments LOD culler. NOT optional and NOT a tuning choice:
  // fragments defaults to LodMode.DEFAULT, whose clipping test runs against
  // `_virtualPlanes` — and this page never calls `useCamera`, so that array is
  // six copies of a default-constructed THREE.Plane() (normal (1,0,0),
  // constant 0, i.e. the plane x = 0). Every item whose bbox lies entirely at
  // x < 0 is then discarded, expressed by OMITTING its index range from
  // geometry.groups — so it stays in the buffer, stays in every data-side
  // query, sits on a mesh reporting visible=true with a full drawRange, and
  // never reaches the framebuffer.
  //
  // Measured on the roofed villa: 852 of 26719 indices (284 triangles) silently
  // undrawn — a facade wall, a stenkant patch, the outermost rafter bay. A
  // model centred on the origin would lose HALF its geometry. Every vision
  // validation rendered before this fix may have been judging an incomplete
  // image, with nothing in any log to say so.
  //
  // Upstream: https://github.com/ThatOpen/engine_fragment/issues/255
  try {
    await model.setLodMode(LodMode.ALL_VISIBLE);
    console.warn('lod: LodMode.ALL_VISIBLE (fragments LOD culler disabled)');
  } catch (e) {
    console.warn(`lod: setLodMode(ALL_VISIBLE) FAILED (${e}) — the culler is still on `
      + `and MAY silently drop elements from these renders`);
  }
  await fragments.core.update(true);
  await new Promise(r => requestAnimationFrame(r));

  const group = model.object;
  // Fragments displays geometry Y-up (the app renders model.object with a
  // Y-up camera). Rotate the display group +90° about X so the model stands
  // Z-up, aligned with the IFC-native axes the data queries (getMergedBox
  // below) and the cardinal cameras + labels use (X = E-W, Y = N-S,
  // Z = vertical) — the axis convention SKILL.md documents. After this the
  // display frame and the data frame coincide (both Z-up).
  group.rotation.x = Math.PI / 2;
  scene.add(group);
  group.updateMatrixWorld(true);

  // 4b. Wait until the fragments worker has FULLY materialised the scene.
  // The decode lands meshes asynchronously over several update ticks
  // (slow under SwiftShader + bwrap), and a single update + one RAF is
  // NOT a completion signal: screenshotting while meshes are still
  // streaming in is exactly the view-dependent "walls/roof missing from
  // the first aerial view" divergence vs the app (#374 — the app never
  // shows this because its render loop keeps updating every frame).
  // Stable = mesh/instance/vertex counts unchanged for 3 consecutive
  // ticks with at least one mesh present; time-capped so a pathological
  // model still renders whatever landed (loud breadcrumb either way).
  {
    let lastKey = '';
    let stable = 0;
    const t0 = performance.now();
    while (stable < 3 && performance.now() - t0 < 30000) {
      await fragments.core.update(true);
      await new Promise(r => requestAnimationFrame(r));
      let meshes = 0;
      let instances = 0;
      let verts = 0;
      group.traverse((o) => {
        if (o.isMesh) {
          meshes++;
          if (o.isInstancedMesh) instances += o.count;
          const pos = o.geometry?.attributes?.position;
          verts += pos ? pos.count : 0;
        }
      });
      const key = `${meshes} meshes / ${instances} instances / ${verts} vertices`;
      stable = (key === lastKey && meshes > 0) ? stable + 1 : 0;
      lastKey = key;
    }
    group.updateMatrixWorld(true);
    if (stable < 3) {
      console.warn(`render: scene did NOT stabilise within 30s (last state: ${lastKey}) — screenshots may be partial`);
    } else {
      console.warn(`render: scene materialised — ${lastKey} (stable x3)`);
    }
  }

  // 5. Identify site / geographic "context" elements so the camera frames to
  //    the BUILDING (not the huge terrain pad — a create_flat_site block
  //    extends ~20 m underground and dwarfs the building) and the underside
  //    view can hide the site pad. Best-effort: a fragments API shift or a
  //    model with no site just leaves contextLocalIds empty, and every view
  //    frames to / shows everything.
  let allIds = [];
  try {
    allIds = await model.getLocalIds();
    const cats = await model.getItemsOfCategories([/IFCSITE/, /IFCGEOGRAPHICELEMENT/]);
    contextLocalIds = Object.values(cats).flat().filter((n) => Number.isInteger(n));
    console.warn(`context elements: ${contextLocalIds.length} of ${allIds.length}`);
  } catch (e) {
    console.warn(`context category query failed (${e}) — no site hiding`);
  }

  // Loud breadcrumb (front-load observability): which IFC element categories
  // carried NO body geometry, so they render as nothing. This is the single
  // most useful signal for "why is a view blank / missing an element" — the
  // app's loader drops anything not in a Body/Model representation (e.g. a
  // Sketch-subcontext element), so a category showing here means the user
  // won't see it in the app either. Same as what the validator sees.
  //
  // Container categories (IfcRoof/IfcWindow/IfcDoor/…) legitimately carry
  // their body on AGGREGATED CHILDREN (roof solids, frame members, glass
  // panes) — the container entity itself has no direct representation, so
  // measuring the bare entity flags correct buildings ("roof lacks body
  // geometry" on a roof that renders fine — the #374 false warnings).
  // Measure the whole subtree via getItemsChildren when the fragments API
  // provides it; when it doesn't, stay SILENT about container categories
  // rather than emit a claim we cannot verify.
  try {
    const CONTAINER_CATS = /^IFC(ROOF|WINDOW|DOOR|STAIR|CURTAINWALL|RAILING)/;
    const childrenApi = typeof model.getItemsChildren === 'function';
    const allCats = await model.getItemsOfCategories([/IFC/]);
    const noGeom = [];
    for (const [cat, ids] of Object.entries(allCats)) {
      const list = ids.filter((n) => Number.isInteger(n));
      // Skip non-physical IFC entities that never carry geometry (project,
      // units, property sets, spatial containers) so the breadcrumb only
      // flags physical elements the user would expect to see.
      if (!list.length || !/^IFC(WALL|SLAB|ROOF|COLUMN|BEAM|WINDOW|DOOR|STAIR|RAILING|PLATE|MEMBER|COVERING|CURTAINWALL|FURNISHING|BUILDINGELEMENTPROXY|FOOTING|PILE|RAMP|CHIMNEY)/.test(cat)) continue;
      if (!childrenApi && CONTAINER_CATS.test(cat)) continue;
      let checkIds = list;
      if (childrenApi) {
        try {
          const kids = await model.getItemsChildren(list);
          const flat = (Array.isArray(kids) ? kids : Object.values(kids ?? {}))
            .flat(Infinity)
            .filter((n) => Number.isInteger(n));
          checkIds = list.concat(flat);
        } catch { /* children query failed — fall back to the direct ids */ }
      }
      let hasGeom = false;
      try {
        const b = await model.getMergedBox(checkIds);
        hasGeom = b && !b.isEmpty();
      } catch { hasGeom = false; }
      if (!hasGeom) noGeom.push(`${cat}(${list.length})`);
    }
    if (noGeom.length) console.warn(`render: categories with no body geometry anywhere in their subtree (they will not appear in the renders, same as in the app): ${noGeom.join(', ')} — diagnostic; verify against the PNGs before reporting`);
  } catch (e) {
    console.warn(`no-geom category summary failed: ${e}`);
  }

  // 6. Frame the camera to the building's AABB, excluding the site/terrain
  //    pad (a create_flat_site block extends ~20 m underground and dwarfs the
  //    building). `getMergedBox` returns the box in the IFC-native Z-up data
  //    frame — which, after the display rotation above, coincides with world
  //    space, so it is used directly (no extra matrix). Fall back to the
  //    whole-model box when there is no separable building set (pure building
  //    IFCs) or the merged-box query fails.
  const everythingBox = new THREE.Box3().setFromObject(group);
  let box = null;
  if (contextLocalIds.length && allIds.length) {
    try {
      const ctx = new Set(contextLocalIds);
      const buildingIds = allIds.filter((id) => !ctx.has(id));
      if (buildingIds.length) {
        const mb = await model.getMergedBox(buildingIds);
        if (mb && !mb.isEmpty()) box = mb;
      }
    } catch (e) {
      console.warn(`building merged-box failed (${e}) — framing to whole model`);
    }
  }
  if (!box || box.isEmpty()) box = everythingBox;
  if (box.isEmpty()) {
    throw new Error('empty bounding box after fragments load — 0 renderable meshes');
  }
  box.getCenter(center);
  const sz = box.getSize(new THREE.Vector3());
  const ez = everythingBox.getSize(new THREE.Vector3());
  // Per-axis min..max, NOT just spans: the z-extent includes the foundation
  // below z=0 and the roof build-up above the walls, so its span is NOT the
  // nominal wall height. The vision validator reads this stdout — a bare
  // "z=2.9" line was misread as "the 2.4m building is 2.9m tall" (#374
  // false warnings); the explicit range + label prevent that.
  console.warn(`frame bbox (whole extent incl. foundation depth + roof — NOT nominal wall height): `
    + `x ${box.min.x.toFixed(2)}..${box.max.x.toFixed(2)}, `
    + `y ${box.min.y.toFixed(2)}..${box.max.y.toFixed(2)}, `
    + `z ${box.min.z.toFixed(2)}..${box.max.z.toFixed(2)} `
    + `| whole-model spans x=${ez.x.toFixed(2)} y=${ez.y.toFixed(2)} z=${ez.z.toFixed(2)}`);
  radius = sz.length() / 2 || 5;

  // Sun in the SW sky, 40° up (W=-X, S=-Y, up=+Z), aimed at the building.
  const el = 40 * Math.PI / 180, h = Math.cos(el), dist = Math.max(60, radius * 6);
  sun.position.set(center.x - 0.707 * h * dist, center.y - 0.707 * h * dist, center.z + Math.sin(el) * dist);
  sun.target.position.copy(center);
  fill.position.set(center.x + h * dist * 0.6, center.y + h * dist * 0.6, center.z + dist * 0.5);

  // ── local visual-debugger passes (opt-in; see the `dbg` block above) ─────
  // Deliberately AFTER framing so `only=`/`hide=` don't move the camera —
  // isolating one wall should show you that wall in the SAME place it sits in
  // the full model, not re-centred.
  if (dbg.census) await elementCensus(allIds);
  if (dbg.winding) windingAudit(group);
  if (dbg.doubleside) {
    // A/B probe for the winding hypothesis: FrontSide culls back faces, so an
    // inside-out shell is invisible (and unpickable). Draw both sides and an
    // element that reappears was present all along, wound backwards.
    let n = 0;
    group.traverse((o) => {
      if (!o.isMesh || !o.material) return;
      for (const m of (Array.isArray(o.material) ? o.material : [o.material])) {
        m.side = THREE.DoubleSide; m.needsUpdate = true; n++;
      }
    });
    console.warn(`doubleside: forced THREE.DoubleSide on ${n} materials`);
  }
  if (dbg.edges) addEdgeOverlay(group);

  window.__viewCount = VIEWS.length;
  window.renderView = renderView;
  window.__ready = true;
}

/**
 * Per-element table: category, IfcRoot.Name, whether the element carries
 * geometry IN THE FRAGMENTS MODEL, and its bbox.
 *
 * This is the check that separates the two failure modes that look identical
 * on screen: "web-ifc never built the solid" vs "the solid exists but never
 * reached the scene". `getItemsGeometry` answers the first, `getMergedBox` the
 * second. Printed as one line per element so a diff between two runs (two IFC
 * versions, two library versions) is greppable.
 */
async function elementCensus(allIds) {
  try {
    const ids = allIds && allIds.length ? allIds : await model.getLocalIds();
    const data = await model.getItemsData(ids, { attributesDefault: true });
    const boxes = await model.getBoxes(ids);
    let geoms = null;
    try { geoms = await model.getItemsGeometry(ids); } catch { /* optional */ }
    const rows = [];
    for (let i = 0; i < ids.length; i++) {
      const d = data[i] ?? {};
      const name = d.Name?.value ?? d._name?.value ?? '';
      const cat = d._category?.value ?? d.category ?? '?';
      // Only physical elements — skip property sets / relationships / units,
      // which legitimately have no geometry and would bury the signal.
      if (!/^IFC(WALL|SLAB|ROOF|COLUMN|BEAM|WINDOW|DOOR|STAIR|RAILING|PLATE|MEMBER|COVERING|CURTAINWALL|FURNISHING|BUILDINGELEMENTPROXY|FOOTING|PILE|RAMP|CHIMNEY|GEOGRAPHICELEMENT|SPACE|SITE|BUILDING)/i.test(String(cat))) continue;
      const b = boxes?.[i];
      const has = !!(b && !b.isEmpty());
      const nMesh = geoms?.[i] ? geoms[i].length : -1;
      const bb = has
        ? `x[${b.min.x.toFixed(2)},${b.max.x.toFixed(2)}] y[${b.min.y.toFixed(2)},${b.max.y.toFixed(2)}] z[${b.min.z.toFixed(2)},${b.max.z.toFixed(2)}]`
        : '—';
      rows.push(`${has ? '  OK ' : ' MISS'} ${String(cat).padEnd(26)} ${String(name || '(unnamed)').padEnd(34)} meshes=${nMesh === -1 ? '?' : nMesh}  ${bb}`);
    }
    rows.sort();
    const missing = rows.filter((r) => r.startsWith(' MISS')).length;
    console.warn(`census: ${rows.length} physical elements in the fragments model, ${missing} with NO geometry\n${rows.join('\n')}`);
  } catch (e) {
    console.warn(`census failed: ${e && e.stack || e}`);
  }
}

/**
 * Triangle-winding audit over the MATERIALISED three.js geometry.
 *
 * The scene draws with `side = THREE.FrontSide`, so a shell whose triangles are
 * wound the wrong way is invisible from outside AND unpickable (three.js
 * raycasting honours `material.side`) while still owning a correct bounding box
 * and a correct entry in every data-side census. That combination is exactly the
 * "the geometry is all there but one end of the building does not draw" symptom,
 * and no data-side check can see it — this is the only place it is observable.
 *
 * Method: split each batched mesh into connected components by WELDED position
 * (fragments duplicates vertices per face, so index-adjacency alone shatters a
 * box into 6 loose quads), then integrate the divergence theorem over each
 * component: V = Σ a·(b×c)/6. A closed shell wound CCW-outward — three.js's
 * front-face convention — yields V > 0. V < 0 means the shell is inside-out.
 * Reported with the component's world bbox so an inverted component can be
 * located in the building.
 */
function windingAudit(group) {
  const Q = 1e4; // weld tolerance 0.1 mm
  const comps = [];
  group.traverse((o) => {
    if (!o.isMesh || !o.geometry || o.isInstancedMesh) return;
    const pos = o.geometry.attributes?.position;
    if (!pos) return;
    const idx = o.geometry.index;
    const n = pos.count;
    const triCount = idx ? idx.count / 3 : n / 3;

    // World-space vertex table (the display group carries a +90° X rotation, so
    // local coordinates are NOT the frame the bboxes and view labels use).
    const v = new Float64Array(n * 3);
    const tmp = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      tmp.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld);
      v[i * 3] = tmp.x; v[i * 3 + 1] = tmp.y; v[i * 3 + 2] = tmp.z;
    }

    // Weld: map every distinct quantised position to a representative index.
    const weld = new Int32Array(n);
    const seen = new Map();
    for (let i = 0; i < n; i++) {
      const k = `${Math.round(v[i * 3] * Q)},${Math.round(v[i * 3 + 1] * Q)},${Math.round(v[i * 3 + 2] * Q)}`;
      const prev = seen.get(k);
      if (prev === undefined) { seen.set(k, i); weld[i] = i; } else { weld[i] = prev; }
    }

    // Union-find over welded vertices, joined through shared triangles.
    const parent = new Int32Array(n);
    for (let i = 0; i < n; i++) parent[i] = weld[i];
    const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[ra] = rb; };
    const tri = (t, j) => (idx ? idx.getX(t * 3 + j) : t * 3 + j);
    for (let t = 0; t < triCount; t++) {
      const a = weld[tri(t, 0)], b = weld[tri(t, 1)], c = weld[tri(t, 2)];
      union(a, b); union(b, c);
    }

    // Accumulate signed volume + bbox per component.
    const acc = new Map();
    for (let t = 0; t < triCount; t++) {
      const ia = tri(t, 0), ib = tri(t, 1), ic = tri(t, 2);
      const root = find(weld[ia]);
      let e = acc.get(root);
      if (!e) { e = { vol: 0, tris: 0, min: [1e30, 1e30, 1e30], max: [-1e30, -1e30, -1e30] }; acc.set(root, e); }
      const ax = v[ia * 3], ay = v[ia * 3 + 1], az = v[ia * 3 + 2];
      const bx = v[ib * 3], by = v[ib * 3 + 1], bz = v[ib * 3 + 2];
      const cx = v[ic * 3], cy = v[ic * 3 + 1], cz = v[ic * 3 + 2];
      e.vol += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
      e.tris++;
      for (const [x, y, z] of [[ax, ay, az], [bx, by, bz], [cx, cy, cz]]) {
        if (x < e.min[0]) e.min[0] = x; if (x > e.max[0]) e.max[0] = x;
        if (y < e.min[1]) e.min[1] = y; if (y > e.max[1]) e.max[1] = y;
        if (z < e.min[2]) e.min[2] = z; if (z > e.max[2]) e.max[2] = z;
      }
    }
    for (const e of acc.values()) comps.push(e);
  });

  // Draw groups: a batched fragments mesh renders as N (start, count,
  // materialIndex) slices, NOT as one blob. A slice whose materialIndex has no
  // material behind it is silently skipped — the mesh still reports
  // visible=true with a full bbox and a full drawRange, so every per-mesh check
  // passes while part of the batch never reaches the framebuffer. Dump the
  // slices with the x-range of the vertices each one actually covers.
  group.traverse((o) => {
    if (!o.isMesh || !o.geometry || o.isInstancedMesh) return;
    const g = o.geometry.groups || [];
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    const pos = o.geometry.attributes?.position;
    const idx = o.geometry.index;
    if (!g.length || !pos) return;
    const tmp = new THREE.Vector3();
    const rows = g.map((gr, n) => {
      const end = gr.count === Infinity ? (idx ? idx.count : pos.count) : gr.start + gr.count;
      let lo = 1e30, hi = -1e30;
      for (let i = gr.start; i < end; i++) {
        const vi = idx ? idx.getX(i) : i;
        tmp.fromBufferAttribute(pos, vi).applyMatrix4(o.matrixWorld);
        if (tmp.x < lo) lo = tmp.x;
        if (tmp.x > hi) hi = tmp.x;
      }
      const mi = gr.materialIndex ?? 0;
      const m = mats[mi];
      return `    group[${n}] start=${gr.start} count=${gr.count} matIndex=${mi}`
        + ` -> ${m ? `${m.type} colour=#${m.color?.getHexString?.() ?? '?'}` : 'NO MATERIAL — SLICE IS NOT DRAWN'}`
        + `  x[${lo.toFixed(2)},${hi.toFixed(2)}]`;
    });
    // Coverage: three.js draws ONLY what a group covers once geometry.groups is
    // non-empty (drawRange is ignored on that path). Any index the slices skip
    // is geometry that is in the buffer, inside the bbox, on a visible mesh —
    // and never reaches the framebuffer.
    const total = idx ? idx.count : pos.count;
    const covered = [];
    for (const gr of g) {
      covered.push([gr.start, gr.count === Infinity ? total : Math.min(total, gr.start + gr.count)]);
    }
    covered.sort((a, b) => a[0] - b[0]);
    const holes = [];
    let cursor = 0;
    for (const [s, e] of covered) {
      if (s > cursor) holes.push([cursor, s]);
      cursor = Math.max(cursor, e);
    }
    if (cursor < total) holes.push([cursor, total]);
    const skipped = holes.reduce((n, [s, e]) => n + (e - s), 0);
    const holeTxt = holes.length
      ? holes.map(([s, e]) => {
          let lo = 1e30, hi = -1e30;
          for (let i = s; i < e; i++) {
            const vi = idx ? idx.getX(i) : i;
            tmp.fromBufferAttribute(pos, vi).applyMatrix4(o.matrixWorld);
            if (tmp.x < lo) lo = tmp.x;
            if (tmp.x > hi) hi = tmp.x;
          }
          return `      NOT DRAWN indices ${s}..${e - 1} (${(e - s) / 3} tris) x[${lo.toFixed(2)},${hi.toFixed(2)}]`;
        }).join('\n')
      : '      (full coverage)';
    console.warn(`groups: mesh with ${g.length} draw slices, ${mats.length} material(s), `
      + `${total} indices, ${skipped} SKIPPED (${skipped / 3} tris)\n${rows.join('\n')}\n${holeTxt}`);
  });

  comps.sort((a, b) => (a.min[0] + a.max[0]) - (b.min[0] + b.max[0]));
  const neg = comps.filter((c) => c.vol < -1e-9);
  const lines = comps.map((c) => {
    const f = (n) => n.toFixed(2);
    return `  ${c.vol < -1e-9 ? 'INVERTED' : c.vol > 1e-9 ? '      ok' : '    flat'}`
      + ` vol=${c.vol >= 0 ? ' ' : ''}${c.vol.toFixed(4)} m3  tris=${String(c.tris).padStart(5)}`
      + `  x[${f(c.min[0])},${f(c.max[0])}] y[${f(c.min[1])},${f(c.max[1])}] z[${f(c.min[2])},${f(c.max[2])}]`;
  });
  console.warn(`winding: ${comps.length} connected shells, ${neg.length} INVERTED (negative signed volume)\n${lines.join('\n')}`);
}

/**
 * Black edge overlay. Coplanar boolean seams (a redundant carve that removes
 * nothing, two solids sharing a face) render as extra edges that are invisible
 * in shaded views — this is how you SEE them.
 */
function addEdgeOverlay(group) {
  const mat = new THREE.LineBasicMaterial({ color: 0x111111 });
  const add = [];
  group.traverse((o) => {
    if (!o.isMesh || !o.geometry) return;
    let eg;
    try { eg = new THREE.EdgesGeometry(o.geometry, 20); } catch { return; }
    if (o.isInstancedMesh) {
      // Cap the instance expansion — an edge overlay is a debugging aid, not a
      // reason to hang the page on a 10k-instance model. Say so when capping.
      const n = Math.min(o.count, 512);
      if (o.count > n) console.warn(`edges: ${o.name || 'instanced mesh'} has ${o.count} instances — overlaying the first ${n} only`);
      const mtx = new THREE.Matrix4();
      for (let i = 0; i < n; i++) {
        o.getMatrixAt(i, mtx);
        const ls = new THREE.LineSegments(eg, mat);
        ls.applyMatrix4(o.matrixWorld.clone().multiply(mtx));
        add.push(ls);
      }
    } else {
      const ls = new THREE.LineSegments(eg, mat);
      ls.applyMatrix4(o.matrixWorld);
      add.push(ls);
    }
  });
  for (const ls of add) scene.add(ls);
  console.warn(`edges: overlaid ${add.length} edge sets`);
}

/**
 * Render one view. Async because toggling the site pad's visibility for the
 * underside view goes through the fragments worker (setVisible + update).
 * page.evaluate awaits the returned Promise, so the driver's per-view wait +
 * screenshot happen after the frame is drawn.
 */
async function renderView(stem) {
  const v = VIEWS.find((x) => x[0] === stem);
  if (!v) throw new Error(`unknown view stem: ${stem}`);

  // Underside (zneg): hide the site pad so the building base isn't occluded.
  const hideSite = stem === 'zneg';
  if (contextLocalIds.length && hideSite !== contextHidden) {
    try {
      await model.setVisible(contextLocalIds, !hideSite);
      await fragments.core.update(true);
      contextHidden = hideSite;
    } catch (e) {
      console.warn(`setVisible(context) failed for ${stem}: ${e}`);
    }
  }

  const u = new THREE.Vector3(v[2][0], v[2][1], v[2][2]).normalize();

  // `target=x,y,z` re-aims the camera without re-framing — point it at a
  // corner to inspect one miter joint at close range.
  let at = center;
  if (dbg.target) {
    const t = dbg.target.split(',').map(Number);
    if (t.length === 3 && t.every(Number.isFinite)) at = new THREE.Vector3(t[0], t[1], t[2]);
    else console.warn(`target=: expected "x,y,z", got "${dbg.target}" — using the model centre`);
  }

  const fit = radius / Math.tan((60 * Math.PI / 180) / 2) * 1.25 * dbg.dist;
  const active = dbg.ortho ? ortho : cam;
  active.up.set(0, 0, 1);
  if (Math.abs(u.z) > 0.99) active.up.set(0, 1, 0); // top/bottom: pick a valid up
  if (dbg.ortho) {
    // `span=` is the ortho HALF-WIDTH in metres; without it, frame the model.
    const h = dbg.span > 0 ? dbg.span : radius * 1.25;
    ortho.left = -h; ortho.right = h; ortho.top = h; ortho.bottom = -h;
    ortho.updateProjectionMatrix();
  }
  active.position.copy(at.clone().addScaledVector(u, fit));
  active.lookAt(at);
  document.getElementById('cap').textContent = v[1];
  renderer.render(scene, active);
}

build().catch((e) => { window.__error = String((e && e.stack) || e); });
