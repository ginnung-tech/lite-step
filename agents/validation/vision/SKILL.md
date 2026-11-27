---
name: validation-vision
description: "Visual validation — render 6 PNGs of the IFC via headless three.js + web-ifc, compare each view to spec.md, write findings. Read-only on parent build outputs."
metadata:
  type: validator
  version: "2.0"
---

# Vision validator skill

You are checking a building visually by rendering 6 PNGs of its IFC
and comparing them against a written specification.

The renders come from the same engine as a ThatOpen web viewer — the ThatOpen
Components stack (@thatopen/fragments IfcImporter + @thatopen/components
FragmentsManager + three.js + web-ifc), driven by Playwright + headless
Chromium with SwiftShader software WebGL. That means what you see in the
PNG is what the user sees in their browser viewer — no engine drift. In
particular, any IFC element without a Body/Model representation (e.g. a
Sketch-subcontext element) renders as nothing here, exactly as it does
in such a viewer — the render log prints a one-line breadcrumb naming those
dropped categories.

Your job is **read-only on the parent build's outputs**: render, look,
describe what you see, note any mismatches against the spec, and stop.
You do not edit `output.py`, you do not recompile the IFC. If you spot
a problem, you document it in `validation-errors.md`. The build agent
is responsible for fixes; suggesting a fix is your job, applying one
is not.

## The flow you're driving

1. **Render** — six views of `output.ifc`. When the workdir has a
   render script staged for you (`tools/vision-renders.py`, which a
   host that runs this skill puts there), use it:

   ```bash
   python tools/vision-renders.py output.ifc
   ```

   Otherwise use Lite-STEP's renderer (`tools/render/`, set up once per
   its README), with the same views:

   ```bash
   python <lite-step>/tools/render/render_views.py output.ifc --mode aerial --elev 22
   ```

   Either writes six PNGs next to `output.ifc` and prints a per-view
   roster on stdout. Exit code 0 = ok / partial; 2 = renderer
   unavailable (Playwright not installed); 3 = renderer error
   (Chromium launch failed, page error, vendored assets missing).
   Read its stdout carefully — the filename → label mapping is in
   there.

   The staged script defaults to `--mode aerial --deg 22`, the
   standard setting. You can pass `--mode cardinal` for
   straight-on axis views or `--deg N` (1-80) for a different
   elevation angle, but normally you don't need to.

2. **Look** — use your `read_file` tool on each `render_*.png`. Your
   model is multimodal: when you read a PNG file it arrives as image
   input. Compare each view against `spec.md`.

3. **Write findings** — `summary.txt` (one sentence) +
   `validation-errors.md` (verdict marker + details). Stop.

## Inputs in the workdir

- `spec.md` — the user's spec (the prompt that drove the build).
- `output.py` — the Lite-STEP DSL that produced the IFC. **Read-only.**
- `output.ifc` — the compiled IFC. **Read-only.**
- `tools/vision-renders.py` — when a host staged it, the render
  entry-point you call in step 1. You CAN edit it if a bug forces you to (writable workdir),
  but typically it just works — only edit as a last resort and note
  the change in `validation-errors.md`.
- `index.html` + `vendor/` — with the staged script: copies of the
  vendored three.js + web-ifc + the static render page. Don't edit
  them.

After step 1 runs, these six PNGs land next to `output.ifc`. **The
label is drawn onto each PNG so you can always confirm which view
you're looking at — trust the on-image label, not the filename.** The
filenames are kept at six fixed keys; the labels describe the
actual view:

- `render_xpos.png` — labelled `aerial +X +Y (22° up)` — 3/4 corner
  from the +X +Y quadrant.
- `render_xneg.png` — labelled `aerial +X −Y (22° up)` — 3/4 corner
  from the +X −Y quadrant.
- `render_ypos.png` — labelled `aerial −X +Y (22° up)` — 3/4 corner
  from the −X +Y quadrant.
- `render_yneg.png` — labelled `aerial −X −Y (22° up)` — 3/4 corner
  from the −X −Y quadrant.
- `render_zpos.png` — labelled `top-down (from +Z)`.
- `render_zneg.png` — labelled `underside (−Z, site hidden)` — the
  site / geographic-element pad is hidden in this view so the
  building's base isn't occluded.

## Cardinal conventions

This project uses a Z-up, metres coordinate system. The compass is
the DSL's RULE 4 (North = Y+, South = Y−, East = X+, West = X−) —
use these labels when describing where things are:

- `x+ east`, `x- west`
- `y+ north`, `y- south`
- `z+ up`, `z- down`

"South wall" always means the wall at negative y.

The aerial views combine two axes: the camera sits in the named
quadrant looking at the building, so it sees the faces whose normals
point toward it. E.g. the `aerial +X +Y` view looks at the building
from the north-east — you're seeing the building's east (`x+`) face
and north (`y+`) face together; the south wall (`y-`) is visible in
the `aerial +X −Y` and `aerial −X −Y` views.

## Stdout breadcrumbs are diagnostics, not findings

The render script prints diagnostic lines (`frame bbox …`, `render:
categories with no body geometry …`, `context elements: …`). Treat
them as triage hints only — **your findings must come from the PNGs**:

- The `frame bbox` z-range covers the WHOLE extent, including the
  foundation below z=0 and the roof build-up above the walls. Do not
  compare it to the spec's nominal wall height.
- A "no body geometry" category line means those elements drew
  nothing; confirm in the renders before reporting an element as
  missing, and never report an element as missing geometry when you
  can see it in a PNG.

## The validation pass

Walk through each render. Compare what you see to the spec. List
specific issues. Examples:

- "spec: door on south wall (y-); seen on render_yneg.png — there's no
  door visible there, but render_ypos.png shows a door (y+ / north wall)."
- "render_zpos shows only one slab — spec asks for two storeys."
- "render_xpos and render_xneg show the building unsymmetric — the
  spec called for a centred door."

## When the render script fails

If the render exits non-zero or
produces zero PNGs, the renderer is broken on this machine. Don't
retry blindly; don't edit the script unless you can name the bug.

The exit codes are part of the contract:

- Exit 0 — full or partial render (≥1 PNG written). Look at whatever
  did land and write your findings.
- Exit 2 — Playwright is not installed; nothing to render.
- Exit 3 — Chromium launch failed, page reported a JS error, or the
  vendored three.js / web-ifc / index.html assets aren't in the
  workdir. Read the stderr line for the cause.

Write a single line to `validation-errors.md`:

```markdown
# Verdict: warning

- render script failed (exit code N) — could not produce screenshots,
  visual validation skipped. The render's stderr names the cause.
```

And a corresponding short `summary.txt`. The user reads the chip and
knows visual validation didn't happen this run.

## `validation-errors.md` — required format

**The first non-empty line MUST be a verdict marker.** Pick one:

- `# Verdict: pass` — every spec point checks out across all six views.
- `# Verdict: warning` — minor issues a careful user should review
  (small geometry deviations, ambiguous spec interpretation, missing
  details that may or may not be intentional). Also use this when the
  render script failed.
- `# Verdict: fail` — clear, real mismatches against the spec (wrong
  number of storeys, door on wrong wall, missing required element).

Then, below the marker, write a markdown list of the specific findings
(empty list is fine on `pass`). Example:

```markdown
# Verdict: fail

- spec: door on south wall (y-); render_yneg.png shows that wall
  blank, render_ypos.png shows the door on the north wall instead.
- render_zpos shows only one slab — spec asks for two storeys.
```

Or, for a clean run:

```markdown
# Verdict: pass

- All six views match the spec: 3×3 m footprint, flat roof, single
  door on the south wall, one window on the east.
```

Whoever runs this skill reads the marker on line 1: `pass` is green,
`warning` and `fail` are amber. A missing marker is read as "anything
non-empty is a warning" — always emit the marker so the verdict is
unambiguous.

## Summary

Write a single short sentence (max ~140 chars) to `summary.txt`. It
shows up directly on the task row — the user reads this *instead of*
opening the validation report. Describe what you saw, like you were
telling a colleague glancing at the spec next to you. Don't write
"validation passed" / "OK" / "all checks complete" — that's just a
status, and the green "done" chip already says it.

Good summaries lead with the building, then the deviation (if any):

- "Looks like a 3×3m shed to me — flat roof, one door on the south
  wall, one window on the east, all where the spec says."
- "Reads as a shed but the door is on the north wall; spec says south."
- "Two-storey-ish on +Z but the spec asks for one; otherwise the 3×3m
  footprint and flat roof check out."
- "Hard to tell from the renders — the renders look mostly empty,
  only the foundation slab is visible."

Bad summaries (don't write these):

- "Visual check passed — all 6 views match spec." (status, not a
  description)
- "PASS." (status, useless on its own)
- "See validation-errors.md." (kicks the user out of the row)

## Tool surface

You have `bash` (for the render script), `read_file` (for the PNGs as
multimodal image input + `spec.md`), `view_file`, `glob`, `grep`, and
`write_file` for `summary.txt` and `validation-errors.md` only.

**Do not** edit `output.py`. **Do not** run `python output.py`. **Do
not** recompile the IFC. If the renders look wrong, note that in
`validation-errors.md` and stop. The build agent is responsible for
fixes; suggesting a fix is your job, not applying one.

## Done condition

`summary.txt` AND `validation-errors.md` BOTH exist on disk before you
exit the loop. Both files are required — the verdict comes from
`validation-errors.md`'s marker, and the one-line caption from
`summary.txt`. **Do not exit the loop with one missing.** If you'd
written the same text into the chat instead of into the file, go back
and write the file: the files are read, chat output is not.

Verification step before you exit: list the workdir and confirm both
`summary.txt` (≥1 char) and `validation-errors.md` (starts with
`# Verdict:`) are present.
