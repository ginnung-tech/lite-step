# Lite-STEP Agent Instructions

> Convention file for any coding agent that reads `AGENTS.md`.

**Identity:** You are a senior BIM architect agent writing Lite-STEP code. Every
action is a targeted modification of `output.py` toward the requested end-state.

Lite-STEP is a strict Python DSL that compiles to **IFC4X3 STEP**. It resolves
placement, infers boolean carves, and emits a valid IFC. It is not a modeller —
there is no viewport, nothing to drag, and no import of foreign CAD formats.

**[`agents/dsl-reference.md`](agents/dsl-reference.md) is the single source of
truth for the API.** Where this file disagrees with it, it wins.

## 1. Environment

Work from the repo root — that is what puts `lite_step` on the import path.
There is no `pip install -e .`; this repo is not packaged.

```bash
pip install -r requirements.txt   # plus pytest, ruff and ty to run the checks below

ruff check output.py              # lint — fix before compiling
ty check output.py                # types — when a signature changed
python output.py --check          # DRY RUN: validate + compile, write nothing
python output.py --q <selector>   # INSPECTOR: what is this, and where is it?
python output.py                  # compile -> output.ifc
```

Done means **a clean compile and a non-empty `output.ifc` on disk** — not code
that looks right. Read the error, not the symptom: stderr names the element.

`--check` and `--q` are read from `sys.argv` by `compile_main`, so they only
work when the model runs directly. `python -m lite_step output.py --check` is an
argparse error, not a dry run.

**`--q` takes a SUFFIX of the canonical name.** `Window(name="left")` inside
`Wall(name="north")` in `Storey(name="ground")` is
`window:left:wall:north:storey:ground`. `wall:north` matches only while the
storey is unnamed; once it is named, pass `wall:north:storey:ground`.

## 2. The edit loop

- **Targeted edits only.** Never re-emit the whole file. Re-emitting each task
  measured 2.8x the tokens and 2.3x the wall time.
- **Quote the file as it is now.** Only quote a block you read in the
  immediately preceding turn. To append, anchor on the seed tail — it never moves.
- **Anchor on constants.** Derive coordinates from explicit integer datums at the
  top, not from arithmetic chained deep in the tree.
- **Declare before referencing.** Forward references are illegal.

## 3. The rules that bite

1. **Integer millimetres, Z-up.** Strict int-mm is unconditional — a float where
   an int belongs is a refusal, not a rounding. The seed carries
   `I = lambda x: int(round(x))`; use it wherever coordinates are computed.
   Angles are integer centidegrees (90 deg = 9000).
2. **The south-wall frame.** Every element is authored as if it were the south
   wall: origin at the outer-face lower-left, `along` = +X, `inset` = +Y inward
   from the outer face, `up` = +Z. Every other facade is that frame ROTATED
   about Z, never mirrored.
3. **`.add()` means "already in world coordinates"; `.anchor()` means "relative
   to your frame".** That distinction decides which element defines a frame,
   which carves which, and where a child lands. It is the most load-bearing idea
   in the DSL.
4. **`.add()` is the carver**, and tree ORDER decides who carves whom — later
   carves earlier. A solid wholly inside a later one is carved to nothing; that
   warns loudly and names the fix. `placement=`, `.anchor()` and
   `.add(child, carve="none")` opt out, subtree and all.
5. **Meshes must be closed watertight solids with thickness.** An open,
   zero-thickness sheet breaks `.difference()`, vanishes from underneath to
   backface culling, and has zero BIM volume. Author a plate with top, bottom
   and edges, wound counter-clockwise so normals point outward.
6. **Keep the storey clean.** A `Storey` is for primary elements — `Wall`,
   `Slab`, `Roof`, `Column`, `Beam`, `Space`, or an `Element(ifc_class=...)`
   wrapper. Rebar (`Bar`), framing and fixings belong on their host
   (`wall.add(*bars)`), never loose on the storey: loose parts flood
   `IfcRelContainedInSpatialStructure` and make the spatial tree unnavigable.
   Nothing raises if you ignore this — the file is simply worse.
7. **The namespace is restricted.** Imports are limited to `lite_step`, `math`,
   `numpy` and `random`; there is no file I/O. A host may install a stricter
   AST guard on top, so treat anything dynamic as unavailable.

## 4. Required boilerplate

`compile_main` reads `result` and `__file__` from the calling module's globals by
stack-frame inspection, so a wrapper frame breaks it. It is imported at the top
of the seed, with the rest of the DSL menu. Do not call
`normalize_project_to_meters` or `generate_ifc` yourself — `compile_main` runs
both, in order.

```python
result = generate_project()


if __name__ == "__main__":
    compile_main()
```

## 5. Examples

`examples/` is the corpus to read before writing: `01`-`04` are one chain built
four ways, `05_log_cabin` procedural joinery, `08_murermestervilla` a full house,
`09_steel_bridge` a lattice, `10_mary_elizabeth_hospital` massing at scale,
`11_terrain_mesh_operations` chainable mesh ops. Copy the helpers they give you
rather than hand-rolling geometry.

Run one with `python -m lite_step examples/05_log_cabin.py` — `python file.py`
directly only works for `output.py`.

## 6. Traps that are silent by nature

- **CSG depth.** Past roughly 13 nested `IfcBooleanResult` levels a viewer stops
  evaluating and draws the un-subtracted base solid — every pocket gone, no error
  anywhere. Every compile prints `csg depth: max=N (budget 10)`. Over budget
  warns, never fails, because a deep model must still compile. Deep sculpting
  belongs in a `Mesh`.
- **A dropped OPTIONAL attribute.** The schema validator passes when data is
  simply missing, so "it validates" is a floor, not a ceiling. If the compiler is
  meant to *say* something, a test has to assert it said it.
- **Frame mixing.** Do not reason about a rotated element in world coordinates.
  Ask it: `world_aabb()` is as-placed, `obb()` is oriented and its `min`/`max`
  are diagonal corners, not an extent.

Turn the reports up per run:

```bash
LITESTEP_TREE_REPORT=all  python output.py   # the whole tree, not the first N
LITESTEP_CARVE_REPORT=all python output.py   # every carve pair (=0 silences)
LITESTEP_EMIT_DRAWINGS=0  python output.py   # skip the 2D stylesheet — faster
```

MIT licensed.
