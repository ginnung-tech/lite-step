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

Python 3.12 or newer, in a virtual environment (README, "Start"). From the
repo root, `pip install -e .` installs the dependencies and the `lite-step`
command, and makes `import lite_step` work from any folder. Without the
install, work from the repo root: that is what puts `lite_step` on the path.

```bash
pip install -e .                  # plus pytest, ruff and ty to run the checks below

ruff check output.py              # lint — fix before compiling
ty check output.py                # types — when a signature changed
python output.py --check          # DRY RUN: validate + compile, write nothing
python output.py --q <selector>   # INSPECTOR: what is this, and where is it?
python output.py                  # compile -> output.ifc
lite-step review output.py        # REVIEW: compile, IFC validation, defects found by geometry -> review/
```

Done means **a clean compile, a non-empty `output.ifc` on disk, and `lite-step
review` with no open error group** — not code that looks right. Read the error,
not the symptom: stderr names the element.

`lite-step review` (from this checkout; it needs nothing beyond Lite-STEP's own
requirements) compiles the model, runs `ifcopenshell.validate`, and
measures the geometry: **floating** parts (a group touching nothing that reaches
the ground), **clashes** (two elements that are not parts of one thing share
volume; a linear element set into another at its end is a joint, not a clash),
**near misses** (a column, beam or member 5-150 mm short of meeting) and meshes
that are not closed solids. It writes `review/findings.md` (read this) and
`review/findings.json`, and exits 1 while an error group is open. Findings are
grouped by kind and IFC classes, name elements by canonical name (pass them to
`--q`), and carry their numbers. Fix the cause in `output.py`, or, when the
contact is meant (a course laid inside its pavement), accept the group with the
reason in `review/accepted.json`:
`[{"group": "clash:IfcCourse|IfcPavement", "reason": "the course is a layer of the pavement"}]`.
Never accept a floating group: something holds it up, so model it.

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
   `numpy` and `random`; there is no file I/O. The one sanctioned read is a cleaned mesh named by path and
   hash, `Mesh(asset="meshes/statue.ply", sha256="...")`: a scan belongs in a file, not in the script. A host may install a stricter
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

## 7. Other skills: Heimdal, the minions, the validators

Everything above is about writing the model. These skills check it, and none of them edits `output.py`.

| Skill | Use it when | Writes |
| --- | --- | --- |
| [`agents/heimdal/SKILL.md`](agents/heimdal/SKILL.md) | someone gives pictures, drawings or notes the model should match, or asks how well it does | `survey/` only |
| [`agents/reconstruction/SKILL.md`](agents/reconstruction/SKILL.md) | someone brings video, drone footage, a photo set or a scan of the real thing | suggestions only: what to do outside Lite-STEP, and how cameras and clean meshes come back in |
| [`agents/validation/ifc/SKILL.md`](agents/validation/ifc/SKILL.md) | a build is done and there is a `spec.md` to hold it to | `validation-errors.md`, `summary.txt` |
| [`agents/validation/vision/SKILL.md`](agents/validation/vision/SKILL.md) | the same, by looking: six rendered views against the spec | `render_*.png`, `validation-errors.md`, `summary.txt` |

A `spec.md` follows [`agents/spec-template.md`](agents/spec-template.md).

**Running as Heimdal.** Read the whole skill first. The library is `tools/survey`, run from this repo's root
with `--workdir` at the project (`python -m tools.survey …`; `tools/` is not installed by `pip install -e .`).
Install its extras once: `pip install -r tools/survey/requirements.txt`. One round on a request:

1. `python -m tools.survey plan begin --workdir P --request R`: prints the round, `maxRounds` and the rules
   for this round, and keeps the plan and the survey as they were.
2. Take pictures in (`add`), fit, measure (`measure`, `inspect`), as the skill says. Write `survey/plan.json`.
3. `python -m tools.survey plan check --workdir P`: the same checks a host runs. Exit 1 lists what breaks a
   rule; fix the plan and check again.
4. `python -m tools.survey plan record --workdir P` appends the round to `iterations.jsonl` (refused while a
   rule is broken), then `python -m tools.survey index --workdir P` last.

Then the decision: `instruct` is one change for whoever builds the model (`plan.json` → `decision.instruction`,
with its `checkSpec`); build it, compile, and start the next round with `plan begin` again. `ask` is a question
for the person; `converged` needs a passed check measured on the person's own pictures this round.

The traps: a picture that is not lined up measures nothing (ask, do not converge); "left to right" in a photo
is not an axis until its camera is fitted; counts and sizes come from geometry, never from looking.

**Building something from pictures of it is a Heimdal job from the first round, not after.** A sculpture, a
vault, a facade: take the pictures in, fit them (drawings with `--view`), and measure each version with
`python -m tools.survey shape`, which gives the outline's error per height and side in millimetres and the
elements at fault. Comparing your own renders with the photos by eye does not converge: a statue built that way
looked roughly right in perspective and measured IoU 0.69 against its elevation drawings; measured and
corrected (lean, datums, and a scan for the skin) it reached 0.87-0.89 on all four.

MIT licensed.
