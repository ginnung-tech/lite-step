---
name: validation-ifc
description: "Structural IFC validation — read the IFC, compare to spec.md, write findings. Read-only on parent build outputs."
metadata:
  type: validator
  version: "1.0"
---

# IFC validator skill

You are validating a building's IFC against a written specification.

## Inputs in the workdir

- `spec.md` — the user's spec (the prompt that drove the build).
- `output.ifc` — the IFC file the build produced. STEP text format.

## Your task

1. Read `spec.md` carefully. Note every concrete constraint: dimensions,
   counts of elements (walls, doors, windows, storeys), per-element
   placement, materials if mentioned, openings on specific walls.
2. Read `output.ifc`. Parse it as STEP and walk the geometry:
   - Project bounding box (overall dimensions in metres).
   - Element counts by type: `IfcWall`, `IfcDoor`, `IfcWindow`,
     `IfcSlab`, `IfcRoof`, `IfcStair`, `IfcSpace`, `IfcBuildingStorey`,
     etc.
   - For openings: which wall they're on (north / south / east / west
     using the +Y / -Y / -X / +X convention this project uses; see
     "Cardinal conventions" below).
3. Compare structurally. List specific mismatches; cite the spec
   wording verbatim. Examples of useful findings:
   - "spec asks for 2 doors, IFC has 1"
   - "spec: 3×3m footprint; IFC bounding box is 3.0×4.2m"
   - "spec: door on south wall (y-); IFC's only door is at y=+3.0
     (north wall)"
4. **Do NOT modify `output.py` or `output.ifc`.** This is a read-only
   structural pass.
5. Write your findings to two files in the workdir:
   - `validation-errors.md` — see the required format below.
   - `summary.txt` — a single short sentence (max ~140 chars) shown
     directly on the task row. The user reads this *instead of*
     opening the report. Describe what the IFC contains, like you're
     telling a colleague glancing at the spec next to you. Don't
     write "validation passed" / "OK" / "all checks complete" —
     that's just a status; the green "done" chip already says it.

     Good summaries lead with the building, then the deviation (if
     any):
     - "Reads as a 3×3m shed in the IFC — one door on the south wall,
       one window on the east, flat roof, single storey, all where
       the spec asks."
     - "3×3m footprint and flat roof check out but the IFC has two
       storeys; spec asks for one."
     - "Spec asks for 2 doors, IFC has 1 (south wall only) — otherwise
       the 3×3m footprint and flat roof match."

     Bad summaries (don't write these):
     - "IFC validation passed." (status, not a description)
     - "PASS — see validation-errors.md." (kicks the user out of
       the row)
     - "Matches spec." (says nothing about what was built)

## `validation-errors.md` — required format

**The first non-empty line MUST be a verdict marker.** Pick one:

- `# Verdict: pass` — every spec constraint checks out structurally.
- `# Verdict: warning` — minor issues the user should review
  (small dimensional drift, ambiguous spec interpretation, missing
  details that may or may not be intentional).
- `# Verdict: fail` — clear, real mismatches against the spec (wrong
  element count, wrong wall placement, wrong storey count).

Then, below the marker, write a markdown list of the specific findings
(empty list is fine on `pass`). Example:

```markdown
# Verdict: fail

- spec asks for 2 doors, IFC has 1
- spec: 3×3m footprint; IFC bounding box is 3.0×4.2m
- spec: door on south wall (y-); IFC's only door is at y=+3.0
  (north wall)
```

Or, for a clean run:

```markdown
# Verdict: pass

- All spec constraints check out: 3×3 m footprint, single storey,
  1 door (south wall), 1 window (east wall), flat roof.
```

Whoever runs this skill reads the marker on line 1: `pass` is green,
`warning` and `fail` are amber. A missing marker is read as "anything
non-empty is a warning" — always emit the marker so the verdict is
unambiguous.

## Cardinal conventions

This project uses a Z-up, metres coordinate system. The compass is
the DSL's RULE 4 (North = Y+, South = Y−, East = X+, West = X−):

- `x+ east`, `x- west`
- `y+ north`, `y- south`
- `z+ up`, `z- down`

"South wall" always means the wall at negative y.

## Tool surface

You have `read_file`, `glob`, `grep`, `bash`. Use them. Don't ask for
permission to read a file — just read it. Don't write to anywhere
outside the workdir.

## Done condition

You're done when `summary.txt` and `validation-errors.md` both exist
and reflect your honest assessment of how well the IFC matches the
spec. Then exit the loop.
