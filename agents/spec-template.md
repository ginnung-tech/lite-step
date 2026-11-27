# Project Specification

## Overview
[One paragraph: type, size, storeys, key features]

## Site
[Location, orientation, constraints]

## Storeys
[For each: elevation, height, function]

## Room Schedule
[For each room: name, storey, approximate dimensions, function]

## Structural System
[Wall types, load paths, special conditions]

## Plan (current build)
[Written by the planning pass (Agent Loop 5a), one block per `plan.jsonl` task,
same order. This section IS the task descriptions — the task loop reads its
block here before touching code. Replaced wholesale by each build's planning
pass; never edited afterwards.]

### Task N: [label — verbatim from plan.jsonl]
- **Intent:** [the user's ask for this part, rewritten concretely]
- **Variables:** [`name = value` pairs — ONLY names from the AGENTS.md variable
  index; user-given numbers override defaults. Omit line if none apply.]
- **System:** [`skills/lod<lod>/systems/<name>/SKILL.md` — the ONE
  skill this task reads, or "none (DSL reference)"]
- **Notes:** [anything that is not a named variable: placement, adjacency,
  material choices, carve direction]

## Open Questions
[Unresolved items]

## Assumptions Made
[Every default applied]

## Build History
[Chronological log, one appended line per build: what was generated, what was changed]