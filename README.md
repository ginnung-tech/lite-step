# Lite-STEP

A strict Python DSL for buildings that compiles to **IFC4X3 STEP**.

You describe a building as ordinary Python objects. The compiler resolves
placement, infers the boolean carves between solids, and writes a valid IFC
file. No viewport, no dragging, no foreign CAD import — you model in code, and
the model is the code.

```python
from lite_step import compile_main
from lite_step.models import Project, Storey, Wall, Box, Point, Material

def generate_project():
    proj = Project(name="shed")
    ground = Storey(name="ground", elevation=0)
    wall = Wall(name="north")
    wall.add(Box(name="body",
                 start=Point(x=0, y=0, z=0),
                 end=Point(x=4000, y=200, z=2400),
                 material=Material(key="Brick_Red_DK")))
    ground.add(wall)
    proj.add_storey(ground)
    return proj

result = generate_project()

if __name__ == "__main__":
    compile_main()
```

## Start

```bash
pip install -r requirements.txt
python output.py                    # → output.ifc beside it
```

`output.py` at the repo root is the seed you edit. Work **from the repo root** —
that is what puts `lite_step` on the import path. For any other file, use
`python -m lite_step <file>.py`.

## Modelling

**Coordinates are integer millimetres, Z-up.** A float where an int belongs is a
refusal, not a rounding — hence the `I = lambda x: int(round(x))` helper the seed
carries.

**`.add()` places in world coordinates; `.anchor()` places relative to the
host's frame.** That one distinction decides which element defines a frame,
which carves which, and where a child ends up.

**`.add()`ed solids carve, and tree order decides who carves whom** — later
carves earlier. A solid wholly inside a later one is carved to nothing; that
warns loudly and names the fix. `.add(child, carve="none")` opts out.

**Names are lowercase leaves; the compiler derives the path.**
`Window(name="left")` inside `Wall(name="north")` becomes
`window:left:wall:north`. You match it by suffix, never write it.

The complete authoring API — every model class, every field, every coordinate
convention — is [`agents/dsl-reference.md`](agents/dsl-reference.md).

## Seeing the model

```bash
python output.py --check            # compile + validate, write nothing
python output.py --q wall:north     # where is it, really: authored / world / oriented boxes
```

`--q` takes a SUFFIX of the canonical name, so once the storey is named ask for
`wall:north:storey:ground`. The IFC it writes opens in any IFC viewer.

## Worked examples

[`examples/`](examples/) builds the same 20-block chain four ways — by hand,
by `Anchor`, by `Product`, then by `Product` hosted as a real window opening.
Read in order, the diff between consecutive files is the lesson. Two further
scripts stand apart from the chain: a procedurally stacked log cabin and a
watertight silver spoon, both built entirely from `Mesh`.

Beyond the chain: a procedural log cabin with mitred joinery, a masonry villa,
a steel lattice bridge, hospital massing, and chainable terrain mesh operations.

## Running untrusted models

`lite_step` executes author-written Python. It ships with a restricted
namespace — imports are limited to `lite_step`, `math`, `numpy` and `random` —
but **no sandbox**; that is the embedding application's decision, and there is a
seat for it at
[`lite_step/compiler/sandbox.py`](lite_step/compiler/sandbox.py).

Running your own models, this does not matter. Running someone else's, install
hooks via `set_sandbox()` and set `LITE_STEP_REQUIRE_SANDBOX=1` so an empty seat
is a loud refusal rather than unguarded execution.

## Agents

[`AGENTS.md`](AGENTS.md) is the working guide for coding agents — where the
skills are, the rules that bite, the edit loop, and the debug tooling.

MIT licensed.
