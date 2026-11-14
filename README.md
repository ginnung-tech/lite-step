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

You need **Python 3.12 or newer** ([python.org](https://www.python.org/downloads/);
on Windows, tick *Add python.exe to PATH* in the installer) and **git**.

**1. Get the code.** On this repository's GitHub page, *Code* gives its
address; then:

```bash
git clone <that address>
cd lite-step
```

**2. Make a virtual environment** (once). It keeps Lite-STEP's packages apart
from the rest of your computer.

```bash
python -m venv .venv
```

**3. Turn it on** (every time you open a new terminal). The prompt then starts
with `(.venv)`.

| Shell | Command |
|---|---|
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows Command Prompt | `.venv\Scriptsctivate.bat` |
| macOS, Linux | `source .venv/bin/activate` |

PowerShell may refuse the first one with *running scripts is disabled on this
system*. Allow scripts you made yourself, once, then try again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

**4. Install Lite-STEP and its dependencies** (once, and again after a
`git pull` that changes `requirements.txt`):

```bash
python -m pip install --upgrade pip
python -m pip install -e .
```

The `-e` keeps the package as this folder, so `git pull` updates it with no
reinstall.

**5. Build the seed model**

```bash
lite-step output.py                 # → output.ifc beside it
```

`output.py` at the repo root is the seed you edit; any model file works the
same way (`lite-step path/to/house.py`), from any folder. `python output.py`
and `python -m lite_step <file>.py` do the same.

In VS Code, pick the environment once: *Python: Select Interpreter* → the one
in `.venv`. New terminals then turn it on by themselves.

| You see | It means |
|---|---|
| `lite-step: command not found` / `is not recognized` | The environment is not on: step 3 |
| `ModuleNotFoundError: No module named 'lite_step'` | Step 4 was run in another environment, or not at all |
| `No matching distribution found for numpy` (or `ifcopenshell`) | Your Python is older than 3.12, or 32-bit: `python --version`, then make `.venv` again with a newer one |

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

**Parameters are the values a person will want to change.** Write them as
`ROOF_PITCH_DEG = param(45, "deg", "Roof pitch")` (`from lite_step import
param`): the script still sees `45`, and tools can list and edit it. Lapin
shows them as project parameters and writes an edit back into that one line
(see [`tools/`](tools/)). Write the assignment at module
scope with a plain literal (a computed value is shown but cannot be edited),
and choose values the model really is built from: a parameter must still
compile when it is changed. A handful per model, not every coordinate; the
examples show how many.

**GlobalIds are stable.** Each named element's GlobalId is derived from its
canonical name and the project's name (an unnamed one's from what it belongs to), so compiling the same model again gives the same
ids, and tools that keep hold of an element (a Lapin link, a BCF topic) keep
it. Renaming or moving an element changes its path, and so its id.

**A unit you repeat is a `Product`.** Its occurrences share one body in the
IFC (`IfcRepresentationMap`), and the compiler decides which occurrences can
share before it writes them: a 52-storey tower's 1,591 facade modules are 9
bodies. Occurrences placed with `placement=` or hosted as openings are grouped
cheaply; one baked in by `.anchor()`, or carved differently from its
siblings, is compared in full.

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

## Tools & Extensions

See [`tools/`](tools/) for editor and CAD tooling:
- **VS Code Extension**: Compiler triggers, in-editor 3D Webview preview, and live push sync.
- **Blender Add-on**: WebSocket bridge syncing IFC models into Blender 5.0+ with Bonsai and presentation PBR rendering.
- **Lapin**: `lite-step lapin push model.py` links the compiled model into [Lapin](https://github.com/Ciuntuc-BIM-Studio/lapin), an open-source desktop BIM modeller, and reloads it on the next push; parameters edited in Lapin come back into the script.

## Agents

[`AGENTS.md`](AGENTS.md) is the working guide for coding agents — where the
skills are, the rules that bite, the edit loop, and the debug tooling.

MIT licensed.
