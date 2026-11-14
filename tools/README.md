# Lite-STEP Tools & Extensions

This directory contains editor and CAD extensions for working with **Lite-STEP**:

- **[Blender Extension](blender/)**: Live synchronization bridge for Blender (5.0+) with Bonsai and PBR presentation rendering.
- **[VS Code Extension](vscode/)**: Authoring, compilation, 3D preview, and multi-target push synchronization for Lite-STEP DSL files (`.py`).
- **[Lapin](lapin/)**: Lite-STEP models in [Lapin](https://github.com/Ciuntuc-BIM-Studio/lapin), an open-source desktop BIM modeller: push a model in, and keep it up to date as the script changes.

---

## 1. Blender Extension (`tools/blender`)

An extension for Blender 5.0+ that runs a local WebSocket server (`ws://127.0.0.1:9876`).

### Capabilities
- **Real-Time Sync**: Listens for incoming IFC models pushed from VS Code or web clients.
- **Bonsai Integration**: Automatically loads received IFC models into the active scene via Bonsai.
- **Presentation Shell (`model.blend`)**: Automatically locates and opens Lite-STEP's presentation shell (`lite_step/assets/model.blend`), binding PBR master shaders, environmental lighting, and floor plan CSS stylesheets.

### Installation
1. Package the extension:
   ```bash
   python tools/build.py blender
   # Generates tools/blender-litestep.zip
   ```
2. Open Blender (5.0+).
3. Navigate to **Edit** > **Preferences** > **Get Extensions**.
4. In the upper-right corner dropdown menu (`⌵`), select **Install from Disk...**
5. Select `tools/blender-litestep.zip` and enable the extension.
6. Find the **Lite-STEP** panel in `View3D > N-Panel > Lite-STEP`.

---

## 2. VS Code Extension (`tools/vscode`)

A rich language and tooling extension for authoring buildings in Python with Lite-STEP.

### Capabilities
- **Lite-STEP Compilation**: Compiles Lite-STEP Python scripts directly into valid IFC4X3 models with automatic environment resolution.
- **3D Preview**: In-editor Webview panel rendering models with ThatOpen Components and Web-IFC WASM.
- **Direct Push to Blender**: Push models directly to the running Blender instance over `ws://127.0.0.1:9876`.
- **Direct Push to Web Clients**: Push scripts or IFC models to connected WebSocket clients over `ws://127.0.0.1:9877`.
- **Editor Ergonomics**: CodeLens compile triggers, file decorations, hover docs, and a dedicated status sidebar.

### Building & Packaging
```bash
python tools/build.py vscode
# Installs npm dependencies, compiles TypeScript with esbuild,
# and generates a .vsix package in tools/vscode/
```

To install directly into VS Code:
```bash
code --install-extension tools/vscode/lite-step-1.0.0.vsix
```

---

## 3. Lapin (`tools/lapin`)

[Lapin](https://github.com/Ciuntuc-BIM-Studio/lapin) is an MIT desktop BIM modeller whose engine takes commands from programs on the same computer. This tool puts a Lite-STEP model into the Lapin you have open and keeps it up to date as the script changes. It needs nothing built: it runs from the checkout.

### Setup

Talking to Lapin needs its `lapinbim` package (MIT, standard library only):

```bash
pip install lapinbim                      # once it is on PyPI
pip install <lapin checkout>/python       # until then
```

### Commands

| Command | Does | Changes Lapin |
|---|---|---|
| `lite-step lapin status` | The Lapins open on this computer, their storeys, and the Lite-STEP models in them | no |
| `lite-step lapin push model.py` | Compiles, then makes Lapin match the model. `--watch` pushes again at every save; `--dry-run` only says what would change | yes, one undo step |
| `lite-step lapin diff model.py` | What a push would create, change and delete | no |
| `lite-step lapin list model.py` | This model's elements in Lapin and the state of each | no |
| `lite-step lapin fetch -o context.ifc` | The open Lapin project as IFC, to read or build against | no |
| `lite-step lapin remove model.py` | Deletes what this model owns in Lapin, and nothing else | yes, one undo step |
| `lite-step lapin ops model.py` | What each element becomes; needs no Lapin. `-o ops.json` writes it | no |

Every command that talks to Lapin takes `--port N` or `--project NAME` when several Lapins are open. A model file may also be a compiled `.ifc`.

### What each element becomes

| In Lapin | When | What the Lapin user can do |
|---|---|---|
| A Studio box or cylinder | The element is a box or cylinder standing on Z, turned about Z only | Resize, move, turn |
| The same, cut by other elements | Everything cut from it is itself such an element, uncarved | The same; the cut follows both elements |
| A body (a mesh with every cut applied) | Anything else | Move, schedule, export; the shape changes only from the script |

A fallback to a body loses editability, never shape.

### How a push behaves

- **One undo step.** Ctrl+Z in Lapin undoes a whole push. A push that fails part way leaves the project as it was.
- **Only what changed.** An unchanged element is not touched. A changed one keeps its id in Lapin, so what the user attached to it survives. A second push of the same model sends nothing.
- **Only its own elements.** The tool marks what it makes (two parameters in the set `LiteStep`) and never changes or deletes anything else: not what you drew, not another model's elements, not a storey.
- **Several models in one project.** Each is marked with its id: the file's name without its ending, or `--id`.
- **The script wins.** A box you resized by hand in Lapin is put back at the next push, and the push names it. Edits in Lapin do not flow back into the script.
- **Storeys** are matched by name, then by elevation; a missing one is created.

### Working against a model that is already in Lapin

`lite-step lapin fetch -o context.ifc` writes the open project as IFC. Read it for positions, storeys and sizes, write the script in the same coordinates, and push: only the script's elements are added. The fetched file is context; nothing reads it back.

### Not carried yet

Types, materials, colours and property sets. Openings are not separate elements in Lapin: they are already cut into their host.

The mapping is `lite_step/lapin/ops.py` (it reads the compiled IFC, so it describes exactly what the compiler wrote); the delta is `tools/lapin/lapin_sync.py`.

---

## Build Automation

To build both tools at once:
```bash
python tools/build.py
```

