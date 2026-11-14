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

| Command | Does |
|---|---|
| `lite-step lapin status` | The Lapins open on this computer, and the IFC files linked in the chosen one |
| `lite-step lapin push model.py` | Compiles to `model.ifc` beside the script, then links it into Lapin; later pushes reload that link. `--watch` pushes again at every save |
| `lite-step lapin fetch -o context.ifc` | The open Lapin project as IFC, to read or build against |
| `lite-step lapin remove model.py` | Unlinks the model from Lapin (Ctrl+Z there brings it back) |

Every command takes `--port N` or `--project NAME` when several Lapins are open. A model file may also be a compiled `.ifc`.

### How a push behaves

- **Lapin's own importer reads the model, as a link.** It is fast (a 10 MB, 11,800-element tower: 18 s to link, 13 s to reload), shapes the model shares stay shared, walls, slabs and doors are recognised as Lapin elements, and every element can be selected on its own. Nothing is grouped.
- **Read-only in Lapin.** The script owns the model: change it there and push. Manage → Links → *Convert to native* hands elements over to Lapin for good; the script no longer updates them.
- **A push is a reload.** The link is found by its file's path. Lite-STEP's GlobalIds are derived from each element's name and the project's, so a reloaded element is the same element to Lapin, and what you dimensioned or tagged there stays attached.
- **Several models in one project.** Each is its own link; two scripts in different folders never touch each other.
- **Your own work in Lapin is untouched**: what you draw there, views, sheets, dimensions.

---

## Build Automation

To build both tools at once:
```bash
python tools/build.py
```

