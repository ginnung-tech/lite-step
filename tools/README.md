# Lite-STEP Tools & Extensions

This directory contains editor and CAD extensions for working with **Lite-STEP**:

- **[Blender Extension](blender/)**: Live synchronization bridge for Blender (5.0+) with Bonsai and PBR presentation rendering.
- **[VS Code Extension](vscode/)**: Authoring, compilation, 3D preview, and multi-target push synchronization for Lite-STEP DSL files (`.py`).
- **[Lapin](lapin/)**: Lite-STEP models in [Lapin](https://github.com/Ciuntuc-BIM-Studio/lapin), an open-source desktop BIM modeller (in progress: the mapping is here, the sync is next).

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

[Lapin](https://github.com/Ciuntuc-BIM-Studio/lapin) is an MIT desktop BIM modeller whose engine takes commands from programs on the same computer. This tool turns a compiled Lite-STEP model into Lapin operations. It needs nothing built: it runs from the checkout.

```bash
lite-step lapin ops model.py              # what each element becomes; no Lapin needed
lite-step lapin ops model.py -o ops.json  # the same as JSON
lite-step lapin ops model.ifc             # from an IFC already compiled
```

Each element becomes one of two things:

| In Lapin | When | What the Lapin user can do |
|---|---|---|
| A Studio box or cylinder | The element is a box or cylinder standing on Z, turned about Z only | Resize, move, turn |
| The same, cut by other elements | Everything cut from it is itself such an element, uncarved | The same; the cut follows both elements |
| A body (a mesh with every cut applied) | Anything else | Move, schedule, export; the shape changes only from the script |

A fallback to a body loses editability, never shape. The mapping is `lite_step/lapin/ops.py`; it reads the compiled IFC, so it describes exactly what the compiler wrote. Sending the operations to a running Lapin (`lite-step lapin push`) is not here yet.

---

## Build Automation

To build both tools at once:
```bash
python tools/build.py
```

