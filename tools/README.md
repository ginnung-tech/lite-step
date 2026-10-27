# Lite-STEP Tools & Extensions

This directory contains editor and CAD extensions for working with **Lite-STEP** and **Ginnung**:

- **[Blender Extension](blender/)**: Live synchronization bridge for Blender (5.0+) with Bonsai and PBR presentation rendering.
- **[VS Code Extension](vscode/)**: Authoring, compilation, 3D preview, and multi-target push synchronization for Lite-STEP DSL files (`.py`).

---

## 1. Blender Extension (`tools/blender`)

An extension for Blender 5.0+ that runs a local WebSocket server (`ws://127.0.0.1:9876`).

### Capabilities
- **Real-Time Sync**: Listens for incoming IFC models pushed from VS Code or Ginnung web apps.
- **Bonsai Integration**: Automatically loads received IFC models into the active scene via Bonsai.
- **Presentation Shell (`model.blend`)**: Automatically locates and opens Lite-STEP's presentation shell (`lite_step/assets/model.blend`), binding PBR master shaders, environmental lighting, and floor plan CSS stylesheets.

### Installation
1. Package the extension:
   ```bash
   python tools/build.py blender
   # Generates tools/blender-ginnung.zip
   ```
2. Open Blender (5.0+).
3. Navigate to **Edit** > **Preferences** > **Get Extensions**.
4. In the upper-right corner dropdown menu (`⌵`), select **Install from Disk...**
5. Select `tools/blender-ginnung.zip` and enable the extension.
6. Find the **Ginnung** panel in `View3D > N-Panel > Ginnung`.

---

## 2. VS Code Extension (`tools/vscode`)

A rich language and tooling extension for authoring buildings in Python with Lite-STEP.

### Capabilities
- **Lite-STEP Compilation**: Compiles Lite-STEP Python scripts directly into valid IFC4X3 models with automatic environment resolution.
- **3D Preview**: In-editor Webview panel rendering models with ThatOpen Components and Web-IFC WASM.
- **Direct Push to Blender**: Push models directly to the running Blender instance over `ws://127.0.0.1:9876`.
- **Direct Push to Web**: Push scripts or IFC models to Ginnung web applications over `ws://127.0.0.1:9877`.
- **Editor Ergonomics**: CodeLens compile triggers, file decorations, hover docs, and a dedicated status sidebar.

### Building & Packaging
```bash
python tools/build.py vscode
# Installs npm dependencies, compiles TypeScript with esbuild,
# and generates a .vsix package in tools/vscode/
```

To install directly into VS Code:
```bash
code --install-extension tools/vscode/ginnung-vscode-1.0.0.vsix
```

---

## Build Automation

To build both tools at once:
```bash
python tools/build.py
```
