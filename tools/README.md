# Lite-STEP Tools & Extensions

This directory contains editor and CAD extensions for working with **Lite-STEP**:

- **[Blender Extension](blender/)**: Live synchronization bridge for Blender (5.0+) with Bonsai and PBR presentation rendering.
- **[VS Code Extension](vscode/)**: Authoring, compilation, 3D preview, and multi-target push synchronization for Lite-STEP DSL files (`.py`).
- **[Lapin](lapin/)**: Lite-STEP models in [Lapin](https://github.com/Ciuntuc-BIM-Studio/lapin), an open-source desktop BIM modeller: push a model in, and keep it up to date as the script changes.
- **[Survey](survey/)**: snap drawings, screenshots and photos to cameras and measure them against a model (a library for the Surveyor's own scripts).
- **[6-Axis Renderer](render/)**: Headless ThatOpen (`IfcImporter` + `FragmentsManager` with `LodMode.ALL_VISIBLE`) renderer outputting 6 cardinal/aerial PNG views of any compiled `.ifc`.

---

## 1. Blender Extension (`tools/blender`)

An extension for Blender 5.0+ that runs a local WebSocket server (`ws://127.0.0.1:9876`).

### Capabilities
- **Real-Time Sync**: Listens for incoming IFC models pushed from VS Code or web clients.
- **Bonsai Integration**: Automatically loads received IFC models into the active scene via Bonsai.
- **Presentation Shell (`model.blend`)**: Automatically locates and opens Lite-STEP's presentation shell (`lite_step/assets/model.blend`), binding PBR master shaders, environmental lighting, and floor plan CSS stylesheets.

### Installation
1. Download `blender-litestep-<version>.zip` from the repository's [Releases](../../../releases) page.
   To build it yourself instead: `python tools/build.py blender` writes `tools/blender-litestep.zip`.
2. Open Blender (5.0+).
3. Navigate to **Edit** > **Preferences** > **Get Extensions**.
4. In the upper-right corner dropdown menu (`⌵`), select **Install from Disk...**
5. Select the zip and enable the extension.
6. Find the **Lite-STEP** panel in `View3D > N-Panel > Lite-STEP`.

---

## 2. VS Code Extension (`tools/vscode`)

A rich language and tooling extension for authoring buildings in Python with Lite-STEP.

### Capabilities
- **Lite-STEP Compilation**: Compiles Lite-STEP Python scripts directly into valid IFC4X3 models with automatic environment resolution.
- **3D Preview**: In-editor Webview panel rendering models with ThatOpen Components and Web-IFC WASM.
- **Direct Push to Blender**: Push models directly to the running Blender instance over `ws://127.0.0.1:9876`.
- **Push to the Ginnung web app** over `ws://127.0.0.1:9877`, two ways:
  - **Push Model to Web** replaces the current project's model with yours (`output.py` + `output.ifc`); the chat and everything else stay.
  - **Push Model to Web as New Project** asks for a name, opens a new project in the page, and loads the model there. It needs a page with projects on; otherwise it says so and nothing changes.
- **Editor Ergonomics**: CodeLens compile triggers, file decorations, hover docs, and a dedicated status sidebar.

### Installation
Download `lite-step-<version>.vsix` from the repository's [Releases](../../../releases) page, then:
```bash
code --install-extension lite-step-<version>.vsix
```
Or, in VS Code: **Extensions** > `...` > **Install from VSIX...**

### Building & Packaging
```bash
python tools/build.py vscode
# Installs npm dependencies, compiles TypeScript with esbuild,
# and generates a .vsix package in tools/vscode/

cd tools/vscode && npm test     # the connection rules, on a real socket
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
| `lite-step lapin status` | The Lapins open on this computer, and the models linked in the chosen one |
| `lite-step lapin push model.py` | Compiles to `model.lapin` beside the script (pass `format=ifc` or `--format ifc` for legacy IFC link), then links it into Lapin; later pushes reload that link. `--watch` pushes again at every save |
| `lite-step lapin pull model.py` | Writes the parameters changed in Lapin into the script (see below) |
| `lite-step lapin fetch -o context.ifc` | The open Lapin project as IFC, to read or build against |
| `lite-step lapin remove model.py` | Unlinks the model from Lapin (Ctrl+Z there brings it back) |

Every command takes `--port N` or `--project NAME` when several Lapins are open. A model file may also be a pre-built `.lapin` or a compiled `.ifc`. `.lapin` output writes Lapin's internal Format 3 package directly and checks the running Lapin version against tested versions; Lapin ignores the embedded `source/parametric.json` and starter script today.

### How a push behaves

- **Lapin's own importer reads the model, as a link.** It is fast (a 10 MB, 11,800-element tower: 18 s to link, 13 s to reload), shapes the model shares stay shared, walls, slabs and doors are recognised as Lapin elements, and every element can be selected on its own. Nothing is grouped.
- **Read-only in Lapin.** The script owns the model: change it there and push. Manage → Links → *Convert to native* hands elements over to Lapin for good; the script no longer updates them.
- **A push is a reload.** The link is found by its file's path. Lite-STEP's GlobalIds are derived from each element's name and the project's, so a reloaded element is the same element to Lapin, and what you dimensioned or tagged there stays attached.
- **Several models in one project.** Each is its own link; two scripts in different folders never touch each other.
- **Your own work in Lapin is untouched**: what you draw there, views, sheets, dimensions.

### Parameters: editing the model from Lapin

A constant written with `param()` is a parameter:

```python
from lite_step import param

ROOF_SLAB_THICK = param(250, "mm", "Roof slab thickness")
```

`param()` returns the value unchanged, so the script reads as before. A push publishes each parameter to Lapin as a project parameter (Manage → Project information), labelled with its unit, in the group `Lite-STEP <script name>`. Change it there, and:

- `lite-step lapin pull model.py` writes the new value into the script, in that one line and nowhere else (comments and layout stay);
- `lite-step lapin push model.py --watch` does that by itself, then compiles and reloads, so the model in Lapin follows the parameter in a few seconds.

The script wins a conflict: a push overwrites a value changed in Lapin that was not pulled, and says so. A parameter whose value the script computes (`param(WIDTH * 2, ...)`) is not published. A whole number stays whole: 400.5 for a millimetre parameter is refused, not rounded. Lapin keeps a few keys for its own project fields (`NAME`, for one): such a parameter is reported and not published; rename the constant.

---

## 3b. Survey (`tools/survey`)

A library for checking a model against its sources: drawings, screenshots, photos. Every source gets a
camera, measurements are read from the source through that camera, and the model is drawn back onto the
source so a person can see it. It is a library first because the useful questions are rarely the ones a
fixed command foresaw: the Surveyor (or any coding agent) writes short scripts on top of it.

```bash
pip install -r tools/survey/requirements.txt   # OpenCV; numpy comes with Lite-STEP
```

| Module | What it does |
| --- | --- |
| `camera` | `OrthoCamera` (`px = ax + k*s*u`, `py = c - s*v`, separate horizontal stretch `k`) and `PerspectiveCamera` (BCF eye, direction, up, vertical field of view); projection both ways; contract shapes |
| `fit` | `fit_ortho` (two control points or more; scale and stretch can be pinned for plates of one printed scale), `fit_perspective` (PnP over the field of view, four points or more), `refine` (coordinate descent), `chamfer` (silhouette agreement) |
| `ink` | `ink_mask` (threshold, drop short strokes, close dotted members), `silhouette_profile` (widths per height in metres) |
| `overlay` | `draw_polyline` (model lines on a source), `agreement` (green both, red model only, blue source only, with the shares within tolerance), `background_mask` (a viewer screenshot's structure) |

```python
from tools.survey import fit_ortho, ink_mask, silhouette_profile
fit = fit_ortho(world_points, pixel_points, direction=(0, 1, 0), up=(0, 0, 1), image_w=w, image_h=h)
rows = silhouette_profile(ink_mask(gray, min_stroke_px=120), fit.camera, (-38.6, -35.6), (-2.0, 6.0))
```

Fit from control points first, then refine against the model: a silhouette-only search can settle in a
wrong pose. CPU only, no network. Tests: `lite_step/tests/test_tools_survey.py`,
`test_tools_survey_inspect.py`.

### Inspecting: a round of minions

```bash
python -m tools.survey inspect --workdir . --n 12 --model output.ifc --model-version 12 [--shots a b] [--workers 8]
python -m tools.survey inspect --workdir . --request survey/rounds/12/request.json
```

Reads only the working directory (the model, and `survey/` with `sources.json`, `cameras.json`, `shots.json`
and the ink masks `ink/{source}.png`, class 1 = structure) and writes only `survey/rounds/{n}/`: per shot a
report (precision, recall, per element group its support and the shift that would line it up, ink the model
lacks as world boxes) and a WebP overlay, then `consensus.json` (only what two sources agree on) and
`round.json` last. A round that cannot run is written as `status: failed` with its error. One process per
core; the scene is tessellated once per model file and cached by its hash. Shapes: the Ginnung `survey`
contract, 1.1.

| Module | What it does |
| --- | --- |
| `scene` | the model as triangles per element (ifcopenshell), with the aggregation tree; cached |
| `drone` | the ID buffer and depth through any survey camera (Open3D ray casting); sections by depth range |
| `inspect` | `diff_shot`, `consensus`, `scores`, `run_round` |

On the Eiffel run (306 shots, v12) it reproduces the earlier minion's ink-weighted F (0.7995 against 0.7996,
median shot within 0.002) in 36 s on 10 processes once the scene is cached; tessellating the 4.9 MB model is
the larger cost (about 70 s).

---

## 4. Who may connect (Blender and VS Code)

Both tools listen on this computer only (`127.0.0.1`), and a web page can reach that address. So each connection is checked before it is accepted:

| Connection | Accepted? |
|---|---|
| A program on this computer (no `Origin` header): VS Code pushing to Blender, a script | Yes |
| A web page whose origin is on the allowlist | Yes, after a handshake |
| Any other web page | No: refused with a bare `403`, before anything is sent |
| A request whose `Host` is not `127.0.0.1:<port>` or `localhost:<port>` | No (this stops DNS rebinding) |

- **The allowlist** has one built-in entry, `https://ginnung.tech`. Add your own in the VS Code setting `litestep.allowedOrigins`, or in Blender under Preferences > Add-ons > Lite-STEP > *Allowed web origins* (separate with commas). Restart the server afterwards.
- **An entry is one exact origin**, such as `https://app.example.com`. It must be `https://`; `http://` is accepted only on `localhost` or `127.0.0.1` (for example `http://localhost:5173` for a dev server). Wildcards, paths and a trailing slash are refused, and the tool tells you which entries it ignored.
- **Why this is enough.** A page cannot set or forge its `Origin` header, and a browser sends `Origin: https://app.example.com` only for a page it loaded from that site over verified TLS. That is why entries must be `https://`, and why there is no token, pairing code or call to any server.
- **The handshake.** The tool sends `hello`. A web page must answer `status: "connected"` within 5 seconds and send nothing before it; until then it is sent nothing else.
- **What a web page may send.** To VS Code: `.py` and `.ifc` files, saved in a new folder of the workspace. To Blender: `.ifc` only, and it can never name a `.blend` file on your disk to open (`blendPath` is honoured from local programs only).
- **A push is your action.** VS Code sends your model only when you press Push (or Push as New Project), and only to pages that completed the handshake. Saving a file sends nothing, and neither does a page connecting. If no page is connected, the push fails and says so; nothing is queued. The tool never asks a page to erase anything.

Programs on this computer run as you and are trusted; these rules are about web pages.

---

## Build Automation

To build both tools at once:
```bash
python tools/build.py
```

Releases are built by `.github/workflows/release-tools.yml` when a `tools-v<version>` tag is pushed. The tag, `tools/vscode/package.json` and `tools/blender/blender_manifest.toml` must carry the same version.

