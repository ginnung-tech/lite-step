# Lite-STEP Tools & Extensions

This directory contains editor and CAD extensions for working with **Lite-STEP**:

- **[Blender Extension](blender/)**: Live synchronization bridge for Blender (5.0+) with Bonsai and PBR presentation rendering.
- **[VS Code Extension](vscode/)**: Authoring, compilation, 3D preview, and multi-target push synchronization for Lite-STEP DSL files (`.py`).
- **[Lapin](lapin/)**: Lite-STEP models in [Lapin](https://github.com/Ciuntuc-BIM-Studio/lapin), an open-source desktop BIM modeller: push a model in, and keep it up to date as the script changes.
- **[Survey](survey/)**: snap drawings, screenshots and photos to cameras and measure them against a model (a library for Heimdal's own scripts).
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
fixed command foresaw: Heimdal (or any coding agent) writes short scripts on top of it.

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

### Taking pictures in

```bash
python -m tools.survey add --workdir . --picture files/view.png --kind viewer_screenshot --title "North face" --bcfv files/view.bcfv
python -m tools.survey add --workdir . --picture files/site.jpg --kind photo --title "Site" --points pts.json [--model output.ifc]
python -m tools.survey index --workdir .
```

`add` copies the picture into `survey/sources/` (the same bytes twice are one source), lists it in
`sources.json`, and fits a camera: a viewer screenshot by the BCF viewpoint it was taken from, anything else
from four or more control points (`[{"px": [x, y], "world": [x, y, z], "by": "heimdal"}]`). Every fit is a new
camera version in `cameras.json` carrying its evidence: `overlay` (the picture with the model's outline in
red, control points green where the picture has them, ringed where the camera puts them) and `agreement`
(share of model pixels within 2 px of the picture). A fit under its gate (80 % for a viewer screenshot, 4 px
mean for control points) is recorded as failed and leaves an `align` question in `survey/questions/`; the
camera is never nudged until it passes. Text and illustrative pictures are listed, not fitted. It prints the
answer for that picture as one JSON line. A photo also gates on its outline lying on the model's
(`outlineWithinPx`, holes closed, 85 %), and five or more control points are checked against each other first: one
that disagrees fails the fit and the question names it. A failed fit's `align` question carries up to three candidate poses (contract
survey-question 1.2: each a camera, its control points and an overlay, ranked by outline); `answer --question Q --choice
pose-k` freezes the one the person picked as the next camera version. `partials` writes one cropped shot per detail (a perspective
camera follows the crop: principal point; lens distortion is in the camera too), `ink` a photo's structure and
openings (class 5, sky seen through the structure; rounds score them as `openings`). `index` writes `survey/index.json` (every file but `rounds/`, with
size and sha256): last, after the files are kept. Heimdal's instructions: `agents/heimdal/SKILL.md`.
Tests: `test_tools_survey_intake.py`.

A drawing (`elevation`, `section`, `plan`, `detail_plane`) is fitted **orthographic** with `--view` (`south` is
the south face, looking north; `north`, `east`, `west`, `top`, or `dx,dy,dz`): three control points, or two
with the drawing's scale pinned by `--px-per-m` (read off its scale bar or two datum ticks). Points are checked
against each other by leaving each out. `--crop x0,y0,x1,y1` takes in one figure of a sheet that holds several,
as a picture of its own (`survey/crops/`).

```bash
python -m tools.survey add --workdir . --picture sheet.png --crop 270,630,900,1160 --kind elevation \
    --title "South elevation" --view south --px-per-m 5.47 --points south-points.json
```

### Cameras from a reconstruction

```bash
python -m tools.survey cameras --workdir . (--colmap recon/sparse/0 | --transforms transforms.json) \
    --images recon/images --align align.json [--n 12] [--frames a.png,b.png]
```

Reads a COLMAP text model (`cameras.txt`, `images.txt`; camera models SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL,
RADIAL, OPENCV) or a Nerfstudio / instant-ngp `transforms.json` (OpenGL convention), and takes `--n` frames spread
around the structure (farthest-point over camera centres) in as photos with their cameras. A reconstruction's frame
is its own, so `--align` is required: `{"points": [{"recon", "world"}, ...]}` (three or more; solved by Umeyama,
gated at 100 mm or 0.5 % of the points' span, the worst pair named when it fails), `{"transform": {"scale",
"rotation", "translation"}}`, or `{"identity": true, "reason"}`. The gate is the alignment's; the model's outline
agreement is recorded with each overlay but does not fail a camera (a model that differs is what is about to be
measured). Images must be the size the reconstruction solved and have square pixels (undistort first). Workflows:
`agents/reconstruction/SKILL.md`. Tests: `test_tools_survey_recon.py`.

### Shape: a freeform mesh against a picture's outline

```bash
python -m tools.survey shape --workdir . --source ID [--elements statue] [--window u0,u1,v0,v1] [--band-m 0.25] [--tol-mm 100]
```

The minions measure where an element is; a sculpture is one element whose shape is wrong. `shape` compares
outlines through a source's passed camera: the picture's silhouette (a drawing's ink grown by `--close-m`,
flooded from the border, shrunk back, so its edge stays where it is drawn; a photo's not-sky) against the
drone's (only `--elements`, by canonical-name suffix, and their parts, when given). Per band of height it
writes each side's edge error in millimetres; runs off the same way become findings with the side in world
terms, the height range, the mean and largest error and the elements at the model's edge there; `missing` and
`extra` are heights one of them lacks entirely. Overall: `iou` and `contour` (mean, p90, max mm). Picture parts
more than 3 m from the model and lines as wide as the picture are left out and reported (`ignoredPictureM2`).
Writes `survey/shape/{source}-v{camera}.json` and an overlay (grey both, red only the model, blue only the
picture). Exact millimetres through an orthographic camera; through a perspective one, at the model's median
depth (`scaleNote`). Through an orthographic camera it also renders the opposite view and the two at right
angles, each centred on the picture's subject, and warns (`viewCheck.warning`) when the chosen view is clearly worse:
a drawing's title names the site's face, not the model's. Tests: `test_tools_survey_shape.py`.

### Inspecting: a round of minions

```bash
python -m tools.survey inspect --workdir . --n 12 --model output.ifc --model-version 12 [--shots a b] [--workers 8]
python -m tools.survey inspect --workdir . --request survey/rounds/12/request.json
```

Reads only the working directory (the model, and `survey/` with `sources.json`, `cameras.json`, `shots.json`
and the ink masks `ink/{source}.png`, class 1 = structure) and writes only `survey/rounds/{n}/`: per shot a
report (precision, recall, per element group its support, the axes it was measured along and the shift that
would line it up, ink the model lacks as world boxes at the model's depth) and a WebP overlay, then
`consensus.json` and `round.json` last. Consensus solves each element's shifts from all views together in 3D
(views at right angles share one axis), counts sources by their root plate, says `two_sources` only when two
sources see the displacement's direction, and reports directions no view sees as unknown, never zero. The
model is measured from a snapshot (its sha256 is in `round.json`); the pool starts processes fresh (never
forked), one per core but one, at lower priority. A round that cannot run is written as `status: failed` with its error. One process per
core; the scene is tessellated once per model file and cached by its hash. Shapes: the `survey` contract,
1.1.

| Module | What it does |
| --- | --- |
| `scene` | the model as triangles per element (ifcopenshell), with the aggregation tree; cached |
| `drone` | the ID buffer and depth through any survey camera (Embree ray casting via embreex, or Open3D); sections by depth range |
| `inspect` | `diff_shot`, `consensus`, `scores`, `run_round` |

On the Eiffel run (306 shots, v12) it reproduces the earlier minion's ink-weighted F (0.7995 against 0.7996,
median shot within 0.002) in 36 s on 10 processes once the scene is cached; tessellating the 4.9 MB model is
the larger cost (about 70 s).

### Heimdal's plan, by hand

`survey/plan.json` (survey 1.5) is Heimdal's plan across rounds: the parts of the building with value,
uncertainty, source and whether the model meets them; what is open (contradictions, questions, missing);
what was closed or changed and why; and the round's decision. `tools/survey/plan.py` checks it, the same
checks a host runs after Heimdal. Run by hand, a round is three commands:

```bash
python -m tools.survey plan begin  --workdir . --request R [--model output.ifc]   # before reading anything
python -m tools.survey plan check  --workdir .   # after plan.json is written; exit 1 lists the broken rules
python -m tools.survey plan record --workdir .   # appends the round to iterations.jsonl (refused if broken)
```

`begin` keeps the plan and every survey file's hash in `survey/.plan-round.json` and prints the round
(counted per request in `iterations.jsonl`), `maxRounds` (3) and this round's rules. `check` holds the plan
to them:

- ids are stable: nothing leaves `items` or `open` without a `closed` entry and its reason;
- a value that moves beyond the two readings' uncertainty (`sqrt(a^2 + b^2)`) has a `changes` entry for this
  round;
- the round and request are the counted ones, and the last round is `converged` or `ask`;
- a request's first round, and a round on the same model as the last, report `previousCheck: not_checked`;
- an instruction carries `checkSpec`, and a reported check or `converged` carries `checkResult`: a number
  measured on the person's own pictures (never a viewer screenshot, text or an illustration), each with a
  passed camera, its `output` a file under `survey/` written this round.

Tests: `test_tools_survey_plan.py` (the fixtures are a real carport plan whose round 3 broke the rules).

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

## 5. Review (`tools/review`)

```bash
lite-step review output.py [--ifc output.ifc] [--out review/] [--no-compile]
```

The check to run before a model is called done, and the one to hand a coding agent: no model calls, every number
measured. In order:

1. **Compile** (`python output.py`); a compile failure ends the review, and `warning:` lines become findings.
2. **Validate** the IFC with `ifcopenshell.validate` (errors only, the first 50).
3. **Triage the geometry** on the scene `tools/survey` builds (cached in `review/.cache/` by the IFC's hash),
   with exact solids from manifold3d:
   - `floating`: a group of touching elements with nothing that reaches the ground or the model's lowest level
     (error when it holds a structural element);
   - `clash`: two elements that are not parts of one thing share volume. Not a clash: parts of one element; a
     linear element set into another at one end (a beam in its column head, a cable anchor); walls lapped at a
     corner or a T; the class pairs in `rules.py` (rebar in its host, a pile in its cap, a member seated in its
     footing, anything in the ground). Error when both are structural and the overlap is deeper than 20 mm;
   - `crossing`: two members crossing mid-length, as an X of bracing drawn as two whole bars (warning);
   - `near_miss`: a column, beam or member 5-150 mm short of meeting another element (warning);
   - `unchecked`: a mesh that is not a closed solid, so none of the above could be measured on it.

Findings are grouped by kind and IFC classes (`clash:IfcBeam|IfcWall`), name elements by canonical name (pass
them to `--q`), and carry their numbers (volume, depth, gap). It writes `review/findings.md` (read this) and
`review/findings.json`, and exits **1 while an error group is open**, 2 when the review cannot run, 0 otherwise.
A group whose contact is meant is accepted in `review/accepted.json` with its reason:
`[{"group": "clash:IfcCourse|IfcPavement", "reason": "the course is a layer of the pavement"}]`; a row without a
reason accepts nothing, and a floating group should never be accepted (something holds it up: model it).

Exact checks are capped at three per pair of elements and the triage at 120 s; suspects left unmeasured are
reported as a `budget` finding, never dropped silently. Loading a very large model is not in that budget.
On the pack's 36 reference models and the Millau trial: the remaining errors are real modelling defects in the
references (floor beams occupying their bent caps, a roof underlay cut through its rafters, a timber wall touching
nothing), the rest warnings. Tests: `test_tools_review.py`.

---

## Build Automation

To build both tools at once:
```bash
python tools/build.py
```

Releases are built by `.github/workflows/release-tools.yml` when a `tools-v<version>` tag is pushed. The tag, `tools/vscode/package.json` and `tools/blender/blender_manifest.toml` must carry the same version.

