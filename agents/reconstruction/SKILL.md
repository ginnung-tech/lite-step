---
name: reconstruction
description: Suggestions for turning video, drone footage, photo sets or an existing scan into clean geometry and calibrated cameras for a Lite-STEP model - what to do outside Lite-STEP (frames, photogrammetry, mesh cleaning) and what to bring in (a clean watertight mesh, aligned cameras). Use when someone brings footage, a photo set or a scan of an existing building, bridge, monument or object, or asks how to use one. Best practice, not a requirement.
---

# Reconstruction: from footage or a scan to a model, suggested workflows

These are suggestions. Every step has alternatives, and the person decides what the project needs: which tools,
which licence they accept for a scan they found, how exact the result must be. What is fixed is only what
Lite-STEP itself requires (a `Mesh` is a closed, watertight solid in integer millimetres) and what a measurement
needs to mean anything (a camera aligned to the model, below).

## The shape of it

Heavy reconstruction works best **outside** Lite-STEP, with tools built for it; Lite-STEP takes in what they
produce. Two things come back in:

1. **Clean meshes** for what is genuinely freeform (a statue, carved stone, rock, terrain, an ornate cornice):
   watertight, decimated to what the use needs, in the model's frame and millimetres.
2. **Calibrated cameras** for the frames, so Heimdal can measure every version of the model against the footage
   (`python -m tools.survey cameras`, then `shape` and `inspect`).

Everything that is architecture or structure (walls, slabs, piers, decks, columns, openings, stairs) is better
**modelled** in Lite-STEP from what the reconstruction measures than imported as mesh: it stays editable, carries
IFC classes and parameters, and stays small. The scan is the reference; the model is the building.

## 1. Frames from video or a drone flight

- Extract frames at a rate that keeps roughly 70-80 % overlap between neighbours (often 1-3 frames per second for
  a drone orbit; more for fast handheld motion), e.g. with `ffmpeg -i flight.mp4 -vf fps=2 frames/%05d.png`.
- Drop blurred frames (the variance of the Laplacian is a common sharpness score) and near-duplicates.
- Keep the drone's telemetry (DJI `.SRT`, XMP or EXIF: GPS, altitude, gimbal angles, focal length): it can give the
  reconstruction a real scale and a real "up" later.
- Fixed focus and exposure, no digital zoom, no stabilisation crop, if you are the one flying or filming.

## 2. Cameras and geometry (photogrammetry)

Common choices, each with its own strengths:

- **COLMAP** (structure from motion: every frame's camera and a sparse point cloud), with **OpenMVS** or COLMAP's own
  dense stereo for a dense cloud and a surface mesh. A reliable default for buildings with texture.
- Learned feature matching (**hloc** with SuperPoint and LightGlue) where COLMAP's matching struggles: glass,
  repeating facades, low texture.
- Feed-forward reconstruction (**MASt3R**, **VGGT**) for short or sparse clips, as a fast start refined by bundle
  adjustment.
- Gaussian-splatting methods with a surface extraction (**2DGS**, **PGSR**) for glossy or thin structure (curtain
  walls, trusses, cables), where classical stereo leaves holes.

These want a GPU and their own environments; none of them is a Lite-STEP dependency.

Export the cameras as a **COLMAP text model** (`colmap model_converter --output_type TXT`) or a Nerfstudio
**`transforms.json`**, and **undistort** the images (`colmap image_undistorter`): Heimdal's cameras take square
pixels and an OpenCV lens model.

## 3. The alignment: why the reconstruction is not yet in the model

A reconstruction solves its cameras in a frame of its **own**. COLMAP does not know where the model's origin is,
which way is up, or what a metre is: its origin is wherever the solve started, its axes are arbitrary, and its
unit is arbitrary (a 40 m bridge may come out 1.3 units long). Before a camera from it can look at the model, the
whole reconstruction has to be moved onto the model's frame by a **similarity transform**: one rotation, one
translation and one scale, `world = s * R * recon + t`.

Nothing can guess that transform for you. It comes from one of:

- **Points known in both** (the most direct): three or more features you can find in the reconstruction's point
  cloud and in the model (a pier corner, the top of a mast, the corner of the deck), spread across the structure
  and not on one line. Four or more, so that one wrong pair shows in the residuals.
- **GPS and gravity** from the drone, turned into the model's frame elsewhere (the model's survey point and
  orientation), handed over as a ready transform.
- **Already in model coordinates**: a georeferenced pipeline that wrote the model's frame and metres. Say so, and
  say which pipeline.

Then:

```bash
python -m tools.survey cameras --workdir . --colmap recon/sparse/0 --images recon/images \
    --align survey/scripts/align.json [--n 12] [--frames a.png,b.png]
```

`align.json` is `{"points": [{"recon": [x, y, z], "world": [x, y, z]}, ...]}`, or
`{"transform": {"scale": s, "rotation": [[...], [...], [...]], "translation": [x, y, z]}}`, or
`{"identity": true, "reason": "..."}`. World coordinates are the model's, in metres. Without it the command refuses
and says what it needs: that refusal is deliberate. It takes in `--n` frames spread around the structure (a
reconstruction is hundreds of near-duplicates; a survey wants a few views from different sides), each as a photo
with its aligned camera and an overlay of the model drawn on it. The alignment's points are reported one by one;
a pair that disagrees with the others fails the gate and is named. Look at the overlays: the model should sit on
the footage wherever the model is right.

## 4. Clean meshes for the freeform parts

Suggested steps (the trial that brought a statue in used `trimesh`, `pymeshlab` and `manifold3d`):

- **Align** the mesh with the same transform as the cameras, so it lands in the model's frame and metres; convert
  to millimetres for Lite-STEP.
- **Clip and split** with exact booleans (manifold3d): cut it off at the plinth, separate what is a different
  element (a flame, a tablet), hollow it if the interior matters.
- **Decimate where it does not show, keep where it does.** Uniform simplification flattens faces and hands first;
  quadric edge collapse with per-vertex weights (more weight on a face, hands, inscriptions) keeps them at a
  fraction of the triangles. Tens of thousands of triangles are plenty for most BIM uses; the viewer and every
  compile pay for each one.
- **Repair to a closed 2-manifold**: watertight, consistently wound outward, no self-intersections, after rounding
  to integer millimetres (round, then check again: rounding can close thin gaps into defects).
- **Check it against the sources**: take the drawings or frames in and run `python -m tools.survey shape` on the
  model with the mesh in it. A found scan is a scan of *something*: a statuette, a maquette or a restored state may
  differ from the real thing, and only a measurement says by how much.

How it gets into `output.py`: save the cleaned mesh as `.ply` or `.stl` in the project (integer millimetres,
already in the model's frame) and name it by path and hash:

```python
skin = Mesh(name="skin", asset="meshes/statue.ply", sha256="<sha256 of the file>")
```

The script stays small and still says exactly which bytes it builds from; a changed file is refused until its new
hash is written in. Writing the vertices into the script as data also works (one trial packed 50 k triangles into
base-85 text), and every agent that reads that script pays for it.

## 5. Measuring each version against the footage

With aligned cameras, every model version is measured like any other survey:

- `python -m tools.survey shape --source FRAME` for the outline (sky or background against the model), per height
  and side in millimetres, with the elements at fault.
- `python -m tools.survey depth --source FRAME --depth-map ...` with the reconstruction's depth map for that frame:
  per element, how far in front of or behind the measured surface it lies (recesses, set-backs, balconies, which an
  outline cannot see).
- `python -m tools.survey inspect` for element positions across all frames together.
- Heimdal's plan and checks as usual (`agents/heimdal/SKILL.md`): a check is a number measured on the person's own
  frames.

## Licences and provenance

A scan or model found online comes with a licence and a history. Note both in the project (where it came from,
under what licence, what it is a scan of) so the person can decide what they can deliver with it. Whether a
licence suits the project is the person's call.
