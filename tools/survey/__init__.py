"""Survey tools: snap sources (drawings, screenshots, photos) to cameras and measure them against a model.

A library first: Heimdal (`agents/heimdal`, or any coding agent) writes its own scripts on top of it,
because the questions that matter are rarely the ones a fixed command foresaw. CPU only, no network.

- `camera`: orthographic and perspective cameras, projection both ways, contract shapes.
- `fit`: cameras from control points (ortho least squares, perspective PnP), refinement, chamfer.
- `ink`: ink masks from drawings, silhouette profiles in metres.
- `overlay`: the model drawn on a source, and the agreement map shown to the person.
- `scene`, `drone`: the model as triangles per element, and the ray-cast ID buffer through any camera.
- `inspect`: a round of minions over the survey's shots (`python -m tools.survey inspect`).
- `intake`: take a picture in, fit its camera, show the fit back or ask (`python -m tools.survey add`).

Optional dependencies: OpenCV, Open3D (`pip install -r tools/survey/requirements.txt`). `scene`, `drone` and
`inspect` are imported on use, so the camera and fitting tools work without Open3D.
"""

from .camera import OrthoCamera, PerspectiveCamera, basis, camera_from_contract
from .fit import Fit, chamfer, fit_ortho, fit_perspective, refine
from .ink import ProfileRow, ink_mask, silhouette_profile
from .overlay import Agreement, agreement, background_mask, draw_polyline

__all__ = [
    "Agreement", "Fit", "OrthoCamera", "PerspectiveCamera", "ProfileRow", "agreement", "background_mask",
    "basis", "camera_from_contract", "chamfer", "draw_polyline", "fit_ortho", "fit_perspective", "ink_mask",
    "refine", "silhouette_profile",
]
