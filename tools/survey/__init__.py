"""Survey tools: snap sources (drawings, screenshots, photos) to cameras and measure them against a model.

A library first: the Surveyor (`agents/surveyor`, or any coding agent) writes its own scripts on top of it,
because the questions that matter are rarely the ones a fixed command foresaw. CPU only, no network.

- `camera`: orthographic and perspective cameras, projection both ways, contract shapes.
- `fit`: cameras from control points (ortho least squares, perspective PnP), refinement, chamfer.
- `ink`: ink masks from drawings, silhouette profiles in metres.
- `overlay`: the model drawn on a source, and the agreement map shown to the person.

Optional dependency: OpenCV (`pip install -r tools/survey/requirements.txt`).
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
