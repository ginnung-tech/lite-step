"""Cross-convention contract test for the opening-local frame.

Pins the contract: opening-local coordinates are **Z-up**
(local x along wall, local y depth through wall, local z = height
from sill). This must match the world convention so that the agent
can transfer Z-up reasoning into Window/Door child contours without
a mental axis swap.

The test authors the same window two ways:

  Z-up local — `Extrude(contour=[Point(x, y=0, z=h), ...], color="glass")`  (canonical)
  Y-up local — `Extrude(contour=[Point(x, y=h, z=0), ...], color="glass")`  (legacy)

Since DSL v1.5 a glass pane is a contour `Box(color="glass")` (the
legacy `Panel` primitive was removed at the cutover); the generator
emits it as an `IfcBuildingElementProxy` via
`_create_opening_contour_solid`.

…and asserts:

1.  Z-up-local glass goes through `_create_opening_contour_solid` cleanly:
    the proxy's placement matrix has its **outward normal**
    aligned with ±Y world (perpendicular to the wall plane), and
    the glass occupies a vertical (`Δz > 1 m`) rectangle on the
    wall.

2.  Y-up-local glass produces the **broken geometry this would ship**: either the proxy is dropped (Newell normal degenerate,
    silent-or-warn return) **or** the proxy's geometry collapses
    to a flat rectangle in the world XY plane (`Δz ≈ 0`) — i.e.
    the glass lies down on the sill. The point is to prove the two
    conventions produce *different* IFC, so if anyone later flips
    the generator back to Y-up locally (e.g. by reverting the transform swap), this test catches it.

If both conventions produce the same vertical glass, the generator
is silently accepting both — which means the Y-up bug is back as a
"works by accident" path that will break the first time a corner-
case contour winding produces a degenerate Newell normal.
"""

from __future__ import annotations

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
np = pytest.importorskip("numpy")

# ifcopenshell.geom is optional (depends on platform OpenCascade build). The
# test skips below if it's missing. Import at module scope so `ifcopenshell`
# isn't accidentally shadowed by a local-scope `import ifcopenshell.geom`
# inside the bbox helper (which would trigger UnboundLocalError).
try:
    import ifcopenshell.geom as _ifc_geom  # noqa: F401
    HAS_GEOM = True
except Exception:
    HAS_GEOM = False

from lite_step.compiler.executor import execute_lite_step_script
from lite_step.ifc.generator import generate_ifc


def _make_source(opening_local: str) -> str:
    """Build a minimal one-wall + one-window Lite-STEP script.

    ``opening_local`` selects the convention for the glass Solid contour
    inside the Window: ``"zup"`` puts height in ``z``, depth in ``y``;
    ``"yup"`` puts height in ``y``, depth in ``z`` (the legacy bug).
    """
    if opening_local == "zup":
        glass_contour = (
            "Point(x=50, y=0, z=50), Point(x=1150, y=0, z=50), "
            "Point(x=1150, y=0, z=1350), Point(x=50, y=0, z=1350)"
        )
    elif opening_local == "yup":
        glass_contour = (
            "Point(x=50, y=50, z=0), Point(x=1150, y=50, z=0), "
            "Point(x=1150, y=1350, z=0), Point(x=50, y=1350, z=0)"
        )
    else:
        raise ValueError(f"unknown convention: {opening_local}")

    return f"""\
from lite_step.models import Project, Storey, Point, Box, Extrude, Wall, Window

def generate_project():
    proj = Project(name="zup_contract")
    storey = Storey(elevation=0)
    proj.add_storey(storey)

    wall = Wall(id="wall_south")
    wall.add(Box(
        start=Point(x=-3000, y=-2000, z=300),
        end=Point(x=3000, y=-1700, z=3215),
        type="sketch", id="body"
    ))
    win = Window(width=1200, height=1400, id="win_0")
    win.add(Extrude(
        contour=[{glass_contour}],
        thickness=6, color="glass", id="glass"
    ))
    wall.anchor(win, along=2000, up=900)
    storey.add(wall)

    return proj


result = generate_project()
"""


def _ifc_glass_bbox(ifc_text: str) -> dict[str, tuple[float, float, float, float, float, float]] | None:
    """Pull world-space bounding boxes for each glass proxy.

    Since DSL v1.5 a glass pane is a contour ``Box(color="glass")`` child
    of the opening, which the generator emits as an
    ``IfcBuildingElementProxy`` (``_create_opening_contour_solid``) — the
    legacy ``Panel`` → ``IfcPlate`` path was removed at the v1.5 cutover.

    Returns ``{proxy_global_id: (xmin, xmax, ymin, ymax, zmin, zmax)}``
    derived from the shape representation + placement transform.
    Returns ``None`` if no proxy exists (the bug-shape case where
    the glass was dropped entirely).
    """
    import tempfile, os

    with tempfile.NamedTemporaryFile(mode="w", suffix=".ifc", delete=False,
                                      encoding="utf-8") as f:
        f.write(ifc_text)
        path = f.name
    try:
        model = ifcopenshell.open(path)
        plates = model.by_type("IfcBuildingElementProxy")
        if not plates:
            return None

        if not HAS_GEOM:
            pytest.skip("ifcopenshell.geom is not available; geometry-extraction skipped")

        settings = _ifc_geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        out: dict[str, tuple[float, float, float, float, float, float]] = {}
        for plate in plates:
            try:
                shape = _ifc_geom.create_shape(settings, plate)
            except Exception:
                continue
            verts = shape.geometry.verts  # flat list [x0,y0,z0, x1,y1,z1, ...]
            if not verts:
                continue
            xs = verts[0::3]
            ys = verts[1::3]
            zs = verts[2::3]
            out[plate.GlobalId] = (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))
        return out
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _compile_to_ifc(source: str) -> str:
    exec_result = execute_lite_step_script(source)
    assert exec_result.success, exec_result.error
    ifc_result = generate_ifc(exec_result.project)
    assert ifc_result.success, ifc_result.error
    assert ifc_result.ifc_content
    return ifc_result.ifc_content


def test_zup_local_glass_is_vertical() -> None:
    """A Z-up-local contour produces a vertical glass panel.

    Glass on a south wall (wall normal in -Y world) should have:
      - non-trivial Z extent (Δz > 1 m): glass spans the window height
      - small Y extent (≈ 0 plus thickness): glass is thin through wall
    """
    ifc_text = _compile_to_ifc(_make_source("zup"))
    bboxes = _ifc_glass_bbox(ifc_text)
    if bboxes is None:
        pytest.fail("Z-up-local glass: no glass proxy was emitted")

    # There is exactly one glass panel in this fixture
    assert len(bboxes) == 1, f"expected 1 glass proxy, got {len(bboxes)}: {bboxes}"
    xmin, xmax, ymin, ymax, zmin, zmax = next(iter(bboxes.values()))

    dz = zmax - zmin
    dy = ymax - ymin
    dx = xmax - xmin

    assert dz > 1.0, (
        f"Z-up-local glass should span > 1 m vertically; got Δz={dz:.4f} m. "
        f"Full bbox: x[{xmin:.3f}..{xmax:.3f}] y[{ymin:.3f}..{ymax:.3f}] "
        f"z[{zmin:.3f}..{zmax:.3f}]"
    )
    # Glass thickness (6 mm) goes through the wall in Y direction. Allow
    # generous tolerance since the panel placement adds a centroid offset.
    assert dy < 0.5, (
        f"Z-up-local glass should be thin through the wall (Δy small); "
        f"got Δy={dy:.4f} m"
    )
    # Glass width is 1.1 m (1200 - 2*50 frame inset)
    assert 0.9 < dx < 1.3, (
        f"Z-up-local glass should be ~1.1 m wide (1200mm - 2*50mm frame); "
        f"got Δx={dx:.4f} m"
    )


def test_yup_local_glass_is_broken_or_dropped() -> None:
    """A Y-up-local contour MUST produce different (broken) geometry.

    Either:
      (a) the proxy is dropped entirely (Newell normal degenerate
          on the world-coord contour), OR
      (b) the proxy exists but lies horizontal on the sill plane:
          Δz ≈ 0 (no vertical extent) and Δx, Δy span the window
          rectangle dimensions.

    If neither (a) nor (b) holds — i.e. the Y-up-local contour produces
    the SAME vertical glass as the Z-up-local one — the generator is
    silently accepting both conventions, which is the regression
    failure mode this test exists to guard against.
    """
    ifc_text = _compile_to_ifc(_make_source("yup"))
    bboxes = _ifc_glass_bbox(ifc_text)

    # Case (a): plate dropped silently
    if bboxes is None or len(bboxes) == 0:
        return

    # Case (b): plate emitted but lies flat
    xmin, xmax, ymin, ymax, zmin, zmax = next(iter(bboxes.values()))
    dz = zmax - zmin
    dy = ymax - ymin

    # Y-up-local glass with all pt.z == 0 puts every point at z=origin_z
    # (the sill plane). After thickness extrusion along the Newell
    # normal (which points in ±Z because all contour points share z),
    # the panel is a horizontal slab — Δz comes ONLY from the thickness
    # (6 mm) plus minor numerical wiggle. Δy spans the window height.
    assert dz < 0.05, (
        f"Y-up-local glass should be a flat horizontal slab "
        f"(Δz≈thickness, tiny). Got Δz={dz:.4f} m. If this assert "
        f"fails it means the generator is silently accepting Y-up "
        f"opening-local — the regression has been reintroduced."
    )
    assert dy > 1.0, (
        f"Y-up-local glass placed flat should span > 1 m in Y "
        f"(window height ended up there). Got Δy={dy:.4f} m"
    )
