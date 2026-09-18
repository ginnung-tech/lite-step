"""Where ``carve=``, ``miter()`` and the container clip meet.

Three features landed within a day of each other — a container's ``.clip()``
reaching its parts, ``miter()`` accepting a container, and
``carve=`` on ``.add()`` — and each shipped with tests that exercised it
ALONE. This file is the composition, and writing it found a real defect in the
overlap: a container miter recorded its joint id on the container while the
displacement pass pairs the leaf SOLIDS, so the auto-carve suppression matched
nothing and fired on a joint the author had already resolved.

Two things are asserted here that no single-feature test can reach:

* **geometry, through the kernel** — the earlier composed fixture asserts
  ``_clips``/``_cuts`` counts and path coordinates on the DSL tree, which is
  the shape exactly (ifcopenshell trims a swept disk's directrix and
  reports the bar truncated at an UNCHANGED vertex count: structure right,
  geometry wrong). These tessellate and read volumes.
* **precedence between the two passes** — an authored joint outranks a
  ``carve=`` direction, which is currently true only because one ``if`` sits
  above another in one loop.
"""
from __future__ import annotations

import math
import os
import tempfile

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
import ifcopenshell.geom  # noqa: E402
import numpy as np  # noqa: E402

from lite_step.compiler.displacement import apply_displacement, carve_pairs  # noqa: E402
from lite_step.compiler.executor import normalize_project_to_meters  # noqa: E402
from lite_step.ifc.generator import generate_ifc  # noqa: E402
from lite_step.models import (  # noqa: E402
    Bar, Box, Pipe, Point, Project, Slab, miter,
)
from lite_step.models.project import Storey  # noqa: E402

BAR_DIA = 16                       # nominal, mm — a stock rebar size
SLEEVE_R = BAR_DIA // 2 + 20       # 2 cm larger in RADIUS than the bar
RUN = 6000                         # slab run, mm
DEPTH = 400                        # slab depth, mm


def _sleeved_slab(name, p0, p1, box_end):
    """One concrete slab carrying a plastic sleeve with a bar inside it.

    The sleeve is a ``Pipe``, not a second ``Bar``: ``Bar`` is REBAR — its
    ``diameter`` must be a stock size from the grade registry and its ``grade``
    resolves against the material registry, which has no plastic. A sheath
    20 mm larger in radius is 56 mm across, not a stock rebar size, so "a bar
    inside a bar" is not an expressible model. ``Pipe`` is radius-based and
    takes any material, which is the honest spelling of a sleeve.
    """
    slab = Slab(name=name)
    slab.add(Box(name="body", material="Concrete_C30-37",
                 start=Point(x=0, y=0, z=0), end=box_end))
    bar = Bar(name="rebar", diameter=BAR_DIA, path=[p0, p1])
    sleeve = Pipe(name="sleeve", radius=SLEEVE_R, path=[p0, p1])
    sleeve.difference(bar)         # the bar carves the plastic
    slab.add(sleeve)
    return slab


def _mitered_pair(carve_b=None):
    """Two sleeved slabs meeting at the origin, mitered on the bisector."""
    a = _sleeved_slab("a", Point(x=0, y=200, z=150), Point(x=RUN, y=200, z=150),
                      Point(x=RUN, y=DEPTH, z=300))
    b = _sleeved_slab("b", Point(x=200, y=0, z=150), Point(x=200, y=RUN, z=150),
                      Point(x=DEPTH, y=RUN, z=300))
    miter(a, b, at=Point(x=0, y=0, z=0), edge=(0, 0, 1))

    proj = Project(name="composed")
    storey = Storey(name="ground", elevation=0)
    storey.add(a)
    if carve_b is None:
        storey.add(b)
    else:
        storey.add(b, carve=carve_b)
    proj.add_storey(storey)
    return proj


def _shapes(ifc_text):
    """``{Name: (volume_m3, world_aabb)}`` for every product with geometry."""
    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w") as fh:
            fh.write(ifc_text)
        model = ifcopenshell.open(path)
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        out = {}
        for prod in model.by_type("IfcProduct"):
            if not getattr(prod, "Representation", None):
                continue
            try:
                shape = ifcopenshell.geom.create_shape(settings, prod)
            except Exception:  # noqa: BLE001 - an unshapeable product is not ours
                continue
            v = np.array(shape.geometry.verts).reshape(-1, 3)
            if not len(v):
                continue
            f = np.array(shape.geometry.faces).reshape(-1, 3)
            t = v[f]
            vol = abs(np.einsum("ij,ij->i", t[:, 0],
                                np.cross(t[:, 1], t[:, 2])).sum() / 6.0)
            out[prod.Name] = (vol, (v.min(axis=0), v.max(axis=0)))
        return out
    finally:
        os.unlink(path)


def _compiled(proj):
    ifc = generate_ifc(normalize_project_to_meters(proj), source_code=None)
    assert ifc.success, ifc.error
    return _shapes(ifc.ifc_content)


# ── geometry, not structure ──────────────────────────────────────────────────

def test_the_sleeve_is_trimmed_at_the_bisector_in_the_emitted_geometry():
    """Read off the kernel, not off ``_clips``.

    The bisector of +x and +y through the origin is the plane x=y, so a sleeve
    running along y=200 is cut at x=200. Asserting that on the DSL path would
    pass while the emitted solid ran the full length — which is the exact
    signature.
    """
    shapes = _compiled(_mitered_pair())
    (_v, (lo, hi)) = shapes["pipe:sleeve:slab:a:storey:ground"]
    assert lo[0] == pytest.approx(0.200, abs=1e-3), (
        f"sleeve starts at x={lo[0]:.3f}, not the bisector at 0.200 — the trim "
        f"did not reach the emitted geometry"
    )
    assert hi[0] == pytest.approx(RUN / 1000, abs=1e-3)   # far end untouched


def test_the_sleeve_is_HOLLOW_in_the_emitted_geometry():
    """The bar's void has to survive the sleeve being trimmed.

    Trimming the sleeve must not drop the hole through it: the cutter is
    reached through its consumer, so a trim that also shortened the CUTTER
    would leave a solid stub. Distinguished by VOLUME, because a solid stub and
    a hollow one have the same extents and the same vertex count order — the
    two are indistinguishable on structure.
    """
    shapes = _compiled(_mitered_pair())
    vol, _ = shapes["pipe:sleeve:slab:a:storey:ground"]

    length = (RUN - 200) / 1000.0
    r_out, r_in = SLEEVE_R / 1000.0, BAR_DIA / 2 / 1000.0
    hollow = math.pi * (r_out ** 2 - r_in ** 2) * length
    solid = math.pi * r_out ** 2 * length

    # 3% covers the tessellation of a circle into a polygon (which always
    # UNDER-states area); the solid/hollow gap is ~9%, so the two cannot be
    # confused at this tolerance.
    assert vol == pytest.approx(hollow, rel=0.03), (
        f"sleeve volume {vol:.6f} — hollow would be {hollow:.6f}, a solid stub "
        f"{solid:.6f}. The bar stopped carving the sleeve."
    )
    assert vol < solid * 0.95


def test_both_concrete_bodies_keep_their_mitered_half():
    """The corner is resolved once: each body keeps its half, and the pair does
    not also auto-carve (which would subtract the same corner twice)."""
    shapes = _compiled(_mitered_pair())
    a = shapes["box:body:slab:a:storey:ground"][0]
    b = shapes["box:body:slab:b:storey:ground"][0]
    assert a == pytest.approx(b, rel=1e-3), (a, b)   # symmetric fixture


# ── the two passes meeting ───────────────────────────────────────────────────

def _inferred(proj):
    norm = normalize_project_to_meters(proj)
    apply_displacement(norm)
    return [(l, w) for l, w, mech in carve_pairs(norm) if mech == "inferred"]


def test_a_container_miter_suppresses_the_auto_carve():
    """The defect this file was written to find.

    ``miter()`` records its joint id on whatever it was called on. Since
    that may be a CONTAINER — while this pass pairs the leaf SOLIDS beneath it,
    whose own ``_miter_joints`` are empty. So the suppression matched nothing
    on exactly the models the container miter was added for, and the auto-carve
    fired on a joint the author had already resolved: one redundant boolean per
    mitered pair, CSG depth spent for a near-no-op.

    Nothing caught it. The composed fixture in test_container_clip.py uses a
    container miter and passes either way, because it asserts clip counts and
    path coordinates rather than carve pairs.
    """
    assert _inferred(_mitered_pair()) == []


def test_the_body_spelling_and_the_container_spelling_agree():
    """Same joint, said two ways. The control for the test above: without it,
    "no auto-carve" could mean the suppression works OR that nothing overlapped.
    """
    a, b = Slab(name="a"), Slab(name="b")
    ba = Box(name="body", start=Point(x=0, y=0, z=0),
             end=Point(x=RUN, y=DEPTH, z=300))
    bb = Box(name="body", start=Point(x=0, y=0, z=0),
             end=Point(x=DEPTH, y=RUN, z=300))
    a.add(ba)
    b.add(bb)
    miter(ba, bb, at=Point(x=0, y=0, z=0), edge=(0, 0, 1))   # on the BODIES
    proj = Project(name="body-miter")
    storey = Storey(name="ground", elevation=0)
    storey.add(a)
    storey.add(b)
    proj.add_storey(storey)
    assert _inferred(proj) == []

    # ...and the same two slabs NOT mitered do overlap and do carve, so the
    # assertions above are about the suppression rather than about the fixture.
    a2, b2 = Slab(name="a"), Slab(name="b")
    a2.add(Box(name="body", start=Point(x=0, y=0, z=0),
               end=Point(x=RUN, y=DEPTH, z=300)))
    b2.add(Box(name="body", start=Point(x=0, y=0, z=0),
               end=Point(x=DEPTH, y=RUN, z=300)))
    plain = Project(name="unmitered")
    st = Storey(name="ground", elevation=0)
    st.add(a2)
    st.add(b2)
    plain.add_storey(st)
    assert len(_inferred(plain)) == 1


@pytest.mark.parametrize("carve", ["self", "other", "none"])
def test_an_authored_joint_outranks_every_carve_direction(carve):
    """Precedence, pinned.

    ``carve=`` names who loses material in an overlap; ``miter()`` says the
    overlap is already resolved. The joint wins for all three directions — an
    inverted carve is still a carve, and there is none to invert.

    True today only because one ``if`` sits above another in one loop, which is
    not a thing to leave unasserted across the two features that read it.
    """
    assert _inferred(_mitered_pair(carve_b=carve)) == []


def test_carve_self_still_inverts_a_pair_that_is_NOT_mitered():
    """The other half of the precedence: suppressing the joint must not also
    suppress ``carve="self"`` everywhere else in the model."""
    a = Box(name="plate_a", start=Point(x=0, y=0, z=0),
            end=Point(x=6000, y=4000, z=300))
    b = Box(name="plate_b", start=Point(x=3000, y=0, z=0),
            end=Point(x=9000, y=4000, z=300))
    proj = Project(name="not-mitered")
    storey = Storey(name="ground", elevation=0)
    storey.add(a)
    storey.add(b, carve="self")
    proj.add_storey(storey)
    assert _inferred(proj) == [("box:plate_b:storey:ground",
                                "box:plate_a:storey:ground")]
