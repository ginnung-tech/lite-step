"""An ``IfcSpace`` stands where its body was authored.

``_create_space`` transfers the first prism child's Representation onto the
``IfcSpace`` and drops the donor product. Dropping the donor's
**ObjectPlacement** with it — and that placement is half the geometry:
``_create_box`` builds the solid centred on LOCAL ``(0,0,0)`` and puts the
box's world centre in the placement. So the space kept the identity placement
it was created with, and every room rendered at world origin.

**Nothing downstream could catch it.** The file is valid, the space has a
Representation, the conformance gate passes at 0 errors, and the model
RENDERS. Only the position is wrong. That is the
``IfcBuildingStorey.Elevation`` shape: a gate proves the slot exists, never
that what is in it is true — so this measures the number.
"""
from __future__ import annotations

import os
import tempfile

import pytest

ifcopenshell = pytest.importorskip("ifcopenshell")
from ifcopenshell.util.placement import get_local_placement  # noqa: E402

from lite_step.compiler.executor import normalize_project_to_meters  # noqa: E402
from lite_step.ifc.generator import generate_ifc  # noqa: E402
from lite_step.models import Box, Point, Project, Space  # noqa: E402
from lite_step.models.project import Storey  # noqa: E402


#: (leaf, x0, x1) in authored millimetres. Deliberately ASYMMETRIC about the
#: origin in x and lifted off z=0: a room centred on (0,0,0) would sit at the
#: identity placement anyway, and pass whatever the code does.
ROOMS = (
    ("living_room", -4000, 450),
    ("bedroom", 600, 4000),
)
Y0, Y1, Z0, Z1 = -2000, 2000, 0, 2800


def _model():
    proj = Project(name="spaces")
    storey = Storey(name="ground", elevation=0)
    for leaf, x0, x1 in ROOMS:
        space = Space(name=leaf)
        space.add(Box(name="body", start=Point(x=x0, y=Y0, z=Z0),
                      end=Point(x=x1, y=Y1, z=Z1)))
        storey.add(space)
    proj.add_storey(storey)

    res = generate_ifc(normalize_project_to_meters(proj), source_code=None)
    assert res.success, res.error
    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w") as fh:
            fh.write(res.ifc_content)
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _origin_of(product):
    m = get_local_placement(product.ObjectPlacement)
    return (m[0][3], m[1][3], m[2][3])


def test_a_space_stands_at_its_authored_centre():
    """The measurement. Before the fix both rooms reported ``(0, 0, 0)``."""
    model = _model()
    spaces = {s.Name.split(":")[1]: s for s in model.by_type("IfcSpace")}
    assert set(spaces) == {leaf for leaf, _, _ in ROOMS}

    for leaf, x0, x1 in ROOMS:
        expected = ((x0 + x1) / 2 / 1000.0, (Y0 + Y1) / 2 / 1000.0,
                    (Z0 + Z1) / 2 / 1000.0)
        got = _origin_of(spaces[leaf])
        assert got == pytest.approx(expected, abs=1e-9), (
            f"{leaf} placed at {got}, authored centre {expected} — the space "
            f"is standing somewhere its body was never authored"
        )


def test_a_space_still_carries_its_own_representation():
    """The donor transfer itself must survive the placement fix: a space with
    a placement and no shape is the same bug pointing the other way."""
    model = _model()
    for space in model.by_type("IfcSpace"):
        assert space.Representation is not None, space.Name


def test_the_donor_leaves_no_orphan_behind():
    """The donor product goes, and so does the identity placement the space
    was created with — but only when nothing chained to it.

    Counted rather than assumed: reassigning the space's placement without
    dropping the one it replaced leaves an ``IfcLocalPlacement`` that places
    nothing, and reassigning it while a child still chains to it would leave a
    dangling reference instead.
    """
    model = _model()
    orphans = [lp for lp in model.by_type("IfcLocalPlacement")
               if not lp.PlacesObject and not lp.ReferencedByPlacements]
    assert orphans == [], f"{len(orphans)} IfcLocalPlacement placing nothing"
    # the donor proxies are gone (only the two spaces carry the bodies)
    assert not model.by_type("IfcBuildingElementProxy")
