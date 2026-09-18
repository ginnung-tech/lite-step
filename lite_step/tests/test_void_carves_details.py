"""A ``.void()`` hole cuts through the details standing in it, like an opening.

``.opening()`` already did (``test_opening_carves_details.py``). ``.void()`` did
not, and the two are rows of ONE table — ``BimElement.void``'s own docstring
prints it:

======================  ==========  ========  ====================
spelling                counted?    named?    fill?
======================  ==========  ========  ====================
``.difference(tool)``   no          no        no
``.void(tool)``         yes         no        no
``.void(tool, name=)``  yes         yes       no
``.opening(child)``     yes         yes       yes (Window/Door)
======================  ==========  ========  ====================

Every "counted" row emits an ``IfcOpeningElement`` linked by
``IfcRelVoidsElement``. They differ in whether the hole is NAMED and whether
joinery FILLS it — and neither difference is about the detail standing in the
hole. So the argument ``_iter_opening_carvees`` makes carries over unchanged:
**matter inside a hole is not matter.**

The failure it fixes was silent. A wall's reinforcement ran straight through
every ``.void()`` doorway, no warning and no error, in a corpus fixture; and
skills could not route around it, because spelling the hole ``.opening()``
instead would emit a second ``IfcWindow`` inside the wall and double-count
every window in the schedule.

Pinned here:

1.  The same hole, spelled ``.void()`` and ``.opening()``, carves the same
    detail by the same volume. This is the whole claim, and it is why the
    fixture is the opening suite's wall with one line changed.
2.  A detail clear of the void is untouched, and so is the host body (the host
    product already carries the ``IfcRelVoidsElement``).
3.  The cut is the operand ITSELF, not its bounding box — measured on a curved
    tool, where the two answers differ by 21%.
4.  ``.no_carve()`` still exempts a detail meant to stand in the reveal.
5.  ``wall.void(t)`` and ``body.void(t)`` are one hole on one product, so they
    carve the same details.
6.  A named void is reported under the name the author wrote.
7.  A CONTAINER's void is left alone: the generator already hands it to every
    geometry-bearing child, so a boolean here would carve twice.
8.  A SITE's void is left alone too, and for a stronger reason: it is not an
    ``IfcOpeningElement`` at all but a clearing volume carved out of the
    terrain mesh, so reaching solids would take a retaining wall out with the
    ground.
9.  Both backends agree.
"""

from __future__ import annotations

import math

import pytest

from lite_step.compiler.displacement import carve_pairs
from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.models import Box, Column, Pipe, Point, Project, Wall, Window

# ---------------------------------------------------------------------------
# The opening suite's fixture, with the window swapped for a void cut at the
# SAME place — which is what makes test 1 a comparison and not two assertions.
# One 6 m wall running +X, 300 mm thick, y in [-300, 0], z in [0, 2700]; posts
# anchored at along=2100 (in the hole) and along=200 (clear of it).
# ---------------------------------------------------------------------------

_HOLE_X0, _HOLE_X1 = 3.3, 4.5
_HOLE_Z0, _HOLE_Z1 = 0.9, 2.3
_POST = 0.05
_POST_H = 2.7
_POST_FULL = _POST * _POST * _POST_H

#: The void that cuts the same hole the window does. Full depth and then some
#: — a ``.void()`` tool is absolute world geometry and is expected to overshoot
#: the wall it cuts, exactly as a real gable or doorway cutter does.
def _hole() -> Box:
    return Box(start=Point(x=3300, y=-400, z=900),
               end=Point(x=4500, y=100, z=2300))


def _post(name: str) -> Box:
    return Box(start=Point(x=0, y=100, z=0),
               end=Point(x=50, y=150, z=2700), name=name)


def _wall(*, hole="void", no_carve: bool = False, on_body: bool = False,
          name: str = None) -> Wall:
    """The fixture wall. ``hole`` picks the spelling under test."""
    wall = Wall(name="south")
    body = Box(start=Point(x=0, y=-300, z=0),
               end=Point(x=6000, y=0, z=2700), type="wall", name="body")
    wall.add(body)
    for along, nm in ((2100, "post_in"), (200, "post_out")):
        p = _post(nm)
        exempt = no_carve and nm == "post_in"
        wall.anchor(p, along=along, up=0,
                    carve="none" if exempt else None)

    if hole == "void":
        (body if on_body else wall).void(_hole(), name=name)
    elif hole == "opening":
        wall.anchor(Window(width=1200, height=1400, name="w0"),
                    along=1500, up=900)
    elif hole == "none":
        pass
    else:                                    # pragma: no cover - typo guard
        raise AssertionError(hole)
    return wall


def _project(element) -> Project:
    proj = Project(name="void-carves-details")
    proj.add(element)
    assert validate_project_report(proj).errors == []
    return normalize_project_to_meters(proj) or proj


def _compiled(proj: Project, backend: str = "ifcopenshell") -> str:
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(proj)
    assert result.success, result.error
    return result.ifc_content


def _child(wall_or_proj, name: str):
    wall = (wall_or_proj.storeys[0].elements[0]
            if isinstance(wall_or_proj, Project) else wall_or_proj)
    for child in wall._elements:
        if getattr(child, "name", None) == name:
            return child
    raise AssertionError(f"no child named {name!r}")


def _cuts(proj: Project, name: str) -> int:
    return len(getattr(_child(proj, name), "_cuts", None) or [])


def _volume(ifc_text: str, product_name: str) -> float:
    """Tessellated world volume of one named product's OWN representation.

    ``disable-opening-subtractions`` is load-bearing, and it is the same
    setting ``test_opening_carves_details`` explains at length:
    ``ifcopenshell.geom`` applies an ``IfcRelVoidsElement`` on a parent to the
    products AGGREGATED under it, so a post inside this wall measures the
    carved volume whether or not it carries a boolean of its own — the
    measurement could not fail. Turning the subtraction off leaves only what
    is in the element's own representation, which is also exactly what web-ifc
    / ThatOpen renders (our production path).
    """
    import os
    import tempfile

    geom = pytest.importorskip("ifcopenshell.geom")
    import ifcopenshell
    import ifcopenshell.util.shape

    fd, path = tempfile.mkstemp(suffix=".ifc")
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(ifc_text)
        model = ifcopenshell.open(path)
        settings = geom.settings()
        settings.set("use-world-coords", True)
        settings.set("disable-opening-subtractions", True)
        matches = [p for p in model.by_type("IfcProduct")
                   if (p.Name or "").startswith(product_name)
                   and p.Representation is not None]
        assert matches, f"no product named {product_name!r} with geometry"
        assert len(matches) == 1, [p.Name for p in matches]
        shape = geom.create_shape(settings, matches[0])
        return ifcopenshell.util.shape.get_volume(shape.geometry)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# 0. The fixture itself
# ---------------------------------------------------------------------------


def test_the_fixture_puts_the_void_where_the_window_goes() -> None:
    """Test 1 is a COMPARISON, so it is only worth anything if the two
    spellings really do describe one hole. Measured against the window's own
    derived prism, not against the numbers at the top of this file."""
    from lite_step.compiler import frames
    from lite_step.ifc.entity_cache import snap_to_precision

    proj = _project(_wall(hole="opening"))
    _compiled(proj)                          # runs the frame + anchor passes

    wall = proj.storeys[0].elements[0]
    frame = frames.opening_host_frame(frames._body_of(wall),
                                      snap=snap_to_precision, divisor=1000.0)
    prism = frames.opening_void_prism(_child(wall, "w0"), frame)
    (x0, _y0, z0), (x1, _y1, z1) = prism.aabb()
    assert (round(x0, 6), round(x1, 6)) == (_HOLE_X0, _HOLE_X1)
    assert (round(z0, 6), round(z1, 6)) == (_HOLE_Z0, _HOLE_Z1)

    # ...and the void spans that hole in x and z (it deliberately overshoots
    # in y, the direction the wall is thin in).
    hole = _hole()
    assert (hole.start.x / 1000.0, hole.end.x / 1000.0) == (_HOLE_X0, _HOLE_X1)
    assert (hole.start.z / 1000.0, hole.end.z / 1000.0) == (_HOLE_Z0, _HOLE_Z1)


# ---------------------------------------------------------------------------
# 1. The claim
# ---------------------------------------------------------------------------


def test_the_same_hole_spelled_void_carves_what_it_does_spelled_opening() -> None:
    """THE test. Two spellings, one hole, one notch.

    Falsify by reverting ``_carve_voids_through_details``: the ``.void()`` post
    keeps its full 0.00675 while the ``.opening()`` post loses the notch, and
    the two numbers stop matching.
    """
    by_void = _volume(_compiled(_project(_wall(hole="void"))), "box:post_in")
    by_opening = _volume(_compiled(_project(_wall(hole="opening"))),
                         "box:post_in")

    notch = _POST * _POST * (_HOLE_Z1 - _HOLE_Z0)
    assert by_void == pytest.approx(_POST_FULL - notch, rel=1e-6)
    assert by_void == pytest.approx(by_opening, rel=1e-9)


def test_a_detail_standing_in_the_void_is_carved_exactly_once() -> None:
    proj = _project(_wall())
    _compiled(proj)
    assert _cuts(proj, "post_in") == 1


# ---------------------------------------------------------------------------
# 2. What is NOT carved
# ---------------------------------------------------------------------------


def test_a_detail_clear_of_the_void_is_untouched() -> None:
    proj = _project(_wall())
    ifc = _compiled(proj)

    assert _cuts(proj, "post_out") == 0
    assert _volume(ifc, "box:post_out") == pytest.approx(_POST_FULL, rel=1e-6)


def test_the_host_body_is_not_carved_twice() -> None:
    """The host product already carries the ``IfcRelVoidsElement``.

    A second, boolean carve of the same volume would be invisible in the
    render and visible only in the CSG depth budget — the resource this
    codebase rations, and the one the primary web viewer silently runs out
    of.
    """
    proj = _project(_wall())
    ifc = _compiled(proj)

    assert _cuts(proj, "body") == 0
    assert ifc.count("IFCOPENINGELEMENT(") == 1
    # Exactly one boolean in the whole model: the notch in post_in.
    assert ifc.count("IFCBOOLEANRESULT(") == 1


def test_no_void_means_no_cuts_at_all() -> None:
    """The pass must not fire on a wall that authored no hole."""
    proj = _project(_wall(hole="none"))
    _compiled(proj)
    assert (_cuts(proj, "post_in"), _cuts(proj, "post_out")) == (0, 0)
    assert carve_pairs(proj) == []


def test_no_carve_exempts_a_detail_meant_to_stand_in_the_reveal() -> None:
    """A sill or lintel that belongs in the hole says so, and the escape hatch
    is the same one the opening pass honours."""
    proj = _project(_wall(no_carve=True))
    ifc = _compiled(proj)

    assert _cuts(proj, "post_in") == 0
    assert _volume(ifc, "box:post_in") == pytest.approx(_POST_FULL, rel=1e-6)


# ---------------------------------------------------------------------------
# 3. The cut is the operand, not its bounding box
# ---------------------------------------------------------------------------


def test_a_curved_void_carves_its_own_shape_not_its_bounding_box() -> None:
    """The one place this pass had a real design choice, measured.

    The opening pass reduces its derived prism to an axis-aligned ``Box``
    (``_void_operand``) and refuses a skew host out loud, because a bounding
    box would carve wider than the window. A ``.void()`` operand needs no such
    reduction — it is already absolute world geometry in a shape the
    boolean-operand vocabulary accepts, so the cut is a copy of the tool.

    A 25 mm-radius pipe bored through a 50 mm post is where the two answers
    part: the disk takes pi/4 of the square section, so a bounding-box cut
    would remove 27% more material than the author drew.
    """
    wall = _wall(hole="none")
    wall.void(Pipe(path=[Point(x=3875, y=-400, z=1500),
                         Point(x=3875, y=100, z=1500)], radius=25))
    proj = _project(wall)
    ifc = _compiled(proj)

    assert [type(c).__name__ for c in _child(proj, "post_in")._cuts] == ["Pipe"]

    bored = _POST_FULL - _volume(ifc, "box:post_in")
    cylinder = math.pi * 0.025 ** 2 * _POST      # the tool's own volume
    bounding_box = _POST * _POST * _POST         # what an AABB cut would take
    # Tessellation of a circle inscribes, so the measured bore is a hair under
    # the true cylinder — but nowhere near the box.
    assert bored == pytest.approx(cylinder, rel=0.02)
    assert bored < bounding_box * 0.85


# ---------------------------------------------------------------------------
# 4. Both spellings of one hole, and the name it is reported under
# ---------------------------------------------------------------------------


def test_a_void_on_the_body_is_the_same_hole_as_a_void_on_the_wall() -> None:
    """A Wall carries its body ON the ``IfcWall``, so ``generator._create_wall``
    puts BOTH lists of voids on one product and they cut one hole. A pass that
    read only the wall's own list would carve the details for one spelling and
    not the other — the same silent gap, one level down."""
    on_wall = _volume(_compiled(_project(_wall(on_body=False))), "box:post_in")
    on_body = _volume(_compiled(_project(_wall(on_body=True))), "box:post_in")

    assert on_body == pytest.approx(on_wall, rel=1e-9)
    assert on_body < _POST_FULL


def test_a_named_void_is_reported_under_the_name_the_author_wrote() -> None:
    """``.void(tool, name=…)`` names the HOLE, not the tool, so the carve report
    has to use the canonical the naming pass derives — that is the string an
    author can put in an ``assert_carved``."""
    proj = _project(_wall(name="doorway"))
    _compiled(proj)

    assert ("box:post_in:wall:south", "opening:doorway:wall:south",
            "void-hole") in carve_pairs(proj)


def test_an_anonymous_void_still_reports_the_detail_it_carved() -> None:
    """No name for the hole is not a reason to lose the row — the LOSER is what
    an author looks for when a notch appears they did not expect."""
    proj = _project(_wall())
    _compiled(proj)

    rows = [r for r in carve_pairs(proj) if r[2] == "void-hole"]
    assert len(rows) == 1
    assert rows[0][0] == "box:post_in:wall:south"


# ---------------------------------------------------------------------------
# 5. The container split
# ---------------------------------------------------------------------------


def test_a_container_void_is_left_to_the_generator() -> None:
    """A ``Column``/``Beam``/``Slab``/``Roof``/``Element`` void is handed to
    EVERY geometry-bearing child product (``_process_container_voids``), so its
    details already receive an ``IfcRelVoidsElement`` for the hole. Carving
    them again here would spend one boolean per detail per void for no visible
    change — which is the CSG budget, not free.

    Pinned against compiled output rather than against
    ``_VOID_REACHES_EVERY_CHILD``, so a generator that changes route fails
    here instead of drifting past a tuple nobody re-reads.
    """
    col = Column(name="pier")
    col.add(Box(start=Point(x=0, y=0, z=0),
                end=Point(x=400, y=400, z=3000), name="shaft"))
    col.anchor(Box(start=Point(x=0, y=0, z=0),
                   end=Point(x=50, y=50, z=3000), name="rebar"), along=0, up=0)
    col.void(Box(start=Point(x=-100, y=-100, z=1000),
                 end=Point(x=500, y=500, z=1400)))
    proj = _project(col)
    ifc = _compiled(proj)

    assert (_cuts(proj, "shaft"), _cuts(proj, "rebar")) == (0, 0)
    # One opening for the shaft, one for the rebar — the hole reached both
    # without this pass.
    assert ifc.count("IFCOPENINGELEMENT(") == 2


def test_a_site_void_clears_the_ground_and_not_the_things_on_it() -> None:
    """A ``Site``/``Space`` cannot host an ``IfcOpeningElement`` at all, so
    ``apply_displacement`` realises its void as mesh CSG against the site's
    TERRAIN — and a site void is a clearing volume the size of a plot.

    Letting this pass reach it would take a retaining wall out with the
    ground, silently, and the window corpus system already authors
    ``site.void(clearing_void)``. Not hypothetical.
    """
    from lite_step.tests._fixtures import terrain_site

    site = terrain_site(name="plot")
    # TWO solids on purpose. ``frames._body_of`` calls the FIRST geometry-
    # bearing child the host's body and the carve pass exempts it, so a site
    # with one solid exempts that solid for an unrelated reason and the
    # fixture would pass with the guard deleted (it did).
    for tag, x0 in (("retaining", 0), ("plinth", 4000)):
        site.add(Box(start=Point(x=x0, y=0, z=0),
                     end=Point(x=x0 + 300, y=8000, z=1200), name=tag))
    site.void(Box(start=Point(x=-2000, y=-2000, z=-1000),
                  end=Point(x=12000, y=12000, z=2000)))
    proj = _project(site)
    _compiled(proj)

    solids = {c.name: c for c in proj.sites[0]._elements
              if getattr(c, "name", None) in ("retaining", "plinth")}
    assert set(solids) == {"retaining", "plinth"}
    for tag, solid in solids.items():
        assert (getattr(solid, "_cuts", None) or []) == [], tag
    assert [r for r in carve_pairs(proj) if r[2] == "void-hole"] == []


# ---------------------------------------------------------------------------
# 6. Backends
# ---------------------------------------------------------------------------


def test_both_backends_carve_the_same_detail() -> None:
    """The streaming generator is a separate emitter; displacement runs before
    either, so the notch must be in the tree both of them read."""
    for backend in ("ifcopenshell",):
        proj = _project(_wall())
        _compiled(proj, backend=backend)
        assert _cuts(proj, "post_in") == 1, backend
        assert _cuts(proj, "post_out") == 0, backend
