"""The compiler prints the container tree it built, on every compile.

``carves: N pairs`` and ``csg depth: max=N`` say what the compiler DID. Neither
says what SHAPE it did it to, so a leaf that landed on the wrong container was
learned about from geometry that looked wrong — or not at all.

What is pinned here, and why each one:

1.  The exact rendering of a known fixture. A report is a STRING contract: an
    assertion on "some line mentions the wall" passes on a report that has lost
    its nesting, its verbs and its units.
2.  **Units.** Everything here runs on the metre tree, and the extents are
    integer millimetres. The fixture's extents are deliberately NOT whole
    metres (306, 508, 45, 145, 1460) because the guard that catches a forgotten
    ``divisor`` is partial by construction — a container whose extents are all
    whole metres rounds cleanly and nothing can see the error. Both divisors
    are pinned: the ``Bounds`` one (which RAISES) and the ``element_extent``
    one (which does not — it silently inflates a ``Material(profile_mm=)`` pad
    by 1000x), and there is a control test proving each fixture value really is
    in the band where its guard fires.
3.  Collapse: identical siblings become one row and a count; siblings that
    differ anywhere in their SUBTREE do not.
4.  The three env settings, which are ``LITESTEP_CARVE_REPORT``'s three.
5.  That the bounded default says what it dropped, and that the counts it keeps
    on a host (``openings=``) survive the truncation that drops the opening's
    own row — which is the whole reason those counts are on the host.
6.  That a report which RAISES does not fail a compile whose geometry is fine.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.tree_report import _MAX_ROWS, tree_report_lines
from lite_step.models import (
    Box, Column, Element, Material, Point, Project, Storey, Sweep, Wall, Window,
)
from lite_step.tests._fixtures import terrain_site

# ---------------------------------------------------------------------------
# The fixture. Every number in it is load-bearing:
#
#   306   the wall's thickness — a real extent that ROUNDS TO ZERO when the
#         metre tree is read at divisor=1.0, so ``Bounds`` raises on it
#   508   the stud bay's thickness, ditto, and the own measured number
#   45x145x2700  the studs — a run of identical siblings to collapse
#   1460x1230    the window — mm scalars the normalizer scales, so reading
#                them raw prints "1x1 mm opening"
#   Timber_C24 (45, 195)  a member section, i.e. a frozen MILLIMETRE fact,
#         which is what ``element_extent``'s divisor exists for
# ---------------------------------------------------------------------------

#: Half-diagonal of the ``Timber_C24`` (45, 195) section, in mm — what
#: ``extent._member_section_reach`` pads a member-form ``Sweep`` by. At the
#: WRONG divisor this pad is 1000x larger, which is what
#: ``test_a_member_sections_pad_is_millimetres_not_metres`` measures.
_SECTION_REACH_MM = round(0.5 * (45.0 ** 2 + 195.0 ** 2) ** 0.5)


def _studbay() -> Column:
    """A nested container holding a run of interchangeable siblings."""
    bay = Column(name="bay")
    for i in range(4):
        x0 = 400 + i * 600
        bay.add(Box(start=Point(x=x0, y=-508, z=0),
                    end=Point(x=x0 + 45, y=-363, z=2700), name=f"stud_{i:02d}"))
    return bay


def _wall() -> Wall:
    wall = Wall(name="south")
    wall.add(Box(start=Point(x=0, y=-306, z=0), end=Point(x=6000, y=0, z=3000),
                 type="wall", name="body"))
    wall.add(_studbay())
    # An opening on the HOST, and an .anchor()ed detail beside it, so the row
    # marker has something to be wrong about in both directions.
    wall.anchor(Window(width=1460, height=1230, name="w0"), along=1500, up=900)
    wall.anchor(Box(start=Point(x=0, y=0, z=0), end=Point(x=508, y=60, z=200),
                    name="corbel"), along=300, up=2400)
    wall.void(Box(start=Point(x=5000, y=-400, z=0),
                  end=Point(x=5400, y=100, z=2100)), name="doorway")
    return wall


def _rafters() -> Column:
    """A body-less container (no ``Box``/``Extrude`` of its own) whose children
    are member-form ``Sweep``\\ s — i.e. the ``Material(profile_mm=)`` pad."""
    col = Column(name="truss")
    for i in range(2):
        y = 1000 + i * 2000
        col.add(Sweep(path=[Point(x=0, y=y, z=3000), Point(x=4000, y=y, z=3000)],
                      material=Material(key="Timber_C24", profile_mm=(45, 195)),
                      name=f"rafter_{i}"))
    return col


def _project() -> Project:
    proj = Project(name="tree-report")
    site = terrain_site(name="plot")
    proj.add(site)
    storey = Storey(name="ground", elevation=0)
    storey.add(_wall())
    storey.add(_rafters())
    proj.storeys.append(storey)
    assert validate_project_report(proj).errors == []
    return normalize_project_to_meters(proj) or proj


def _compiled(proj: Project, backend: str = "ifcopenshell"):
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(proj)
    assert result.success, result.error
    return result


def _rendered(proj: Project = None, **kw) -> list:
    """The report as the compile produces it — COMPILED first, deliberately.

    ``generate_ifc`` runs the naming, frame, anchor-bake and displacement
    passes before it prints, and the report reads all four. Calling
    ``tree_report_lines`` on a merely-normalized project answers from a
    half-built tree: an ``.anchor()``ed child still carries an unresolved
    ``_anchor_spec``, which ``iter_geometry_points`` skips, so it reports
    ``(no extent)`` for a Box with corners on it. Testing the shortcut would
    have pinned the wrong string.
    """
    proj = _project() if proj is None else proj
    _compiled(proj)
    return tree_report_lines(proj, **kw)


def _tree(err: str) -> list:
    """The tree block out of a captured stderr — it shares the stream with the
    carve, CSG-depth and space-boundary reports."""
    out, keep = [], False
    for line in err.splitlines():
        if line.startswith("tree:"):
            keep = True
        elif keep and not line.startswith((" ", "@")):
            keep = False
        if keep:
            out.append(line)
    return out


# ---------------------------------------------------------------------------
# 1. The rendering
# ---------------------------------------------------------------------------


def test_the_report_renders_the_tree_the_fixture_builds() -> None:
    """THE test. The whole string, for a tree small enough to write out.

    Every claim this report makes is in here at once: nesting by indentation,
    the ``@`` verb marker, canonical ``type:leaf`` labels, millimetre extents,
    the collapsed stud run, the folded body, ``(no body)``, the opening and
    void counts, and the generic ``Element``'s IFC class.
    """
    assert _rendered() == [
        "tree: tree-report  16 nodes  (@ = .anchor()ed, extents in mm)",
        " site:plot  32000x29000x15000 mm",
        "   geographicelement:terrain  32000x29000x15000 mm  "
        "IfcGeographicElement  (no body)  +mesh",
        " storey:ground  elev=0 mm",
        "   wall:south  ~6000x508x3000 mm  openings=1  voids=1",
        "     box:body  6000x306x3000 mm",
        "     column:bay  1845x145x2700 mm",
        "       box:stud_00 .. stud_03  x4  45x145x2700 mm",
        "    @window:w0  1460x1230 mm opening  along=1500 up=900 mm",
        "    @box:corbel  508x60x200 mm",
        "   column:truss  ~4000x2045x195 mm  (no body)",
        "     sweep:rafter_0 .. rafter_1  x2  ~4000x45x195 mm",
    ]


def test_the_leading_tilde_says_the_number_is_a_bound(capsys) -> None:
    """``~`` is ``Bounds.exact == False``, which ``compiler.extent`` already
    surfaces rather than hides — "a loose bound presented as an extent is a
    plausible wrong number".

    Both of the fixture's ``~`` rows earn it for a stated reason, and neither
    reason is the row's own shape: ``wall:south`` because it contains an
    OPENING, ``column:truss`` and its rafters because a member-form ``Sweep``'s
    section is padded. The plain ``box:`` rows beside them have no tilde, which
    is what makes the marker informative rather than decorative.
    """
    from lite_step.compiler.extent import extent_is_exact

    proj = _project()
    _compiled(proj)
    wall = proj.storeys[0].elements[0]
    truss = proj.storeys[0].elements[1]
    assert not extent_is_exact(wall)
    assert not extent_is_exact(truss)
    assert extent_is_exact(wall.elements[0])            # box:body

    rows = _rendered(proj)
    assert [r for r in rows if "~" in r] == [
        "   wall:south  ~6000x508x3000 mm  openings=1  voids=1",
        "   column:truss  ~4000x2045x195 mm  (no body)",
        "     sweep:rafter_0 .. rafter_1  x2  ~4000x45x195 mm",
    ]


def test_the_tree_reaches_stderr_on_an_ordinary_compile(capsys) -> None:
    """The report is worth nothing if it only exists when a test calls it. Same
    channel as its two siblings, and beside them."""
    _compiled(_project())
    err = capsys.readouterr().err

    assert err.count("carves: ") == 1
    assert err.count("csg depth: ") == 1
    assert _tree(err)[0].startswith("tree: tree-report  16 nodes")


# ---------------------------------------------------------------------------
# 2. Units — the trap, and the controls that prove the fixture can see it
# ---------------------------------------------------------------------------


def test_a_sub_metre_thickness_is_reported_in_millimetres() -> None:
    """306 mm, not 0 mm and not 1 mm. This is one level up."""
    rows = "\n".join(_rendered())
    assert "box:body  6000x306x3000 mm" in rows
    assert "box:stud_00 .. stud_03  x4  45x145x2700 mm" in rows


def test_the_fixtures_thicknesses_are_ones_a_forgotten_divisor_raises_on() -> None:
    """The CONTROL for the test above: a fixture at whole-metre extents cannot
    tell a correct divisor from a coincidence.

    ``Bounds.__post_init__`` refuses an extent that is real but rounds to zero,
    which is what a metre-domain 0.306 does. Measured here rather than assumed,
    because the guard is documented as PARTIAL and the fixture's whole job is
    to land inside the part it covers.

    **And 508 is the part it does not cover, deliberately kept in the fixture.**
    A metre-domain 0.508 rounds to a perfectly ordinary 1 mm and nothing
    raises — so on that one number only the asserted STRING can tell a correct
    divisor from a coincidence. Both halves are here so that a reader does not
    take the raising half as the whole guarantee.
    """
    from lite_step.models.bounds import Bounds, BoundsError

    def box(thickness_m):
        return ((0.0, 0.0, 0.0), (6.0, thickness_m, 3.0))

    for thickness_m in (0.306, 0.045, 0.145):
        with pytest.raises(BoundsError, match="rounds to 0 mm"):
            Bounds.from_aabb(box(thickness_m))

    assert Bounds.from_aabb(box(0.508)).size.y == 1


def test_a_member_sections_pad_is_millimetres_not_metres() -> None:
    """The SECOND divisor, and the one no guard can see.

    ``Material(profile_mm=)`` is a frozen millimetre fact, so
    ``element_extent`` must be told the tree is in metres before it can divide
    the pad down. It is a PAD, not an extent, so a wrong divisor produces a
    perfectly well-formed ``Bounds`` — a 45x195 section reaching 45 and 195
    METRES around a 4 m rafter — and nothing raises. That is the whole reason
    ``Bounds``'s guard is documented as partial.

    The rafter is authored as a bare 4 m centreline, so 45 and 195 come from
    the material and from nowhere else; every digit in this row that is not
    4000 is the divisor being right.
    """
    rows = "\n".join(_rendered())
    assert "sweep:rafter_0 .. rafter_1  x2  ~4000x45x195 mm" in rows
    # …and what the wrong divisor would have printed, so the assertion above is
    # known to be discriminating rather than merely true.
    assert "45000" not in rows and "195000" not in rows


def test_an_openings_scalars_are_millimetres_too() -> None:
    """``width``/``height`` and the ``ChildAnchor`` scalars are ordinary mm
    dimensions the normalizer scaled, and they are NOT read through
    ``element_extent`` — so they need the conversion stated separately. Read
    raw, a 1460x1230 window at along=1500 printed ``1x1 mm opening`` at
    ``along=2 up=1``."""
    rows = "\n".join(_rendered())
    assert "@window:w0  1460x1230 mm opening  along=1500 up=900 mm" in rows


# ---------------------------------------------------------------------------
# 3. Collapse
# ---------------------------------------------------------------------------


def test_a_run_of_identical_siblings_is_one_row_and_a_count() -> None:
    studs = [r for r in _rendered() if "stud_" in r]
    assert studs == ["       box:stud_00 .. stud_03  x4  45x145x2700 mm"]


def test_siblings_that_differ_only_below_the_surface_do_not_collapse() -> None:
    """The signature is the whole SUBTREE, not the row.

    Two columns with identical extents whose CHILDREN differ are two different
    things, and a report that merged them would state ``x2`` about a shape only
    one of them has. Falsify by dropping the child tuple from
    ``_stamp_signatures``: these two collapse and the assertion fails.
    """
    proj = Project(name="pair")
    storey = Storey(name="g", elevation=0)
    for tag, kids in (("a", 1), ("b", 2)):
        col = Column(name=tag)
        col.add(Box(start=Point(x=0, y=0, z=0),
                    end=Point(x=400, y=400, z=3000), name="shaft"))
        for i in range(kids):
            col.anchor(Box(start=Point(x=0, y=0, z=0),
                           end=Point(x=45, y=45, z=2900), name=f"bar_{i}"),
                       along=0, up=0)
        storey.add(col)
    proj.storeys.append(storey)
    proj = normalize_project_to_meters(proj) or proj

    rows = _rendered(proj)
    assert [r for r in rows if r.startswith("   column")] == [
        "   column:a  400x400x3000 mm",
        "   column:b  400x400x3000 mm",
    ]


def test_an_added_and_an_anchored_child_never_collapse_together() -> None:
    """The verb is in the signature, because it is the distinction this report
    exists to make visible — two rows that differ ONLY by it are two facts."""
    proj = Project(name="verbs")
    storey = Storey(name="g", elevation=0)
    wall = Wall(name="w")
    wall.add(Box(start=Point(x=0, y=-306, z=0), end=Point(x=6000, y=0, z=3000),
                 type="wall", name="body"))
    tool = dict(start=Point(x=1000, y=-306, z=100),
                end=Point(x=1200, y=-246, z=300))
    wall.add(Box(name="added", **tool))
    wall.anchor(Box(start=Point(x=0, y=0, z=0), end=Point(x=200, y=60, z=200),
                    name="anchored"), along=2000, up=100)
    storey.add(wall)
    proj.storeys.append(storey)
    proj = normalize_project_to_meters(proj) or proj

    rows = [r for r in _rendered(proj) if "200x60x200" in r]
    assert rows == ["     box:added  200x60x200 mm",
                    "    @box:anchored  200x60x200 mm"]


# ---------------------------------------------------------------------------
# 4. The three settings
# ---------------------------------------------------------------------------


def _stderr_of_a_compile(monkeypatch, capsys, setting) -> str:
    if setting is None:
        monkeypatch.delenv("LITESTEP_TREE_REPORT", raising=False)
    else:
        monkeypatch.setenv("LITESTEP_TREE_REPORT", setting)
    _compiled(_deep_project())
    return capsys.readouterr().err


#: ``_deep_project``'s shape, as three numbers the assertions below derive from
#: rather than restate. 3 storeys x 5 walls x (a body + a bay of 3 studs).
_DEEP = (3, 5, 3)


def _deep_project() -> Project:
    """Wide and deep enough to exceed the default's caps.

    The walls are deliberately all DIFFERENT lengths, so none of them collapse:
    a fixture whose siblings all merge measures the collapse and never reaches
    the cap it was built to exercise. The studs inside each one are identical,
    so both halves are covered by one tree. Storeys are stacked in Z as well as
    in ``elevation`` — geometry is authored in world coordinates here, so three
    storeys at one Z is three walls occupying one space, and the displacement
    pass says so at length.
    """
    storeys, walls, studs = _DEEP
    proj = Project(name="wide")
    for s in range(storeys):
        z0 = s * 3000
        storey = Storey(name=f"level_{s}", elevation=z0)
        for w in range(walls):
            length = 4000 + w * 500
            wall = Wall(name=f"w{w}")
            wall.add(Box(start=Point(x=w * 7000, y=-306, z=z0),
                         end=Point(x=w * 7000 + length, y=0, z=z0 + 3000),
                         type="wall", name="body"))
            bay = Column(name="bay")
            for i in range(studs):
                x0 = w * 7000 + 400 + i * 600
                bay.add(Box(start=Point(x=x0, y=-508, z=z0),
                            end=Point(x=x0 + 45, y=-363, z=z0 + 2700),
                            name=f"stud_{i:02d}"))
            wall.add(bay)
            storey.add(wall)
        proj.storeys.append(storey)
    assert validate_project_report(proj).errors == []
    return normalize_project_to_meters(proj) or proj


#: Nodes in ``_deep_project``: per storey, a storey + per wall a wall, its
#: body, its bay and its studs.
_DEEP_NODES = _DEEP[0] * (1 + _DEEP[1] * (3 + _DEEP[2]))
#: Rows an UNCAPPED render costs: a storey, then per wall the wall, its body,
#: its bay and ONE collapsed stud row.
_DEEP_ROWS_ALL = _DEEP[0] * (1 + _DEEP[1] * 4)


def test_the_default_is_the_bounded_tree(monkeypatch, capsys) -> None:
    rows = _tree(_stderr_of_a_compile(monkeypatch, capsys, None))
    assert _DEEP_NODES == 93
    assert rows[0] == (f"tree: wide  {_DEEP_NODES} nodes  "
                       f"(@ = .anchor()ed, extents in mm)")
    assert len(rows) == 1 + _MAX_ROWS + 1        # header + rows + the notice


def test_all_prints_every_node(monkeypatch, capsys) -> None:
    rows = _tree(_stderr_of_a_compile(monkeypatch, capsys, "all"))
    assert _DEEP_ROWS_ALL == 63                  # …and the default showed 24
    assert not [r for r in rows if "more nodes not shown" in r]
    assert len(rows) == 1 + _DEEP_ROWS_ALL


def test_zero_is_silent(monkeypatch, capsys) -> None:
    """A corpus hash run compiles twenty models and reads
    their stderr; ``LITESTEP_CARVE_REPORT=0`` exists for exactly that, and a
    second report without the same escape makes the tooling noisier."""
    err = _stderr_of_a_compile(monkeypatch, capsys, "0")
    assert _tree(err) == []
    assert "carves: " in err                    # the compile really did happen


def test_an_unrecognised_setting_is_the_default(monkeypatch, capsys) -> None:
    """Same contract as ``LITESTEP_CARVE_REPORT``: only ``all`` and ``0`` mean
    anything, and a typo must not silence the report."""
    rows = _tree(_stderr_of_a_compile(monkeypatch, capsys, "yes"))
    assert len(rows) == 26


# ---------------------------------------------------------------------------
# 5. What the bound drops, and what it keeps
# ---------------------------------------------------------------------------


def test_the_default_states_how_many_nodes_it_dropped() -> None:
    """Every node is accounted for: printed as a row, folded into one, counted
    by an ``xN``, or named in the notice. Nothing falls between.

    ``_deep_project`` has no folds and no collapses among the rows the cap
    reaches, so the arithmetic here is the plain one; the fixture in
    ``test_every_node_is_accounted_for`` covers the other three terms.
    """
    rows = _rendered(_deep_project())
    shown = len(rows) - 2                        # header and the notice itself

    assert shown == _MAX_ROWS
    assert rows[-1] == (f"  +{_DEEP_NODES - _MAX_ROWS} more nodes not shown  "
                        f"(LITESTEP_TREE_REPORT=all for the whole tree)")


def test_every_node_is_accounted_for() -> None:
    """The header's count is the tree's; the rows either print a node, fold
    one, or count a run of them. A ``+N more`` that under-states what it cut is
    the failure this repo keeps fixing, so the books balance in a test.
    """
    import re

    rows = _rendered()
    total = int(re.search(r"  (\d+) nodes", rows[0]).group(1))
    printed = len(rows) - 1
    folded = sum(1 for r in rows if "  +" in r and "more nodes" not in r)
    collapsed = sum(int(m) - 1
                    for m in re.findall(r"  x(\d+)  ", "\n".join(rows)))

    assert not [r for r in rows if "more nodes not shown" in r]
    assert (printed, folded, collapsed, total) == (11, 1, 4, 16)
    assert printed + folded + collapsed == total


def test_the_bound_spends_its_budget_breadth_first() -> None:
    """A depth-first budget put every row into the first wall of the first
    storey and never mentioned that the model has three (measured on
    ``bridge-roadway``: 24 rows of one ``spatialelement:arch``'s diagonals, no
    storey row at all). "Did my ``.add()`` land on the wrong container" is a
    question about the TOP of the tree, so the top is what the default reaches
    first.

    Falsify by rendering depth-first: the storey count drops to 1.
    """
    rows = _rendered(_deep_project())
    assert len([r for r in rows if r.startswith(" storey:")]) == _DEEP[0]
    assert len([r for r in rows
                if r.startswith("   wall:")]) == _DEEP[0] * _DEEP[1]


def test_a_hosts_opening_count_survives_the_truncation_that_drops_its_row() -> None:
    """Why the counts are on the HOST and not only in the child rows.

    ``openings=`` is what an author most often gets wrong, and the opening's
    own row is a level below the host — the first thing a bounded default gives
    up. Thirty walls of THIRTY DIFFERENT lengths, so none of them collapse and
    the cap is really the thing doing the cutting.
    """
    wide = Project(name="many")
    storey = Storey(name="g", elevation=0)
    for w in range(30):
        wall = Wall(name=f"w{w}")
        wall.add(Box(start=Point(x=w * 7000, y=-306, z=0),
                     end=Point(x=w * 7000 + 4000 + w * 50, y=0, z=3000),
                     type="wall", name="body"))
        wall.anchor(Window(width=1460, height=1230, name="w0"),
                    along=1500, up=900)
        storey.add(wall)
    wide.storeys.append(storey)
    wide = normalize_project_to_meters(wide) or wide

    rows = _rendered(wide)
    assert not [r for r in rows if "window:w0" in r]        # dropped…
    assert len([r for r in rows if "openings=1" in r]) >= 1  # …but not lost


# ---------------------------------------------------------------------------
# 6. A report is never a reason a compile fails
# ---------------------------------------------------------------------------


def test_a_raising_report_does_not_fail_a_compile(monkeypatch, capsys) -> None:
    """"A bug in a REPORT must never take down a model whose geometry is fine."

    Not hypothetical here: this report reads ``Bounds``, which RAISES on an
    extent it cannot state honestly, so the raising path is live.
    """
    import lite_step.compiler.tree_report as tr

    def boom(_proj, **_kw):
        raise ValueError("deliberate")

    monkeypatch.setattr(tr, "tree_report_lines", boom)
    result = _compiled(_project())
    err = capsys.readouterr().err

    assert result.success
    assert "warning: container tree report failed (ValueError: deliberate)" in err
    assert _tree(err) == []


