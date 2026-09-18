"""``.parent`` back-reference + the shared-element check.

Two features that have to land together, because they are the two halves of
one statement about the model: *containment is a tree*.

Until now nothing in the compiler could walk UP. Every tree property
(``_canonical_name``, ``_frame``) is computed by a top-down pass precisely
because ``BimElement`` carried no parent pointer — ``compiler/naming.py:9``
and ``compiler/frames.py:16`` both say so in as many words. A world-coordinate
query has to reach the root from an arbitrary element, so it needs the
pointer.

And a back-reference is only meaningful if there is exactly ONE parent. The
compiler already assumed that and never enforced it: reach one object through
two containers and ``_canonical_name`` / ``_frame`` / ``_anchor_spec`` are
stamped twice, last write wins, nothing raises. So the pointer and the check
are the same change.

The load-bearing test here is
``test_anonymous_shared_element_has_no_pre_existing_error`` — it proves the
check is not redundant with the duplicate-canonical error that already exists.
Without it this whole module could be decoration.
"""
import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.compiler.frames import resolve_child_anchors
from lite_step.compiler.naming import find_shared_elements, stamp_canonical_names
from lite_step.models import (
    Beam,
    Box,
    Column,
    Door,
    Element,
    Extrude,
    Mesh,
    Point,
    Project,
    Roof,
    Site,
    Slab,
    Space,
    Storey,
    Wall,
    Window,
)


# Fixtures sit AWAY from the origin and off round numbers on purpose: a
# back-reference bug is invisible at the origin, and so is a placement one.
_X0, _Y0, _Z0 = 3170, -2410, 1130


def _box(name=None, dx=1210, dy=347, dz=2703):
    return Box(
        start=Point(x=_X0, y=_Y0, z=_Z0),
        end=Point(x=_X0 + dx, y=_Y0 + dy, z=_Z0 + dz),
        name=name,
    )


def _wall(name="south"):
    return Wall(name=name).add(_box(name="body"))


# ---------------------------------------------------------------------------
# .parent is set by every attachment path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "container,child",
    [
        (Wall(name="w"), _box(name="body")),
        (Column(name="c"), _box(name="body")),
        (Beam(name="b"), _box(name="body")),
        (Slab(name="s"), _box(name="body")),
        (Roof(name="r"), _box(name="body")),
        (Space(name="sp"), _box(name="body")),
        (Element(ifc_class="IfcCovering", name="e"), _box(name="body")),
        (Site(name="site"), _box(name="body")),
        (Window(width=1210, height=1403, name="win"), _box(name="pane")),
        (Door(width=903, height=2107, name="dr"), _box(name="leaf")),
    ],
    ids=lambda v: type(v).__name__,
)
def test_add_sets_parent_on_every_container(container, child):
    """Every ``.add()`` routes through one helper, so none can be missed.

    Parametrised over all ten containers rather than spot-checked: the
    asymmetry this guards against is real in this file already — ``Wall`` and
    ``Site`` still have no ``get_elements()`` because it was added
    container-by-container.
    """
    assert child.parent is None
    container.add(child)
    assert child.parent is container


def test_anchor_sets_parent():
    wall = _wall()
    win = Window(width=1210, height=1403, name="w0")
    wall.anchor(win, along=1507, up=903)
    assert win.parent is wall


def test_opening_sets_parent():
    body = _box(name="body")
    win = Window(width=1210, height=1403, name="w0")
    body.opening(win, along=307, up=903)
    assert win.parent is body


def test_storey_and_project_add_set_parent():
    wall = _wall()
    proj = Project(name="t")
    proj.add(wall)
    assert isinstance(wall.parent, Storey)
    assert wall.parent.parent is proj


def test_explicit_add_storey_sets_parent():
    storey = Storey(name="ground", elevation=0)
    proj = Project(name="t")
    proj.add_storey(storey)
    assert storey.parent is proj


def test_site_routed_to_sites_gets_the_project_as_parent():
    """``Project.add()`` routes a Site out of the storeys entirely, so it is
    the one top-level element whose parent is the Project itself."""
    site = Site(name="site")
    proj = Project(name="t")
    proj.add(site)
    assert proj.sites == [site]
    assert site.parent is proj


def test_parent_chain_reaches_the_root_from_a_nested_child():
    """The whole point: an arbitrary leaf can find the Project."""
    wall = _wall()
    body = wall._elements[0]
    proj = Project(name="t")
    proj.add(wall)

    node, chain = body, []
    while node is not None:
        chain.append(type(node).__name__)
        node = getattr(node, "parent", None)
    assert chain == ["Box", "Wall", "Storey", "Project"]


def test_boolean_operands_are_deliberately_not_parented():
    """An operand is consumed into its host's shape, not contained by it.

    Conflating the two would let a world query apply the host's placement to
    coordinates that already carry it. The shared-element check still covers
    operands — that is about the naming walk, not containment.
    """
    host, cutter = _box(name="host"), _box(name="cutter")
    host.difference(cutter)
    assert cutter.parent is None


# ---------------------------------------------------------------------------
# .parent survives the passes that rewrite the tree
# ---------------------------------------------------------------------------


def test_parent_survives_the_anchor_bake():
    """``resolve_child_anchors`` rewrites the child's coordinates in place —
    it must not drop the back-reference while doing so."""
    wall = _wall()
    win = Window(width=1210, height=1403, name="w0")
    wall.anchor(win, along=1507, up=903)
    proj = Project(name="t")
    proj.add(wall)

    resolve_child_anchors(proj)
    assert win.parent is wall


def test_parent_survives_normalize_and_points_INTO_the_copy():
    """``normalize_project_to_meters`` deep-copies the whole project.

    The copy's parents must point at the COPY, never back at the original —
    a back-reference that escapes its own tree would let a world query resolve
    against millimetre coordinates after normalization, which is the 1000x
    class of bug this repo has already paid for once.
    """
    wall = _wall()
    proj = Project(name="t")
    proj.add(wall)

    copied = normalize_project_to_meters(proj)
    copied_wall = copied.storeys[0].elements[0]
    copied_body = copied_wall._elements[0]

    assert copied_wall is not wall
    assert copied_body.parent is copied_wall
    assert copied_wall.parent is copied.storeys[0]
    assert copied_wall.parent.parent is copied
    # ...and nothing in the copy reaches the original.
    assert copied_body.parent is not wall
    assert copied.storeys[0].parent is not proj


def test_derivation_by_call_does_not_inherit_the_parent():
    """A derived copy is a NEW element that has not been attached anywhere.

    Also guards a performance trap: ``_parent`` points UP, so deep-copying it
    would drag the template's whole ancestor tree — and most of the model —
    through every ``instance(...)`` call. Derivation-by-call is the idiom the
    shared-element warning recommends, so it is on the hot path.
    """
    wall = _wall()
    proj = Project(name="t")
    proj.add(wall)
    body = wall._elements[0]
    assert body.parent is wall

    derived = body(name="body2")
    assert derived.parent is None
    # The template is untouched.
    assert body.parent is wall


def test_parent_agrees_with_the_naming_walk():
    """``.parent`` is not a second source of truth for containment.

    The canonical name is computed by a top-down walk; ``.parent`` is written
    at authoring time. They describe the same tree, so every element's
    canonical must be a suffix-extension of its parent's — if the two ever
    disagree, one of them is lying about where the element lives.
    """
    wall = _wall("north")
    win = Window(width=1210, height=1403, name="w0")
    wall.anchor(win, along=1507, up=903)
    proj = Project(name="t")
    proj.add(wall)
    stamp_canonical_names(proj)

    assert win.parent is wall
    assert win.ifc_name.endswith(wall.ifc_name)
    body = wall._elements[0]
    assert body.parent is wall
    assert body.ifc_name.endswith(wall.ifc_name)


# ---------------------------------------------------------------------------
# The shared-element check
# ---------------------------------------------------------------------------


def _two_covers(shared):
    """One object attached inside two different containers."""
    proj = Project(name="t")
    proj.add(
        Element(ifc_class="IfcCovering", name="a").add(shared),
        Element(ifc_class="IfcCovering", name="b").add(shared),
    )
    return proj


def test_shared_element_is_reported():
    proj = _two_covers(_box(name="shelf"))
    (warning,) = find_shared_elements(proj)
    assert "attached in 2 places" in warning
    assert "box:shelf" in warning


def test_the_warning_names_both_paths_distinctly():
    """Two attachment sites must render as two different strings.

    The first cut of this built the path label from the CHILD, so both
    placements printed identically (``box[elements#1], box[elements#1]``) and
    the message could not say where to look.
    """
    proj = _two_covers(_box(name="shelf"))
    (warning,) = find_shared_elements(proj)
    assert "covering:a" in warning
    assert "covering:b" in warning


def test_the_warning_names_the_surviving_canonical():
    """Last-write-wins is the actual damage, so say which one won."""
    proj = _two_covers(_box(name="shelf"))
    (warning,) = find_shared_elements(proj)
    assert "surviving canonical" in warning
    assert "box:shelf:covering:b" in warning


def test_anonymous_shared_element_has_no_pre_existing_error():
    """THE reason this check exists — and the test that could delete it.

    A NAMED shared element is caught today, incidentally: the object is
    stamped twice, so the duplicate-canonical check sees one object twice
    under its final name and errors. An ANONYMOUS one is stamped
    ``_canonical_name = None`` both times and slips through with nothing —
    no error, no warning, a silently wrong building.

    If this ever starts failing because the pre-existing checks caught it
    honestly, this whole module is redundant and should go.
    """
    proj = _two_covers(_box())          # anonymous — no name=
    report = validate_project_report(proj)

    assert find_shared_elements(proj), "the new check must fire"
    unrelated = [e for e in report.errors
                 if "canonical" in e.lower() and "attached in" not in e]
    assert unrelated == [], (
        "an anonymous shared element is expected to slip through the "
        f"pre-existing identity checks; got {unrelated}"
    )


def test_shared_element_is_an_ERROR_since_the_v10_cutover():
    """It shipped as a warning only because refusing it would have stopped the
    skills corpus compiling — the shim-first sequence. The corpus
    is clean now (zero models warn), so the refusal landed.

    Promoting it matters more than a tidy-up because the failure is UNDEFINED
    rather than wrong: ``_canonical_name`` / ``_frame`` / ``_anchor_spec`` are
    single-valued per OBJECT, so a bounding query on a shared element answers
    from whichever write happened to land last.
    """
    proj = _two_covers(_box())
    report = validate_project_report(proj)
    assert any("attached in 2 places" in e for e in report.errors), \
        f"expected a refusal, got errors={report.errors} warnings={report.warnings}"
    assert not any("attached in 2 places" in w for w in report.warnings), \
        "still reported as a warning too — one verdict, not two"


def test_derivation_by_call_is_silent():
    """The recommended fix must not itself trip the check.

    Two structurally identical elements from two constructor calls are two
    objects; only the same object twice is the defect. Identity is ``id()``
    for exactly this reason.
    """
    template = _box(name="shelf")
    proj = Project(name="t")
    proj.add(
        Element(ifc_class="IfcCovering", name="a").add(template(name="s0")),
        Element(ifc_class="IfcCovering", name="b").add(template(name="s1")),
    )
    assert find_shared_elements(proj) == []


def test_an_ordinary_tree_is_silent():
    proj = Project(name="t")
    wall = _wall("south")
    wall.anchor(Window(width=1210, height=1403, name="w0"), along=1507, up=903)
    site = Site(name="site").add(
        Element(ifc_class="IfcGeographicElement", predefined_type="TERRAIN",
                name="terrain").add(
                    Mesh(vertices=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0),
                                   Point(x=0, y=1000, z=0)],
                         faces=[[0, 1, 2]], name="ground")))
    proj.add(wall, site)
    assert find_shared_elements(proj) == []


def test_shared_boolean_operand_is_reported_too():
    """Operands are not ``.parent``-ed, but they ARE on the naming walk —
    which is where the double-stamp happens, so the check must cover them."""
    cutter = _box(name="cutter")
    proj = Project(name="t")
    proj.add(
        Element(ifc_class="IfcCovering", name="a").add(_box(name="h1").difference(cutter)),
        Element(ifc_class="IfcCovering", name="b").add(_box(name="h2").difference(cutter)),
    )
    assert any("cutter" in w for w in find_shared_elements(proj))


def test_equality_does_not_recurse_through_the_parent():
    """The bug that nearly shipped, and the reason ``__eq__`` is overridden.

    Pydantic v2's ``BaseModel.__eq__`` compares ``__pydantic_private__``
    wholesale. With ``_parent`` in there, equality walks UPWARD: two walls ->
    their storey -> its element list -> those same two walls, without end.

    Measured before the override: ``RecursionError`` after 663 frames, and a
    subset that runs 1208 tests in 18 s took over nine minutes and 13 GB of
    RAM. A plain timeout would not have caught it — the recursion is caught
    and retried often enough to look like slowness rather than a crash.
    """
    a, b = _wall("south"), _wall("north")
    proj = Project(name="t")
    proj.add(a, b)

    assert (a == b) is False                    # must not raise RecursionError
    assert (a == a) is True
    assert (proj.storeys[0] == proj.storeys[0]) is True


def test_equality_still_descends_into_children():
    """Excluding the upward pointer must not blind equality to real differences.

    ``_elements`` is still compared, so a container whose child changed is
    still unequal — the override removes exactly one edge, not the walk.
    """
    a = Wall(name="south").add(_box(name="body"))
    b = Wall(name="south").add(_box(name="body", dx=9999))
    b.id = a.id                                  # neutralise the auto-id
    assert a != b, "a differing child must still make containers unequal"


def test_find_shared_elements_works_on_a_never_stamped_project():
    """Self-sufficient, like ``resolve_placement_matrices``.

    The compile path stamps first and the check just reads the counts; a
    direct call (a test, a ``--q`` query) has to stamp for itself rather than
    silently answer "nothing shared" from an empty dict.
    """
    proj = _two_covers(_box(name="shelf"))
    assert proj._visit_counts is None
    assert find_shared_elements(proj), "must stamp for itself, not report clean"


def test_visit_counts_are_cleared_when_not_counting():
    """A stale count dict describes a tree that no longer exists.

    ``stamp_canonical_names`` runs at several entry points and only one asks
    for counts. If a later non-counting stamp left the previous dict in place, the
    check would answer from a snapshot taken before the tree changed — the
    same escaped-reference class as the ``_parent``-after-normalize bug.
    """
    proj = _two_covers(_box(name="shelf"))
    stamp_canonical_names(proj, count_visits=True)
    assert any(c > 1 for c in proj._visit_counts.values())

    stamp_canonical_names(proj)                    # the default: no counting
    assert proj._visit_counts is None


def test_a_site_subtree_is_walked():
    """Sites live in ``project.sites``, not in a storey — a check that only
    walked storeys would miss every terrain element."""
    shared = _box(name="shelf")
    proj = Project(name="t")
    proj.add(
        Site(name="site").add(
            Element(ifc_class="IfcEarthworksFill", name="a").add(shared),
            Element(ifc_class="IfcEarthworksFill", name="b").add(shared),
        )
    )
    assert find_shared_elements(proj), "the site subtree must be walked"
