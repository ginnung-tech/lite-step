"""DSL v2.1 hierarchical-naming contract pins.

``name=`` is an optional lowercase LEAF; the compiler derives the canonical
name from containment as ``type:leaf`` pairs, **leaf-first**. This file pins
the observable v2.1 contract end-to-end (validate -> normalize -> generate),
independent of the migrated legacy suites:

1.  ``Window()`` is identical to ``Window(name=None)`` (anonymous) — same IFC.
2.  An anonymous element is absent from the manifest and emits ``Name=None``.
3.  A same-scope, same-type leaf collision errors with BOTH locations.
4.  Cross-scope derivation without a rename compiles.
5.  Same-scope derivation without a rename fails.
6.  ``Anchor(host="wall:north")`` resolves; a miss lists candidates; an
    ambiguous suffix errors listing the collisions.
7.  Moving an element between containers churns its GUID (identity change).
8.  Canonical order is pinned LEAF-FIRST (a deliberate, test-breaking choice).
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project,
    validate_project_report,
)
from lite_step.compiler.naming import stamp_canonical_names
from lite_step.ifc.embedder import extract_litestep_meta
from lite_step.ifc.generator import generate_ifc
from lite_step.models import Anchor, Box, Project, Point, Wall, Window
from lite_step.models.project import Storey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box(name=None, z=0, **kw) -> Box:
    return Box(start=Point(x=0, y=0, z=z), end=Point(x=5000, y=3000, z=z + 200),
               name=name, **kw)


def _building(*elements) -> Project:
    # Single storey is ANONYMOUS (DSL v2.1) -> contributes no path segment, so
    # top-level canonicals are segment-free (``wall:north``, not
    # ``wall:north:storey:...``). This mirrors Project.add's auto-storey.
    b = Project(name="t")
    s = Storey(elevation=0)
    for e in elements:
        s.add(e)
    b.add_storey(s)
    return b


def _compile(proj: Project, source="x=1"):
    normalized = normalize_project_to_meters(proj)
    result = generate_ifc(normalized, source_code=source)
    assert result.success, result.error
    return result.ifc_content


def _derive(box: Box, z: int) -> Box:
    """Derive a copy shifted in Z via real field overrides (not the `z=` test
    helper kwarg, which is not a model field)."""
    return box(start=box.start(z=z), end=box.end(z=z + 200))


def _canonical(elem) -> str:
    return elem._canonical_name


# ---------------------------------------------------------------------------
# 1. Window() == Window(name=None)
# ---------------------------------------------------------------------------


def test_default_name_is_none_identical_ifc():
    """Window() and Window(name=None) are the same anonymous element."""
    wall_a = Wall(name="north")
    wall_a.add(_box())
    wall_a.anchor(Window(width=1200, height=1400), along=1500, up=900)
    wall_b = Wall(name="north")
    wall_b.add(_box())
    wall_b.anchor(Window(width=1200, height=1400, name=None),
                  along=1500, up=900)
    # The proposition itself, stated directly — so this test still asserts
    # something if the text comparison below is ever weakened.
    assert Window(width=1200, height=1400).name is None
    assert Window(width=1200, height=1400, name=None).name is None

    # No source embedding (its meta is unrelated). Two things in emitted STEP
    # vary by DESIGN on every compile and must be normalised, and ONLY these
    # two: the random GlobalIds, and the FILE_NAME header timestamp (these two
    # compiles run back to back, so they differ whenever they straddle a
    # one-second tick — a real, independent flake, measured at 1 in 15).
    #
    # Deliberately NOT normalised: set-valued attribute order. That was a third
    # source of drift until IfcUnitAssignment.Units was pinned in
    # generator._canonical_unit_order, and sorting it here instead would make
    # this test blind to a re-introduced Python set. This comparison is now the
    # thing that catches one.
    import re
    def strip(s):
        s = re.sub(r"'[0-9A-Za-z_$]{22}'", "'GUID'", s)
        return re.sub(r"(FILE_NAME\('',)'[^']*'", r"\1'<ts>'", s)
    a = strip(_compile(_building(wall_a), source=None))
    b = strip(_compile(_building(wall_b), source=None))
    assert a == b


# ---------------------------------------------------------------------------
# 2. Anonymous: absent from manifest + Name=None
# ---------------------------------------------------------------------------


def test_anonymous_absent_from_manifest_and_name_none():
    proj = _building(_box(name="corner"), _box(z=2000))  # 2nd is anonymous
    content = _compile(proj)
    meta = extract_litestep_meta(content)
    assert "box:corner" in meta.manifest
    # No anonymous entry, and no id leaked as a key. The single storey is
    # anonymous (DSL v2.1) so it contributes no manifest key either.
    assert all(":" in k or k in ("t", "Site") for k in meta.manifest)
    # The anonymous proxy carries Name=$ (None).
    proxies = [l for l in content.splitlines() if "BUILDINGELEMENTPROXY" in l]
    anon = [l for l in proxies if "'box:corner'" not in l]
    assert anon and ",$,$," in anon[0].replace("'", "")


# ---------------------------------------------------------------------------
# 3. Same-scope same-type collision — BOTH locations
# ---------------------------------------------------------------------------


def test_same_scope_same_type_collision_reports_both():
    a = _box(name="north", id="a_id")
    b = _box(name="north", z=1000, id="b_id")
    errs = [e for e in validate_project(_building(a, b))
            if "Duplicate canonical name" in e]
    assert len(errs) == 1
    assert "box:north" in errs[0]
    # BOTH construction sites are identified (internal ids disambiguate).
    assert "a_id" in errs[0] and "b_id" in errs[0]


def test_different_types_same_leaf_coexist():
    wall = Wall(name="north")
    wall.add(_box())
    proj = _building(wall, _box(name="north", z=3000))  # wall:north vs box:north
    assert [e for e in validate_project(proj)
            if "Duplicate canonical name" in e] == []


# ---------------------------------------------------------------------------
# 4 + 5. Derivation across / within scope
# ---------------------------------------------------------------------------


def test_cross_scope_derivation_without_rename_compiles():
    tpl = _box(name="body")
    south = Wall(name="south")
    south.add(tpl)
    north = Wall(name="north")
    north.add(_derive(tpl, 1000))  # same leaf "body", diff container -> distinct
    assert [e for e in validate_project(_building(south, north))
            if "Duplicate canonical name" in e] == []


def test_same_scope_derivation_without_rename_fails():
    tpl = _box(name="body")
    wall = Wall(name="south")
    wall.add(tpl, _derive(tpl, 1000))  # both box:body:wall:south
    errs = [e for e in validate_project(_building(wall))
            if "Duplicate canonical name" in e]
    assert len(errs) == 1
    assert "box:body:wall:south" in errs[0]


# ---------------------------------------------------------------------------
# 6. Anchor host resolution — resolve / miss / ambiguous
# ---------------------------------------------------------------------------


def test_anchor_host_resolves_unique_suffix():
    wall = Wall(name="north")
    wall.add(_box())  # anonymous body -> host "wall:north" is unique
    shelf = _box(name="shelf", z=5000, placement=Anchor(host="wall:north"))
    assert [e for e in validate_project_report(_building(wall, shelf)).errors
            if "Anchor" in e] == []


def test_anchor_host_miss_lists_known_names():
    wall = Wall(name="north")
    wall.add(_box())
    shelf = _box(name="shelf", z=5000, placement=Anchor(host="wall:missing"))
    errs = [e for e in validate_project_report(_building(wall, shelf)).errors
            if "Anchor" in e]
    assert len(errs) == 1
    assert "matches no element" in errs[0]
    assert "wall:north" in errs[0]  # known names listed


def test_anchor_host_exact_match_wins_over_named_child():
    # An EXACT canonical name resolves to that element even when a named
    # child carries it as a trailing pair-suffix. Without exact-match
    # precedence, naming a child (openings are commonly named) would make
    # the container unreferenceable — host="wall:north" would ambiguate
    # with "window:left:wall:north". The author who writes the wall's own
    # full name means the wall.
    wall = Wall(name="north")
    wall.add(_box(name="body"))  # named child sharing the wall:north suffix
    shelf = _box(name="shelf", z=5000, placement=Anchor(host="wall:north"))
    errs = [e for e in validate_project_report(_building(wall, shelf)).errors
            if "Anchor" in e]
    assert errs == []  # resolves cleanly to the wall, no ambiguity error


def test_resolve_host_ambiguity_branch():
    # The defensive ambiguity branch: a pair-suffix fragment that is NOT
    # itself a canonical name but is a trailing pair-suffix of 2+ names.
    from lite_step.compiler.naming import resolve_host
    status, matched, cands = resolve_host(
        "wall:north", ["window:left:wall:north", "door:right:wall:north"]
    )
    assert status == "ambiguous" and matched is None
    assert cands == ["door:right:wall:north", "window:left:wall:north"]


# ---------------------------------------------------------------------------
# 7. Move-between-containers churns the GUID (identity change)
# ---------------------------------------------------------------------------


def test_move_between_containers_changes_identity():
    """Same leaf, different container -> different canonical -> different GUID.

    The GUID is derived from the canonical name (identity), so relocating an
    element is a delete + create at the identity level (patch-mode semantics)."""
    def guid_of(canonical, content):
        for line in content.splitlines():
            if f"'{canonical}'" in line and line.strip().startswith("#"):
                # GUID is the first quoted attr on a rooted entity line.
                import re
                m = re.search(r"\('([0-9A-Za-z_$]{22})'", line)
                if m:
                    return m.group(1)
        return None

    south = Wall(name="south")
    south.add(_box())
    south.anchor(Window(width=1200, height=1400, name="w"), along=1500, up=900)
    c_south = _compile(_building(south))

    north = Wall(name="north")
    north.add(_box())
    north.anchor(Window(width=1200, height=1400, name="w"), along=1500, up=900)
    c_north = _compile(_building(north))

    g_south = guid_of("window:w:wall:south", c_south)
    g_north = guid_of("window:w:wall:north", c_north)
    assert g_south is not None and g_north is not None
    assert g_south != g_north


# ---------------------------------------------------------------------------
# 8. Canonical order is LEAF-FIRST (deliberate, pinned)
# ---------------------------------------------------------------------------


def test_canonical_order_is_leaf_first():
    wall = Wall(name="north")
    body = _box(name="body")
    win = Window(width=1200, height=1400, name="left")
    wall.add(body)
    wall.anchor(win, along=1500, up=900)
    proj = _building(wall)
    stamp_canonical_names(proj)
    # The element's OWN pair comes FIRST, ancestors follow outward.
    assert _canonical(wall) == "wall:north"
    assert _canonical(body) == "box:body:wall:north"
    assert _canonical(win) == "window:left:wall:north"
    # Explicitly NOT the root-first ordering.
    assert _canonical(win) != "wall:north:window:left"


# ---------------------------------------------------------------------------
# Leaf-name grammar (construction gate)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["North", "wall:north", "wall.north", "a b", "A"])
def test_invalid_leaf_names_rejected_at_construction(bad):
    with pytest.raises(Exception) as exc:
        _box(name=bad)
    assert "leaf" in str(exc.value).lower()


def test_valid_leaf_names_accepted():
    for good in ["north", "wall_1", "a", "box_2b"]:
        _box(name=good)  # no raise


# ---------------------------------------------------------------------------
# Storey obeys the universal segment rule (DSL v2.1)
# ---------------------------------------------------------------------------


def _multi_storey(*storeys) -> Project:
    b = Project(name="t")
    for s in storeys:
        b.add_storey(s)
    return b


def test_named_storey_emits_path_segment():
    """A NAMED storey contributes a trailing ``storey:<leaf>`` pair; the
    storey's own canonical is ``storey:<leaf>`` (its IfcBuildingStorey Name)."""
    ground = Storey(name="ground", elevation=0)
    wall = Wall(name="north")
    wall.add(_box())
    ground.add(wall)
    proj = _multi_storey(ground, Storey(name="first", elevation=3000))
    stamp_canonical_names(proj)
    assert _canonical(ground) == "storey:ground"
    # Element canonical is leaf-first, storey pair is the outermost tail.
    assert _canonical(wall) == "wall:north:storey:ground"


def test_two_floors_same_wall_leaf_coexist():
    """``wall:north`` on two NAMED floors are distinct canonicals — the storey
    segment scopes them — so they compile without a uniqueness collision."""
    ground = Storey(name="ground", elevation=0)
    g_wall = Wall(name="north")
    g_wall.add(_box())
    ground.add(g_wall)
    first = Storey(name="first", elevation=3000)
    f_wall = Wall(name="north")
    f_wall.add(_box())
    first.add(f_wall)
    proj = _multi_storey(ground, first)
    assert [e for e in validate_project(proj)
            if "Duplicate canonical name" in e] == []
    stamp_canonical_names(proj)
    assert _canonical(g_wall) == "wall:north:storey:ground"
    assert _canonical(f_wall) == "wall:north:storey:first"


def test_anonymous_single_storey_is_segment_free():
    """The single-storey common case (anonymous auto-storey) contributes no
    segment — canonicals stay exactly as the pre-storey pins assert."""
    wall = Wall(name="north")
    wall.add(_box(name="body"))
    proj = Project(name="t")
    proj.add(wall)  # auto-creates the anonymous storey
    stamp_canonical_names(proj)
    assert proj.storeys[0].name is None
    assert proj.storeys[0]._canonical_name is None
    assert _canonical(wall) == "wall:north"  # NO storey tail


def test_unnamed_multi_storey_errors():
    """More than one storey with any anonymous storey is a compile ERROR."""
    ground = Storey(name="ground", elevation=0)
    ground.add(_box(name="a"))
    anon = Storey(elevation=3000)  # anonymous — illegal in a multi-storey proj
    anon.add(_box(name="b"))
    errs = [e for e in validate_project(_multi_storey(ground, anon))
            if "anonymous storey" in e]
    assert len(errs) == 1
    assert "must be named" in errs[0]


# ---------------------------------------------------------------------------
# .add() varargs — the documented "all .add() accept varargs" contract
# ---------------------------------------------------------------------------


def test_building_and_storey_add_accept_varargs():
    """``Project.add`` and ``Storey.add`` take varargs like every semantic
    container, so the canonical-example idiom ``proj.add(south, north)``
    compiles. Taking a single element only would make the documented
    multi-element form raise ``TypeError``."""
    # Project.add(*elements) → both land in the (anonymous) first storey.
    proj = Project(name="t")
    a, b = _box(name="a"), _box(name="b", z=1000)
    proj.add(a, b)
    assert proj.storeys[0].elements == [a, b]
    # Storey.add(*elements) and the *walls splat form.
    s = Storey(elevation=0)
    walls = [Wall(name="north"), Wall(name="south")]
    s.add(*walls)
    assert s.elements == walls
    # Single-arg still works (backward compatible).
    s2 = Storey(elevation=0)
    s2.add(_box(name="solo"))
    assert len(s2.elements) == 1
