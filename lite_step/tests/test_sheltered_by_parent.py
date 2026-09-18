"""A child cannot reach what its parent cannot — even when its own shape is
undecidable.

The narrow phase (#19) answers exactly for right-angled volumes and returns
"keep the carve" for everything else. That safe default produced a real
asymmetry:

* a wall's body is a ``Box`` → decidable → its carve against a distant rafter
  is correctly dropped;
* a rebar ``Bar`` is a swept disk → ``convex_solid_obb`` returns ``None`` →
  undecidable → its carve against the SAME rafter is kept.

So the wall stopped carving something it does not touch while 23 bars *inside
that wall* carried on carving it. Measured across the corpus: 300 such
booleans in five systems, every one removing nothing, and 0 of 2,635 products
changed volume when they were dropped.

Containment is transitive through disjointness: if the child lies wholly
inside a solid that is provably disjoint from the host, the child is provably
disjoint too — whatever shape the child is. That is the whole argument, and it
is why this is exact rather than a heuristic.

**Only case (1) is eligible** — child WHOLLY INSIDE parent. A child that pokes
out, or sits outside entirely, may legitimately touch what its parent does
not: the miter leaves room for children so they miter individually, so
reaching past the parent's own solid is designed behaviour, not an escape.
Those keep their carves, which is what stops this from suppressing a real
joint.
"""

from __future__ import annotations

from lite_step.compiler.displacement import _sheltered_by_parent
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.compiler.extent import convex_solid_obb
from lite_step.compiler.naming import stamp_canonical_names
from lite_step.models import Bar, Box, Material, Point, Project, Sweep, Wall
from lite_step.models.project import Storey

# The PREDICATE is tested directly, not through `apply_displacement`.
#
# Three drafts tried to drive it through the whole pass and all three were
# dead: the padded broad phase would not pair a synthetic rafter with a
# synthetic bar at any height that also left their solids disjoint, so every
# assertion held with the rule disabled. The corpus reproduces that pairing —
# 300 dropped booleans over five systems, 0 of 2,635 products changed volume —
# and that measurement is the integration evidence. These tests pin the RULE.


def _prepared(*elements):
    """Metres, then names — the two things a real compile does first.

    `normalize_project_to_meters` RETURNS a copy, and `_parent` is stamped by
    the naming walk. Skip either and the rule has nothing to stand on.
    """
    proj = Project(name="shelter")
    storey = Storey(elevation=0)
    proj.add_storey(storey)
    storey.add(*elements)
    proj = normalize_project_to_meters(proj)
    stamp_canonical_names(proj)
    return proj


def _find(proj, kind, name):
    out = []

    def walk(e):
        if type(e).__name__ == kind and str(getattr(e, "name", "")) == name:
            out.append(e)
        for c in (getattr(e, "_elements", None) or []):
            walk(c)

    for st in proj.storeys:
        for e in st.elements:
            walk(e)
    return out[0]


def _member(name, z, x0=-500, x1=500):
    return Sweep(name=name, path=[Point(x=x0, y=0, z=z), Point(x=x1, y=0, z=z)],
                 material=Material(key="Timber_C24", profile_mm=(45, 195)))


def _wall(bar_top=2990, body_top=3000):
    wall = Wall(name="w")
    wall.add(Box(name="body", start=Point(x=-3000, y=-100, z=0),
                 end=Point(x=3000, y=100, z=body_top), type="wall"))
    wall.add(Bar(name="r0", diameter=12, grade="B500B",
                 path=[Point(x=0, y=0, z=50), Point(x=0, y=0, z=bar_top)]))
    return wall


def test_a_bar_is_undecidable_and_a_box_is_not():
    """The asymmetry the fix exists for. If this flips, the fix is moot."""
    bar = Bar(name="r", diameter=12, grade="B500B",
              path=[Point(x=0, y=0, z=0), Point(x=0, y=0, z=1000)])
    box = Box(name="b", start=Point(x=-100, y=-100, z=-100),
              end=Point(x=100, y=100, z=100))
    assert convex_solid_obb(bar, 1.0) is None
    assert convex_solid_obb(box, 1.0) is not None


def test_case_1_child_wholly_inside_a_parent_that_cannot_reach():
    """The rule fires: the bar is inside the body, the body is provably
    disjoint from the member, so the bar is too — whatever shape it is."""
    proj = _prepared(_member("rafter", 6000), _wall())
    bar = _find(proj, "Bar", "r0")
    far = _find(proj, "Sweep", "rafter")
    assert _sheltered_by_parent(far, bar, {}) is True


def test_case_2_child_pokes_OUT_of_its_parent():
    """Never sheltered. The miter leaves room for children so they miter
    individually, so reaching past the parent is designed behaviour."""
    wall = Wall(name="w")
    wall.add(Box(name="body", start=Point(x=-3000, y=-100, z=0),
                 end=Point(x=3000, y=100, z=3000), type="wall"))
    wall.add(Bar(name="r0", diameter=12, grade="B500B",
                 path=[Point(x=0, y=0, z=50), Point(x=0, y=0, z=6100)]))
    proj = _prepared(_member("rafter", 6000), wall)
    assert _sheltered_by_parent(_find(proj, "Sweep", "rafter"),
                                _find(proj, "Bar", "r0"), {}) is False


def test_the_parent_that_DOES_reach_shelters_nothing():
    """The counter-test. Member moved onto the wall: the parent reaches, so
    the argument does not apply and the bar keeps the safe default."""
    proj = _prepared(_member("rafter", 1500), _wall())
    assert _sheltered_by_parent(_find(proj, "Sweep", "rafter"),
                                _find(proj, "Bar", "r0"), {}) is False


def test_an_orphan_is_never_sheltered():
    lone = Bar(name="r0", diameter=12, grade="B500B",
               path=[Point(x=0, y=0, z=400), Point(x=0, y=0, z=600)])
    proj = _prepared(_member("rafter", 6000), lone)
    assert _sheltered_by_parent(_find(proj, "Sweep", "rafter"),
                                _find(proj, "Bar", "r0"), {}) is False


def test_a_container_is_resolved_to_the_solids_it_owns():
    """A `Wall` is not a solid — `convex_solid_obb` returns None for it, and a
    first version asked the container directly and could therefore never fire.
    The body child is what answers."""
    proj = _prepared(_wall())
    wall = _find(proj, "Wall", "w")
    assert convex_solid_obb(wall, 1.0) is None
    assert convex_solid_obb(_find(proj, "Box", "body"), 1.0) is not None
