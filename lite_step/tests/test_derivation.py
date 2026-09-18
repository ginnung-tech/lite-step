"""Derivation-by-call (DSL v1.5 §DERIVATION, WS1 PR-D).

Any element or Point is callable: ``instance(field=value)`` returns a copy
with those fields replaced.

Pinned behavior:

1.  Copy-with-override on frozen primitives (Point/Point2D) and on
    elements/containers.
2.  The copy is deep — private containers (``_elements``, ``_cuts``,
    ``_adds``, ``_fills``, ``_openings``) and mutable field values never
    stay shared with the template; mutation of one never leaks into the
    other, in either direction.
3.  Unknown field → actionable ``TypeError`` (extra="forbid" posture).
4.  Deriving from a NAMED template without overriding ``name=`` is allowed
    at copy time; compile fails with the duplicate-name ERROR (RULE 8) once
    both land in one Project.
5.  ``id`` regenerates unless explicitly overridden — a copy is a new
    element.
6.  Under the strict int-mm flag, a float in a derivation call is rejected
    exactly like construction; carried-over (already validated) values are
    never re-gated.
"""

from __future__ import annotations

import pytest

from lite_step import disable_strict_int_mm, enable_strict_int_mm
from lite_step.compiler.executor import (
    execute_lite_step_script,
    validate_project,
)
from lite_step.models import (
    Project,
    Point,
    Point2D,
    Slab,
    Box,
    Extrude,
    Wall,
    Window,
)


@pytest.fixture(autouse=True)
def _reset_strict_flag():
    yield
    enable_strict_int_mm()  # post-cutover default


def _box(name=None, **kwargs) -> "BimElement":
    return Box(
        start=Point(x=-6000, y=-4500, z=0),
        end=Point(x=6000, y=-4200, z=2700),
        name=name,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# 1. Frozen primitives: Point / Point2D
# ---------------------------------------------------------------------------


class TestPointDerivation:
    def test_point_copy_with_override(self):
        p1 = Point(x=-6000, y=-4500, z=0)
        p2 = p1(y=4200)
        assert (p2.x, p2.y, p2.z) == (-6000, 4200, 0)
        assert p2 is not p1

    def test_template_point_is_unchanged(self):
        p1 = Point(x=-6000, y=-4500, z=0)
        p1(y=4200, z=3000)
        assert (p1.x, p1.y, p1.z) == (-6000, -4500, 0)

    def test_point_multi_field_override(self):
        p = Point(x=1, y=2, z=3)(x=10, z=30)
        assert (p.x, p.y, p.z) == (10, 2, 30)

    def test_point_zero_overrides_is_plain_copy(self):
        p1 = Point(x=1, y=2, z=3)
        p2 = p1()
        assert p2 == p1 and p2 is not p1

    def test_point2d_derivation(self):
        q = Point2D(x=45, y=195)(y=95)
        assert (q.x, q.y) == (45, 95)

    def test_point_unknown_field_is_actionable(self):
        p = Point(x=0, y=0, z=0)
        with pytest.raises(TypeError) as exc_info:
            p(w=5)
        msg = str(exc_info.value)
        assert "'w'" in msg
        assert "x" in msg and "y" in msg and "z" in msg  # valid fields listed

    def test_derived_point_composes_with_arithmetic(self):
        # spec example: s1.start(y=...) on a Point that also does + / -
        base = Point(x=100, y=200, z=0) + Point(x=1, y=1, z=1)
        derived = base(z=500)
        assert (derived.x, derived.y, derived.z) == (101, 201, 500)


class TestPointDerivationStrict:
    def test_strict_rejects_float_override_like_construction(self):
        enable_strict_int_mm()
        p = Point(x=-6000, y=-4500, z=0)
        with pytest.raises(Exception) as exc_info:
            p(y=4200.5)
        msg = str(exc_info.value)
        assert "y=4200.5" in msg
        assert "I(" in msg  # the actionable fix, same as construction

    def test_strict_rejects_integral_float_override_too(self):
        enable_strict_int_mm()
        with pytest.raises(Exception):
            Point(x=0, y=0, z=0)(y=4200.0)

    def test_strict_accepts_int_override(self):
        enable_strict_int_mm()
        p = Point(x=-6000, y=-4500, z=0)(y=4200)
        assert p.y == 4200

    def test_strict_does_not_regate_carried_over_values(self):
        # Points produced by pipeline arithmetic store floats; deriving with
        # an int override must not re-validate (and wrongly reject) them.
        enable_strict_int_mm()
        summed = Point(x=100, y=200, z=0) + Point(x=1, y=1, z=1)
        derived = summed(z=500)
        assert (derived.x, derived.y) == (101, 201)

    def test_strict_point2d_derivation_gated(self):
        enable_strict_int_mm()
        with pytest.raises(Exception):
            Point2D(x=45, y=195)(y=97.5)


# ---------------------------------------------------------------------------
# 2. Elements: copy-with-override, fresh id, validated copy
# ---------------------------------------------------------------------------


class TestElementDerivation:
    def test_solid_copy_with_override_spec_example(self):
        s1 = Box(name="wall_s_body",
                   start=Point(x=-6000, y=-4500, z=0),
                   end=Point(x=6000, y=-4200, z=2700))
        s2 = s1(name="wall_n_body", start=s1.start(y=4200), end=s1.end(y=4500))
        assert s2.name == "wall_n_body"
        assert (s2.start.y, s2.end.y) == (4200, 4500)
        # non-overridden geometry carried over
        assert (s2.start.x, s2.end.x) == (s1.start.x, s1.end.x)
        # template untouched
        assert (s1.start.y, s1.end.y) == (-4500, -4200)

    def test_derived_copy_gets_fresh_id(self):
        s1 = _box()
        assert s1() .id != s1.id

    def test_explicit_id_override_is_kept(self):
        s1 = _box()
        assert s1(id="explicit_id").id == "explicit_id"

    def test_name_carried_over_by_default(self):
        s1 = _box(name="tpl")
        assert s1().name == "tpl"

    def test_name_can_be_cleared_to_anonymous(self):
        s1 = _box(name="tpl")
        assert s1(name=None).name is None

    def test_unknown_field_is_actionable(self):
        s1 = _box()
        with pytest.raises(TypeError) as exc_info:
            s1(thickness_mm=22)
        msg = str(exc_info.value)
        assert "'thickness_mm'" in msg
        assert "Box" in msg
        assert "start" in msg  # valid fields listed

    def test_derived_copy_is_validated_not_blind(self):
        # DSL v8: Box and Extrude are separate classes. A Box has no contour
        # field, so deriving one with contour= trips validation (unknown
        # field) exactly like construction would — proving the derived copy
        # is validated, not blindly merged.
        s1 = _box()
        with pytest.raises(Exception, match="unknown field"):
            s1(contour=[Point(x=0, y=0, z=0), Point(x=1000, y=0, z=0),
                        Point(x=1000, y=1000, z=0)])

    def test_slab_is_container_of_box_extrude(self):
        # DSL v8: Slab is a container (no geometry-direct start/end/contour);
        # it aggregates Box/Extrude children via .add().
        slab = Slab(name="foundation")
        slab.add(Box(start=Point(x=-5500, y=0, z=-300),
                     end=Point(x=5500, y=300, z=0)))
        assert len(slab.elements) == 1
        with pytest.raises(Exception):
            slab.add(Sweep(start=Point(x=0, y=0, z=0),
                           end=Point(x=1, y=0, z=0)))  # off-table child rejected

    def test_props_deep_copied(self):
        s1 = _box(name="a", props={"Pset_X": {"U": 0.9}})
        s2 = s1(name="b")
        s1.props["Pset_X"]["U"] = 1.4
        s1.props["Pset_Y"] = {"New": 1}
        assert s2.props == {"Pset_X": {"U": 0.9}}
        s2.props["Pset_X"]["FromCopy"] = True
        assert "FromCopy" not in s1.props["Pset_X"]


# ---------------------------------------------------------------------------
# 3. Deep-copy isolation of private containers
# ---------------------------------------------------------------------------


class TestDeepCopyIsolation:
    def test_container_children_isolated_template_to_copy(self):
        w1 = Wall(name="wall_a")
        w1.add(_box())
        w2 = w1(name="wall_b")
        # mutate template AFTER deriving — copy unaffected
        w1.anchor(Window(width=1200, height=1400), along=1500, up=900)
        assert len(w1._elements) == 2
        assert len(w2._elements) == 1

    def test_container_children_isolated_copy_to_template(self):
        w1 = Wall(name="wall_a")
        w1.add(_box())
        w2 = w1(name="wall_b")
        w2.anchor(Window(width=1200, height=1400), along=1500, up=900)
        assert len(w1._elements) == 1
        assert len(w2._elements) == 2

    def test_child_objects_are_copies_not_shared(self):
        w1 = Wall(name="wall_a")
        w1.add(_box(color="wall"))
        w2 = w1(name="wall_b")
        assert w2._elements[0] is not w1._elements[0]
        # mutating a child of the copy never leaks into the template
        w2._elements[0].color = "glass"
        assert w1._elements[0].color == "wall"
        # ... and vice versa
        w1._elements[0].color = "roof"
        assert w2._elements[0].color == "glass"

    def test_boolean_cut_lists_isolated(self):
        void = Box(start=Point(x=0, y=0, z=0), end=Point(x=900, y=400, z=2100))
        s1 = _box(name="a").difference(void)
        s2 = s1(name="b")
        s1.difference(Box(start=Point(x=0, y=0, z=0), end=Point(x=50, y=50, z=50)))
        assert len(s1._cuts) == 2
        assert len(s2._cuts) == 1
        assert s2._cuts[0] is not void  # deep copy, not a shared reference

    def test_boolean_adds_and_openings_isolated(self):
        s1 = _box(name="a").union(
            Box(start=Point(x=0, y=0, z=0), end=Point(x=10, y=10, z=10)))
        s1.opening(Window(width=1200, height=1400), along=500, up=900)
        s2 = s1(name="b")
        s1.union(Box(start=Point(x=0, y=0, z=0), end=Point(x=5, y=5, z=5)))
        s1.opening(Window(width=600, height=600), along=2500, up=900)
        assert (len(s1._adds), len(s2._adds)) == (2, 1)
        assert (len(s1._openings), len(s2._openings)) == (2, 1)


# ---------------------------------------------------------------------------
# 4. Derived-without-rename → duplicate-name compile ERROR (end-to-end)
# ---------------------------------------------------------------------------


class TestDerivedNameEnforcement:
    def test_copy_time_is_allowed(self):
        s1 = _box(name="tpl")
        s2 = s1()  # no error at copy time — enforcement is at compile
        assert s2.name == "tpl"

    def test_duplicate_name_compile_error_with_derivation_hint(self):
        b = Project(name="t")
        s1 = _box(name="tpl")
        b.add(s1)
        b.add(s1(start=s1.start(y=4200), end=s1.end(y=4500)))
        errors = [e for e in validate_project(b)
                  if "Duplicate canonical name" in e]
        assert len(errors) == 1
        assert "tpl" in errors[0]
        assert "verride name=" in errors[0]  # the derivation hint

    def test_renamed_derivation_compiles_clean(self):
        b = Project(name="t")
        s1 = _box(name="wall_s_body")
        b.add(s1)
        b.add(s1(name="wall_n_body", start=s1.start(y=4200),
                 end=s1.end(y=4500)))
        assert [e for e in validate_project(b)
                if "Duplicate canonical name" in e] == []

    def test_end_to_end_script_duplicate_name_fails_compile(self):
        """Full sandbox path: derivation works in the restricted namespace
        and the un-renamed copy is rejected by compile validation."""
        script = """
s1 = Box(name="wall_s_body",
           start=Point(x=-6000, y=-4500, z=0),
           end=Point(x=6000, y=-4200, z=2700))
s2 = s1(start=s1.start(y=4200), end=s1.end(y=4500))  # name NOT overridden

proj = Project(name="dup")
proj.add(s1)
proj.add(s2)
result = proj
"""
        exec_result = execute_lite_step_script(script)
        assert exec_result.success, exec_result.error
        errors = validate_project(exec_result.project)
        assert any("Duplicate canonical name" in e and "box:wall_s_body" in e
                   for e in errors)

    def test_end_to_end_script_renamed_derivation_is_clean(self):
        script = """
s1 = Box(name="wall_s_body",
           start=Point(x=-6000, y=-4500, z=0),
           end=Point(x=6000, y=-4200, z=2700))
s2 = s1(name="wall_n_body", start=s1.start(y=4200), end=s1.end(y=4500))

proj = Project(name="ok")
proj.add(s1)
proj.add(s2)
result = proj
"""
        exec_result = execute_lite_step_script(script)
        assert exec_result.success, exec_result.error
        assert [e for e in validate_project(exec_result.project)
                if "Duplicate canonical name" in e] == []

    def test_loop_template_derivation_pattern(self):
        """RULE 11 pattern: template + comprehension with per-copy rename."""
        b = Project(name="rafters")
        r0 = _box(name="rafter_0")
        b.add(r0)
        for i in range(1, 4):
            b.add(r0(name=f"rafter_{i}"))
        assert [e for e in validate_project(b)
                if "Duplicate canonical name" in e] == []
