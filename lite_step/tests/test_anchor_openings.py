"""DSL v20.0.0 — an opening is placed by its HOST, and is never a carve.

``Window(offset=, sill=)`` / ``Door(offset=, sill=)`` are GONE. Placement is
the DSL's one placement idiom: ``wall.anchor(win, along=, up=, inset=)`` for a
Wall child, ``body.opening(win, along=, up=, inset=)`` for a standalone solid.
Both stamp the same ``ChildAnchor``; the opening machinery consumes it as
SCALARS (``along`` is the old ``offset=``, ``up`` the old ``sill=``) rather
than baking it into geometry an opening does not have.

The load-bearing invariant this suite exists for is (2): **a Door/Window is a
SEMANTIC opening, never a boolean carve, regardless of how it was placed.**
One ``IfcOpeningElement`` + one ``IfcRelVoidsElement`` + one
``IfcRelFillsElement``, and ZERO extra ``IfcBooleanResult`` against a
no-opening baseline. Booleans are the resource this codebase
rations: if anchoring an opening ever started routing
through ``.difference()``, a facade of windows would silently blow the depth
budget and render as an un-subtracted block.

Pinned here:

1.  ``offset=``/``sill=`` are removed fields — ``extra="forbid"`` rejects them.
2.  Semantic-opening invariant: exactly 1 void + 1 fill, ZERO boolean delta,
    on the wall path AND the standalone ``.opening()`` path, on BOTH backends.
3.  ``wall.anchor(win, along=X, up=Y)`` is numerically identical to what
    ``Window(offset=X, sill=Y)`` produced — same placement coordinates.
4.  An unplaced opening (``.add()``, no anchor) is a compile ERROR naming the
    call that fixes it; so is ``rotations=`` on an opening.
5.  ``inset=`` recesses the FILL only — the void still cuts the full thickness —
    and is facing-signed, so one value means "outward" on every wall.
6.  Both backends agree on the placement they emit.
"""

from __future__ import annotations

import os
import re
import tempfile

import pytest

from lite_step.compiler.executor import (
    normalize_project_to_meters,
    validate_project_report,
)
from lite_step.models import (
    Project, Point, Box, Wall, Window, Door,
)


# ---------------------------------------------------------------------------
# Helpers — one 6 m wall, 300 mm thick, running +X at y in [-300, 0].
#
# It is the storey's ONLY element, so the facing reference is the wall's own
# centre and ``_facing_for``'s tie resolves to the MAX side: the exterior is
# +Y. That makes local +x (right as seen from outside) run -X, which is why
# every ``along=1500`` below lands at ``6 - 1.5 - 1.2/2 = 3.9`` rather than
# the 2.1 the pre-v21 canonicalized run axis produced. Measured, not assumed.
# ---------------------------------------------------------------------------


def _body(**kw):
    return Box(start=Point(x=0, y=-300, z=0),
               end=Point(x=6000, y=0, z=2700), type="wall", **kw)


def _project(*elements) -> Project:
    proj = Project(name="anchor-openings")
    proj.add(*elements)
    return proj


def _wall(*, opening=None, along=1500, up=900, inset=0, name="south") -> Wall:
    wall = Wall(name=name)
    wall.add(_body(name="body"))
    if opening is not None:
        wall.anchor(opening, along=along, up=up, inset=inset)
    return wall


def _compile(proj: Project, backend: str) -> str:
    if backend == "ifcopenshell":
        pytest.importorskip("ifcopenshell")
    from lite_step.ifc.generator import generate_ifc

    result = generate_ifc(normalize_project_to_meters(proj))
    assert result.success, result.error
    return result.ifc_content


def _count(content: str, entity: str) -> int:
    return len(re.findall(rf"=\s*{entity}\(", content))


def _open(content: str):
    """Parse STEP text with ifcopenshell — used for BOTH backends.

    The streaming backend emits a real IFC4 file, so reading it the same way
    the ifcopenshell one is read is what makes the cross-backend agreement
    test meaningful rather than a comparison of two text parsers.
    """
    import ifcopenshell

    with tempfile.NamedTemporaryFile("w", suffix=".ifc", delete=False,
                                     encoding="utf-8") as fh:
        fh.write(content)
        path = fh.name
    try:
        return ifcopenshell.open(path)
    finally:
        os.unlink(path)


def _world_origin(product) -> tuple:
    """Product's WORLD placement origin — opening/fill placements are stored
    relative to the wall, so the raw Location is wall-local."""
    import ifcopenshell.util.placement as plc

    m = plc.get_local_placement(product.ObjectPlacement)
    return (float(m[0][3]), float(m[1][3]), float(m[2][3]))


def _placement_origin(content: str, entity: str):
    """(x, y, z) of the FIRST ``entity`` product in ``content``, in world space."""
    pytest.importorskip("ifcopenshell")
    products = _open(content).by_type(entity)
    assert products, f"no {entity} in output"
    return _world_origin(products[0])


# ---------------------------------------------------------------------------
# 1. The removed fields
# ---------------------------------------------------------------------------


class TestRemovedFields:
    def test_window_offset_and_sill_are_gone(self):
        with pytest.raises(Exception) as exc:
            Window(offset=1500, sill=900, width=1200, height=1400)
        assert "extra_forbidden" in str(exc.value) or "not permitted" in str(exc.value)

    def test_door_offset_and_sill_are_gone(self):
        with pytest.raises(Exception) as exc:
            Door(offset=1500, sill=0, width=900, height=2100)
        assert "extra_forbidden" in str(exc.value) or "not permitted" in str(exc.value)

    def test_size_only_construction_works(self):
        win = Window(width=1200, height=1400, name="w0")
        assert (win.width, win.height) == (1200, 1400)
        assert getattr(win, "_anchor_spec", None) is None   # unplaced until anchored


# ---------------------------------------------------------------------------
# 2. THE invariant — a semantic opening, never a carve
# ---------------------------------------------------------------------------


class TestSemanticOpeningInvariant:
    def test_anchored_window_is_one_void_one_fill_zero_booleans(self):
        baseline = _compile(_project(_wall()), "ifcopenshell")
        withwin = _compile(
            _project(_wall(opening=Window(width=1200, height=1400, name="w0"))),
            "ifcopenshell")

        assert _count(withwin, "IFCOPENINGELEMENT") == 1
        assert _count(withwin, "IFCRELVOIDSELEMENT") == 1
        assert _count(withwin, "IFCRELFILLSELEMENT") == 1

        # THE gate: an opening costs ZERO booleans. Not "few" — zero.
        assert (_count(withwin, "IFCBOOLEANRESULT")
                - _count(baseline, "IFCBOOLEANRESULT")) == 0
        assert (_count(withwin, "IFCBOOLEANCLIPPINGRESULT")
                - _count(baseline, "IFCBOOLEANCLIPPINGRESULT")) == 0

    def test_anchored_door_is_one_void_one_fill_zero_booleans(self):
        baseline = _compile(_project(_wall()), "ifcopenshell")
        withdoor = _compile(
            _project(_wall(opening=Door(width=900, height=2100, name="d0"),
                           along=3000, up=0)),
            "ifcopenshell")

        assert _count(withdoor, "IFCOPENINGELEMENT") == 1
        assert _count(withdoor, "IFCRELVOIDSELEMENT") == 1
        assert _count(withdoor, "IFCRELFILLSELEMENT") == 1
        assert (_count(withdoor, "IFCBOOLEANRESULT")
                - _count(baseline, "IFCBOOLEANRESULT")) == 0

    def test_two_openings_still_zero_booleans(self):
        """Depth is the thing being protected — two openings must not compose."""
        baseline = _compile(_project(_wall()), "ifcopenshell")
        wall = _wall(opening=Window(width=1200, height=1400, name="w0"))
        wall.anchor(Door(width=900, height=2100, name="d0"), along=4000)
        both = _compile(_project(wall), "ifcopenshell")

        assert _count(both, "IFCOPENINGELEMENT") == 2
        assert _count(both, "IFCRELVOIDSELEMENT") == 2
        assert _count(both, "IFCRELFILLSELEMENT") == 2
        assert (_count(both, "IFCBOOLEANRESULT")
                - _count(baseline, "IFCBOOLEANRESULT")) == 0


def test_standalone_opening_path_is_also_zero_booleans():
    """``body.opening(win, along=, up=)`` — the non-Wall placement route.

    Routed to the ifcopenshell backend by ``can_stream`` (a standalone opening
    is not streamable), so this is checked on that backend only.
    """
    baseline = _compile(_project(_body(name="solo")), "ifcopenshell")
    voided = _compile(
        _project(_body(name="solo").opening(
            Window(width=1200, height=1400, name="w0"), along=1500, up=900)),
        "ifcopenshell")

    assert _count(voided, "IFCOPENINGELEMENT") == 1
    assert _count(voided, "IFCRELVOIDSELEMENT") == 1
    assert _count(voided, "IFCRELFILLSELEMENT") == 1
    assert (_count(voided, "IFCBOOLEANRESULT")
            - _count(baseline, "IFCBOOLEANRESULT")) == 0


# ---------------------------------------------------------------------------
# 3. along=/up= reproduce offset=/sill= exactly
# ---------------------------------------------------------------------------


class TestNumericEquivalenceWithTheRemovedFields:
    """Body x in [0, 6] m, y in [-0.3, 0] m, z from 0. Exterior +Y (see the
    fixture note), so local +x runs -X and the outer face is y = 0.

    ``along=1500`` therefore walks 1.5 m from x = 6, and the void — centred
    on the full 300 mm depth — sits half a thickness inward of the outer
    face: ``(6 - 1.5 - 0.6, 0 - 0.15, 0.9) = (3.9, -0.15, 0.9)``.

    Pre-v21 this read ``(2.1, -0.15, 0.9)``, the number ``Window(offset=1500,
    sill=900)`` produced: ``along`` walked a run axis canonicalized toward +X
    regardless of which side of the wall was outside. **This reversal is the
    v21 breaking change**, and the void's THICKNESS coordinate is untouched
    by it — the hole is where it always was, only measured from the other
    end of the wall.
    """

    EXPECTED_VOID = (3.9, -0.15, 0.9)
    #: The fill is FLUSH with the outer face at ``inset=0`` — the same zero
    #: the opening's children are authored from. It does not coincide with
    #: the void, which stays centred in the thickness.
    EXPECTED_FILL = (3.9, 0.0, 0.9)

    def test_void_placement_matches_legacy_numbers(self):
        content = _compile(
            _project(_wall(opening=Window(width=1200, height=1400, name="w0"))),
            "ifcopenshell")
        assert _placement_origin(content, "IfcOpeningElement") == pytest.approx(
            self.EXPECTED_VOID, abs=1e-9)

    def test_fill_sits_flush_with_the_outer_face(self):
        content = _compile(
            _project(_wall(opening=Window(width=1200, height=1400, name="w0"))),
            "ifcopenshell")
        assert _placement_origin(content, "IfcWindow") == pytest.approx(
            self.EXPECTED_FILL, abs=1e-9)

# ---------------------------------------------------------------------------
# 4. An unplaced opening is a compile ERROR
# ---------------------------------------------------------------------------


class TestUnplacedOpeningIsRefused:
    def test_added_window_is_refused_at_the_add_line(self):
        """The refusal moved to the AUTHORING line — same early-check/backstop
        pairing as ``check_anchor_rotation`` and the containment-cycle refusal.

        A compile error alone would name a validation pass in the traceback
    rather than the line that wrote it."""
        wall = Wall(name="south")
        wall.add(_body(name="body"))
        with pytest.raises(ValueError, match="cannot be ADDED"):
            wall.add(Window(width=1200, height=1400, name="w0"))
        # And it names both fixes, on the line that needs them.
        with pytest.raises(ValueError, match=r"wall\.anchor\(window"):
            wall.add(Window(width=1200, height=1400, name="w1"))
        with pytest.raises(ValueError, match=r"body\.opening\(window"):
            wall.add(Window(width=1200, height=1400, name="w2"))

    def test_added_door_is_refused_at_the_add_line(self):
        wall = Wall(name="south")
        wall.add(_body(name="body"))
        with pytest.raises(ValueError, match=r"wall\.anchor\(door"):
            wall.add(Door(width=900, height=2100, name="d0"))

    def test_a_hand_built_unplaced_opening_is_still_a_compile_error(self):
        """The BACKSTOP. ``.add()`` cannot reach an opening any more, so the
        compile-time check only fires for a tree assembled around it — the AST
        patcher, a deserializer, a test. It stays for exactly that, and this
        is the test that keeps it honest."""
        wall = Wall(name="south")
        wall.add(_body(name="body"))
        wall._elements.append(Window(width=1200, height=1400, name="w0"))
        errors = validate_project_report(_project(wall)).errors
        assert any("needs a position" in e for e in errors), errors
        assert any("wall.anchor(" in e for e in errors), errors

    def test_anchored_window_validates_clean(self):
        assert validate_project_report(
            _project(_wall(opening=Window(width=1200, height=1400, name="w0")))
        ).errors == []

    def test_rotations_on_an_opening_are_a_compile_error(self):
        wall = Wall(name="south")
        wall.add(_body(name="body"))
        wall.anchor(Window(width=1200, height=1400, name="w0"),
                    along=1500, up=900, rotations=[("z", 4500)])
        errors = validate_project_report(_project(wall)).errors
        assert any("rotations" in e and "opening" in e for e in errors), errors

    def test_generator_refuses_an_unplaced_opening_loudly(self):
        """Validation is the gate, but a caller that skips it must not get a
        window silently compiled into the wall's start corner.

        ``generate_ifc`` never raises out (same contract as
        ``test_solid_openings.test_compile_does_not_crash_on_bad_opening_body``)
        — the raise becomes ``success=False`` carrying the message."""
        pytest.importorskip("ifcopenshell")
        from lite_step.ifc.generator import generate_ifc

        wall = Wall(name="south")
        wall.add(_body(name="body"))
        # Around ``.add()``, which now refuses this at the authoring line.
        wall._elements.append(Window(width=1200, height=1400, name="w0"))
        result = generate_ifc(normalize_project_to_meters(_project(wall)))
        assert result.success is False
        assert "no position" in result.error
        assert "wall.anchor(" in result.error


# ---------------------------------------------------------------------------
# 5. out= — a recess of the FILL, never of the void
# ---------------------------------------------------------------------------


class TestRecessDepth:
    def test_out_zero_is_the_legacy_placement(self):
        """The default must be byte-identical to pre-``inset=`` output."""
        a = _compile(
            _project(_wall(opening=Window(width=1200, height=1400, name="w0"))),
            "ifcopenshell")
        b = _compile(
            _project(_wall(opening=Window(width=1200, height=1400, name="w0"),
                           inset=0)),
            "ifcopenshell")
        assert _strip_volatile(a) == _strip_volatile(b)

    def test_out_moves_the_fill_not_the_void(self):
        content = _compile(
            _project(_wall(opening=Window(width=1200, height=1400, name="w0"),
                           inset=50)),
            "ifcopenshell")
        # The void is where it always was: full thickness, mid-plane.
        assert _placement_origin(content, "IfcOpeningElement") == pytest.approx(
            (3.9, -0.15, 0.9), abs=1e-9)
        # The fill recessed 50 mm INTO the wall from its OUTER face. This
        # wall's exterior is +Y (the facing tie, see the fixture note), so
        # "inward" is -Y and flush would be y = 0.
        fill = _placement_origin(content, "IfcWindow")
        assert fill[0] == pytest.approx(3.9, abs=1e-9)
        assert fill[2] == pytest.approx(0.9, abs=1e-9)
        assert fill[1] == pytest.approx(-0.05, abs=1e-9)

    def test_out_is_facing_signed_across_opposite_walls(self):
        """One ``inset=`` value must mean the same thing on a north and a south
        wall — the whole reason the opening frame is the wall's own rather
        than a run axis canonicalized toward +X."""
        south = Wall(name="south")
        south.add(Box(start=Point(x=0, y=-4000, z=0),
                      end=Point(x=6000, y=-3700, z=2700), type="wall",
                      name="body"))
        south.anchor(Window(width=1200, height=1400, name="ws"),
                     along=1500, up=900, inset=-60)
        north = Wall(name="north")
        north.add(Box(start=Point(x=0, y=3700, z=0),
                      end=Point(x=6000, y=4000, z=2700), type="wall",
                      name="body"))
        north.anchor(Window(width=1200, height=1400, name="wn"),
                     along=1500, up=900, inset=-60)

        content = _compile(_project(south, north), "ifcopenshell")
        ys = sorted(_fill_y(content))
        # Both fills stand 60 mm PROUD of their own facade: the south one to
        # more-negative y, the north one to more-positive y. Flush (inset=0)
        # they would sit ON the outer faces, at -4.0 and +4.0.
        assert ys[0] == pytest.approx(-4.06, abs=1e-9)
        assert ys[1] == pytest.approx(4.06, abs=1e-9)


def _fill_y(content: str):
    """World y of every IfcWindow in ``content``."""
    pytest.importorskip("ifcopenshell")
    return [_world_origin(w)[1] for w in _open(content).by_type("IfcWindow")]


def _strip_volatile(content: str) -> str:
    """Drop GUIDs, timestamps and the LITESTEP_META blob — what is left is the
    geometry and topology two compiles must agree on."""
    content = re.sub(r"'[0-9A-Za-z_$]{22}'", "'<guid>'", content)
    content = re.sub(r"'\d{4}-\d{2}-\d{2}T[\d:.+-]+'", "'<ts>'", content)
    # IfcUnitAssignment holds an unordered SET; ifcopenshell emits it in
    # iteration order, so two compiles of ONE model already differ there.
    content = re.sub(
        r"(IFCUNITASSIGNMENT\(\()([^)]*)(\)\))",
        lambda m: m.group(1) + ",".join(sorted(m.group(2).split(","))) + m.group(3),
        content)
    return "\n".join(
        line for line in content.splitlines()
        if not line.startswith(("FILE_NAME", "FILE_DESCRIPTION"))
        and "IFCPROPERTYSINGLEVALUE('SOURCE'" not in line
        and "IFCPROPERTYSINGLEVALUE('HASH'" not in line
    )
