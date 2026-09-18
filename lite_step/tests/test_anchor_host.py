"""``Anchor.host`` — name-based host reference (DSL v1.5, WS1 PR-D).

Model/API surface pins from PR-D. The placement engine that CONSUMES
Anchor landed in WS1 PR-F — geometry-level coverage lives in
``test_placement_engine.py``; this file keeps the reference-resolution
contract pinned. Pinned behavior:

1.  ``host=`` refers to a NAMED element (``name=``) and is the only host
    field (the pre-v1.5 ``host_id=`` alias was removed at the v1.5 hard
    cutover — passing it is now rejected by ``extra="forbid"``).
2.  Neither set → validation error at construction.
3.  ``validate_project_report``: an Anchor whose ``host`` names a
    non-existent element is a compile ERROR (caught at compile, not at
    render); anchoring to an anonymous element is an ERROR with a rename
    hint.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import validate_project_report
from lite_step.models import Anchor, Project, Point, Box, Extrude


def _box(name=None, offset=0, **kwargs) -> "BimElement":
    return Box(
        start=Point(x=offset, y=0, z=0),
        end=Point(x=offset + 5000, y=300, z=2700),
        name=name,
        **kwargs,
    )


def _building(*elements) -> Project:
    b = Project(name="anchor-test")
    for e in elements:
        b.add(e)
    return b


# ---------------------------------------------------------------------------
# 1. Model surface
# ---------------------------------------------------------------------------


class TestAnchorModel:
    def test_host_accepted(self):
        a = Anchor(host="wall_south", attach_to="face", offset_along=2000)
        assert a.host == "wall_south"
        assert a.host_ref == "wall_south"

    def test_host_id_is_rejected(self):
        # DSL v1.5 hard cutover (identity hardening): the pre-v1.5 host_id=
        # alias was removed; it is now rejected by extra="forbid".
        with pytest.raises(Exception) as exc_info:
            Anchor(host_id="wall_south")
        assert "host_id" in str(exc_info.value)

    def test_neither_set_is_error(self):
        with pytest.raises(Exception) as exc_info:
            Anchor(attach_to="face")
        assert "host=" in str(exc_info.value)

    def test_anchor_stays_frozen(self):
        a = Anchor(host="wall_south")
        with pytest.raises(Exception):
            a.host = "other"

    def test_extra_fields_still_forbidden(self):
        with pytest.raises(Exception):
            Anchor(host="wall_south", rotation=9000)


# ---------------------------------------------------------------------------
# 2. Compile-time host resolution
# ---------------------------------------------------------------------------


class TestHostResolution:
    def test_host_naming_existing_element_is_clean(self):
        b = _building(
            _box(name="wall_south"),
            _box(name="shelf", offset=6000,
                 placement=Anchor(host="box:wall_south", attach_to="face")),
        )
        report = validate_project_report(b)
        assert [e for e in report.errors if "Anchor" in e] == []
        assert [w for w in report.warnings if "Anchor" in w] == []

    def test_unresolvable_host_is_compile_error(self):
        b = _building(
            _box(name="wall_south"),
            _box(name="shelf", offset=6000,
                 placement=Anchor(host="wall_ghost")),
        )
        report = validate_project_report(b)
        errs = [e for e in report.errors if "Anchor" in e]
        assert len(errs) == 1
        assert "'wall_ghost'" in errs[0]
        assert "matches no element" in errs[0]

    def test_host_matching_only_an_anonymous_id_is_miss(self):
        """DSL v2.1: id= is never a canonical name, so anchoring to an
        anonymous element's id resolves to nothing — a miss ERROR."""
        anon = _box()  # anonymous — auto elem_<hash> id, no canonical
        b = _building(
            anon,
            _box(name="shelf", offset=6000, placement=Anchor(host=anon.id)),
        )
        report = validate_project_report(b)
        errs = [e for e in report.errors if "Anchor" in e]
        assert len(errs) == 1
        assert "matches no element" in errs[0]

    def test_host_resolves_against_container_children_names(self):
        from lite_step.models import Wall

        wall = Wall(name="wall_w")
        wall.add(_box(name="wall_w_body"))
        b = _building(
            wall,
            _box(name="shelf", offset=6000,
                 placement=Anchor(host="box:wall_w_body:wall:wall_w")),
        )
        report = validate_project_report(b)
        assert [e for e in report.errors if "Anchor" in e] == []
