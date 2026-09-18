"""Opting a subtree out of the inferred-carve pass — WS-A §1.8, finished.

``placement=Transform(origin=Point(x=0, y=0, z=0))`` appeared 16 times across
10 skill files. It SAYS "placed at the origin"; it MEANS "opt this subtree out
of the inferred-carve pass", and omitting it silently zeroes a terrain mesh.
Every use site carried a comment because the code could not say it.

The verb `.no_carve()` was §1.8's answer and is now DELETED; ``.add(carve=
"none")`` is the spelling. Two things settled that, and both are silent
failures the verb could not avoid by construction:

* ``Product(source)`` SNAPSHOTS, so ``source.no_carve()`` afterwards did
  nothing at all — measured, five ground plates kept carving.
* The same source added in two places was exempt in both or neither, because
  the flag rode on the element rather than on the placement.

This file keeps what is still true after the removal: that the exemption
matches what the idiom bought by accident, that the idiom still works for its
own reason, and that the state survives the boundaries the pass runs across.
The DIRECTIONS themselves (``"other"`` / ``"self"`` / ``"none"``, precedence,
scope, refusals) are ``test_carve_direction.py``.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.displacement import apply_displacement, carve_pairs
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import Box, Element, Point, Project, Transform, Wall
from lite_step.models.project import Storey


def _big(name):
    """A slab of ground the whole model stands in — the shape that made the
    idiom load-bearing in the first place."""
    return Box(name=name, start=Point(x=-20000, y=-20000, z=-2000),
               end=Point(x=20000, y=20000, z=0), material="Concrete")


def _occupant(name):
    return Box(name=name, start=Point(x=-1000, y=-1000, z=-1500),
               end=Point(x=1000, y=1000, z=3000), material="Concrete")


def _pairs(*tops):
    """``tops`` is ``element`` or ``(element, carve)``."""
    proj = Project(name="t")
    storey = Storey(elevation=0)
    for t in tops:
        if isinstance(t, tuple):
            storey.add(t[0], carve=t[1])
        else:
            storey.add(t)
    proj.add_storey(storey)
    norm = normalize_project_to_meters(proj)
    apply_displacement(norm)
    return carve_pairs(norm)


def _idiom(name):
    box = _occupant(name)
    box.placement = Transform(origin=Point(x=0, y=0, z=0))
    return box


def test_without_the_opt_out_the_occupant_carves():
    """The baseline the other tests are measured against. Without this the
    exemption tests would pass on a model where nothing carves anyway —
    fixtures at defaults cannot falsify."""
    pairs = _pairs(_big("ground"), _occupant("post"))
    assert pairs, "the occupant must carve the ground when nothing opts out"
    assert any(loser == "box:ground" for loser, _w, _m in pairs)


def test_carve_none_takes_the_element_out_of_the_pass():
    assert _pairs(_big("ground"), (_occupant("post"), "none")) == []


def test_carve_none_matches_the_identity_transform_idiom_exactly():
    """The migration claim, stated as an equality rather than asserted in
    prose: the argument buys the same exemption the idiom bought."""
    arg = _pairs(_big("ground"), (_occupant("post"), "none"))
    idiom = _pairs(_big("ground"), _idiom("post"))
    assert arg == idiom == []


def test_the_idiom_still_works():
    """It keeps working for its OWN reason — a pre-placement AABB cannot be
    judged — not because the exemption special-cases it. Nothing in the corpus
    spells it this way any more, so this is the guard against removing the
    placement exemption while cleaning up after the migration."""
    assert _pairs(_big("ground"), _idiom("post")) == []


def test_the_exemption_covers_the_whole_subtree():
    """An exempt assembly's own parts are exempt with it — or the bricks inside
    a bracket eat the wall the bracket hangs off."""
    wall = Wall(name="shell")
    wall.add(_occupant("post"))
    assert _pairs(_big("ground"), (wall, "none")) == []


def test_the_exemption_is_available_on_containers_too():
    """The idiom it replaced sat on whatever the skill was placing — often an
    ``Element`` wrapper, not a primitive."""
    wrapper = Element(ifc_class="IfcBuildingElementProxy", name="deck")
    wrapper.add(_occupant("plank"))
    assert _pairs(_big("ground"), (wrapper, "none")) == []


def test_the_exemption_survives_normalization():
    """The pass runs on the normalized COPY, so the flag has to cross that
    boundary. Pinned separately because ``normalize_project_to_meters``
    rebuilds elements via ``model_construct`` in places."""
    proj = Project(name="t")
    storey = Storey(elevation=0)
    storey.add(_big("ground"))
    storey.add(_occupant("post"), carve="none")
    proj.add_storey(storey)
    norm = normalize_project_to_meters(proj)
    posts = [e for e in norm.storeys[0].elements if e.name == "post"]
    assert posts and posts[0]._no_carve is True


def test_derivation_by_call_carries_an_exemption_already_taken():
    """``instance(field=…)`` deep-copies private state, so deriving from an
    element that has ALREADY been added exempt yields an exempt copy — even
    though the derived one is added without the argument.

    Pinned as behaviour rather than argued as design. It is the one way the
    per-placement property can still be sidestepped, and the sidestep is
    LOUD in a way the verb's failure was not: the author reads
    ``derived = exempt_post(name="post2")`` and can see where it came from,
    whereas ``source.no_carve()`` after a ``Product(source)`` looked like it
    did something and did nothing. Deriving BEFORE the add — the normal
    order — carries nothing, which is the case above it.
    """
    original = _occupant("post")
    assert _pairs(_big("ground"), (original, "none")) == []

    derived = original(name="post2")
    assert derived._no_carve is True
    assert _pairs(_big("ground"), derived) == []

    # The control: a copy derived from an element that was never added exempt
    # carries nothing, so the assertion above is about the deep-copy and not
    # about the fixture failing to overlap.
    fresh = _occupant("post")(name="post3")
    assert fresh._no_carve is False
    assert _pairs(_big("ground"), fresh) != []


def test_the_verb_is_gone():
    """``.no_carve()`` is removed, not deprecated.

    A live model that calls it must fail LOUDLY. The alternative — leaving a
    no-op or a warning shim — reproduces the exact failure the removal exists
    to end: a call that reads as an exemption and grants none.
    """
    box = _occupant("post")
    assert not hasattr(box, "no_carve")
    with pytest.raises(AttributeError, match="no_carve"):
        box.no_carve()
