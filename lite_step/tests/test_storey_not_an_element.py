"""``.add()`` refuses a ``Storey`` and names ``add_storey()``.

Pre-fix, ``proj.add(Storey(name="first", elevation=3000))`` was accepted in
silence: ``Project.add`` filed the storey into the DEFAULT storey's element
list — a storey as a child element of another storey — and the model died
much later, inside emission, with::

    AttributeError: 'Storey' object has no attribute 'ifc_name'

naming neither the mistake nor the fix. ``.add()`` is the universal
containment verb everywhere else in the DSL and ``add_storey`` is one
keystroke away, so this is the ordinary mistake, not a careless one. The
repo's own standard is already stricter elsewhere: ``wall.add(window)`` is
a compile ERROR that names the ``.anchor()`` call which fixes it.

Both element-taking ``.add()`` methods on the containment spine are
covered. ``Project.add`` delegates to ``storeys[0].add``, so guarding only
one would leave the other route open — and ``proj.storeys[0].add(storey)``
is the spelling an author reaches for second, right after reading that
storeys hold elements.
"""

from __future__ import annotations

import pytest

from lite_step.compiler.executor import execute_lite_step_script
from lite_step.models import Box, Point, Project, Site, Storey


def _storey() -> Storey:
    """Non-zero, non-round elevation — a storey at 0 cannot show that the
    elevation survived, and cannot fail a normalizer that drops it."""
    return Storey(name="first", elevation=3175)


def test_project_add_refuses_a_storey():
    proj = Project(name="house")
    with pytest.raises(ValueError) as exc:
        proj.add(_storey())
    assert "add_storey" in str(exc.value), str(exc.value)


def test_the_refusal_teaches_the_correct_spelling():
    """The message must name BOTH verbs: the one that adds the storey and
    the one that fills it. An error that only says "wrong" sends the author
    back to the docs, which is the cost this refusal exists to remove."""
    proj = Project(name="house")
    with pytest.raises(ValueError) as exc:
        proj.add(_storey())
    msg = str(exc.value)
    assert "proj.add_storey(" in msg, msg
    assert "proj.storeys[n].add(" in msg, msg
    # It names the offending storey, so a 4-storey file says which line.
    assert "'first'" in msg, msg
    # And it names the failure it prevents, so the two are searchable
    # together when someone hits the error in an archived log.
    assert "ifc_name" in msg, msg


def test_a_refused_add_does_not_mutate_the_project():
    """No phantom storey on the way to the error.

    ``Project.add`` auto-creates the default storey before delegating, so a
    guard placed one line later would leave a project holding an empty
    anonymous storey it never asked for — and multi-storey models require
    every storey to be NAMED, so that phantom would turn one clear error
    into a second, unrelated one.
    """
    proj = Project(name="house")
    with pytest.raises(ValueError):
        proj.add(_storey())
    assert proj.storeys == []
    assert proj.sites == []


def test_storey_add_refuses_a_storey():
    """The delegate target, reached directly."""
    proj = Project(name="house")
    proj.add_storey(Storey(name="ground", elevation=0))
    with pytest.raises(ValueError) as exc:
        proj.storeys[0].add(_storey())
    assert "add_storey" in str(exc.value)


def test_a_refused_storey_add_leaves_the_storey_empty():
    proj = Project(name="house")
    proj.add_storey(Storey(name="ground", elevation=0))
    proj.storeys[0].add(Box(name="plinth",
                            start=Point(x=-3100, y=850, z=0),
                            end=Point(x=4700, y=2300, z=615)))
    with pytest.raises(ValueError):
        proj.storeys[0].add(_storey(), Box(name="second",
                                           start=Point(x=0, y=0, z=0),
                                           end=Point(x=100, y=100, z=100)))
    # The valid sibling in the SAME varargs call must not have landed
    # either — a partially-applied .add() is worse than a refused one.
    assert [e.name for e in proj.storeys[0].elements] == ["plinth"]


def test_the_refusal_fires_on_a_storey_anywhere_in_the_varargs():
    proj = Project(name="house")
    with pytest.raises(ValueError):
        proj.add(
            Box(name="a", start=Point(x=0, y=0, z=0),
                end=Point(x=100, y=100, z=100)),
            _storey(),
        )
    assert proj.storeys == []


def test_add_storey_still_works_and_keeps_the_elevation():
    """The guard must not break the verb it recommends."""
    proj = Project(name="house")
    proj.add_storey(Storey(name="ground", elevation=0))
    proj.add_storey(_storey())
    assert [s.name for s in proj.storeys] == ["ground", "first"]
    assert proj.storeys[1].elevation == 3175


def test_site_is_still_routed_not_refused():
    """``Project.add`` has exactly one other special-cased argument type.

    A guard written as "refuse anything that isn't a BimElement" would have
    taken ``Site`` with it; this states that it did not.
    """
    proj = Project(name="house")
    proj.add(Site(name="plot"))
    assert len(proj.sites) == 1
    assert proj.storeys == []


def test_the_refusal_reaches_an_authored_script():
    """The path an LLM-authored model actually takes: the refusal must
    surface as a compile error with the advice intact, not as an internal
    traceback the executor swallows."""
    script = """
from lite_step.models import Project, Storey

def generate_project():
    proj = Project(name="house")
    proj.add(Storey(name="first", elevation=3175))
    return proj

result = generate_project()
"""
    r = execute_lite_step_script(script)
    assert r.success is False
    assert "add_storey" in (r.error or ""), r.error
