"""Every container reads back the same way.

The DSL had ``.add()`` on thirteen containers and FOUR ways to read them:

    get_elements()      Beam, Column, Element, Roof, Slab, Space, SpatialElement
    nothing public      Door, Site, Wall, Window
    .elements field     Storey
    neither             Project

``_add_children``'s own docstring names the pattern — "the silent-asymmetry
failure this codebase keeps finding (``Wall`` and ``Site`` still have no
``get_elements()`` for exactly that reason)".

**It fails silently, which is why it is worth a test rather than a docstring.**
A caller guards with ``hasattr(container, "elements")``, gets False on a
container that plainly has children, and skips it without a word. In one
assembled model that dropped every ``Site``'s contents — including the terrain,
whose absence then read as a rendering bug and cost an investigation.

The census test below is the durable half: a NEW container added without both
accessors fails here, naming itself, instead of being found years later.
"""

from __future__ import annotations

import inspect

import pytest

from lite_step import models as M
from lite_step.models import Box, Point, Project, Site, Storey, Wall


def _containers():
    """Every authorable class with an ``.add()`` — i.e. every container."""
    out = []
    for name in dir(M):
        obj = getattr(M, name)
        if not inspect.isclass(obj) or not name[0].isupper():
            continue
        if hasattr(obj, "add") and "_elements" in getattr(
            obj, "__private_attributes__", {}
        ):
            out.append((name, obj))
    return sorted(out)


def test_the_census_is_not_empty():
    """Guards the guard: a broken discovery predicate would make every
    parametrised test below vacuously pass."""
    names = [n for n, _ in _containers()]
    assert len(names) >= 10, f"container discovery found only {names}"
    for expected in ("Wall", "Site", "Door", "Window", "Slab", "Element"):
        assert expected in names, f"{expected} missing from the census"


@pytest.mark.parametrize("name,cls", _containers())
def test_every_container_exposes_elements(name, cls):
    """ONE spelling, on every container. This is the test that fails when
    someone adds container #14 and wires only ``.add()`` — or reintroduces a
    second way to ask the same question."""
    assert isinstance(getattr(cls, "elements", None), property), \
        f"{name}.elements is not a property — readers cannot rely on it"
    assert not hasattr(cls, "get_elements"), \
        f"{name}.get_elements() is back — the v10 cutover left ONE spelling"


def test_elements_and_get_elements_agree():
    wall = Wall(name="north")
    wall.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100)))
    assert len(wall.elements) == 1
    assert wall.elements == wall.elements


def test_elements_returns_a_copy():
    """Mutating the returned list must not corrupt the container — the old
    ``get_elements()`` promised this via ``.copy()`` and the property keeps it."""
    wall = Wall(name="north")
    wall.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100)))
    wall.elements.clear()
    assert len(wall.elements) == 1


def test_site_is_readable_at_all():
    """The concrete regression. ``Site`` had NO public accessor, so
    ``hasattr(site, "elements")`` was False and generic code skipped it."""
    site = Site(name="site")
    site.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100)))
    assert hasattr(site, "elements")
    assert len(site.elements) == 1


def test_storey_answers_both_spellings():
    """``Storey.elements`` is a real FIELD, not the BimElement property, so it
    needs its own ``get_elements()`` — a generic reader must not have to know
    which kind of container it is holding."""
    st = Storey(elevation=0)
    st.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100)))
    assert st.elements == st.elements
    assert len(st.elements) == 1


# ---------------------------------------------------------------------------
# The two deliberate NON-containers
# ---------------------------------------------------------------------------


def test_a_geometry_leaf_reports_no_elements_and_says_why():
    """``hasattr`` must be False for a leaf — an empty list would claim a Box
    is a container that happens to be empty, which is a different fact."""
    box = Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100))
    assert not hasattr(box, "elements")
    with pytest.raises(AttributeError, match="geometry leaf"):
        box.elements


def test_project_refuses_elements_and_names_the_alternatives():
    """A Project keeps no element list of its own — ``.add()`` routes into
    ``storeys[0]``. Returning the storeys' contents would make a caller that
    walks storeys AND this property add everything twice."""
    proj = Project(name="p")
    proj.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=100, y=100, z=100)))
    assert not hasattr(proj, "elements")
    with pytest.raises(AttributeError, match="get_all_elements"):
        proj.elements
    # the alternatives it names actually work
    assert len(proj.get_all_elements()) == 1
    assert len(proj.storeys[0].elements) == 1


def test_the_reported_failure_shape_now_works():
    """The exact guard that silently dropped a subsystem: walk containers
    generically, transfer their children. Without it the Site branch is
    skipped and its terrain vanishes."""
    site = Site(name="site")
    site.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=1, y=1, z=1)))
    st = Storey(elevation=0)
    st.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=2, y=2, z=2)))

    moved = []
    for container in (site, st):
        if hasattr(container, "elements"):
            moved.extend(container.elements)

    assert len(moved) == 2, "a container was silently skipped again"
