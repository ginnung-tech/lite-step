"""Copying one element never copies the model around it.

``BimElement.__deepcopy__`` drops the upward ``_parent`` pointer so a copy of
one element does not drag in its container, storey and every sibling. Since
pydantic 2.14, ``model_copy(deep=True)`` no longer calls that override — it
deep-copies ``__pydantic_private__`` itself — so ``_parent`` was copied and
every inferred carve copied a storey. Measured on an 829-carve model:
15 s on pydantic 2.13, 185 s on 2.14.0. ``detached_copy`` goes through
``copy.deepcopy`` and so through the override on every pydantic version.
"""
from __future__ import annotations

import contextlib

from lite_step.compiler.displacement import apply_displacement
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.models import Box, Point, Project, Wall
from lite_step.models.elements import detached_copy
from lite_step.models.project import Storey


def _storey_with_siblings():
    p = Project(name="copy")
    s = Storey(name="ground")
    host = Box(name="slab", start=Point(x=0, y=0, z=0), end=Point(x=2000, y=1000, z=300))
    post = Box(name="post", start=Point(x=900, y=400, z=-200), end=Point(x=1100, y=600, z=2000))
    s.add(host)
    s.add(post)
    for i in range(20):
        s.add(Box(name=f"far_{i}", start=Point(x=10000 + i * 1000, y=0, z=0),
                  end=Point(x=10500 + i * 1000, y=500, z=500)))
    p.add_storey(s)
    return normalize_project_to_meters(p)


@contextlib.contextmanager
def _no_storey_copies():
    """Copying a Storey or the Project while this is open is the bug. Armed
    only around the copy under test: normalising deep-copies the whole
    project, legitimately."""
    def refuse(self, memo=None):
        raise AssertionError(f"deep-copied {type(self).__name__} — an element "
                             f"copy followed _parent up the tree")
    saved = {cls: cls.__dict__.get("__deepcopy__") for cls in (Storey, Project)}
    try:
        for cls in saved:
            cls.__deepcopy__ = refuse
        yield
    finally:
        for cls, orig in saved.items():
            if orig is None:
                del cls.__deepcopy__
            else:
                cls.__deepcopy__ = orig


def test_detached_copy_does_not_copy_the_storey():
    proj = _storey_with_siblings()
    post = proj.storeys[0].elements[1]
    assert post._parent is not None, "fixture: the element must have a parent"
    with _no_storey_copies():
        copy = detached_copy(post, name=None)
    assert copy._parent is None
    assert copy.name is None and "name" in copy.__pydantic_fields_set__
    assert copy is not post and copy.start == post.start


def test_an_inferred_carve_copies_only_the_occupant():
    proj = _storey_with_siblings()
    with _no_storey_copies():
        apply_displacement(proj)
    slab = proj.storeys[0].elements[0]
    assert len(slab._cuts) == 1
    assert slab._cuts[0]._parent is None
    assert slab._cuts[0]._carve_pair is not None


def test_a_nested_element_copy_stays_inside_its_subtree():
    """A container child's parent is the container, whose parent is the
    storey: two levels up, and neither may be copied."""
    p = Project(name="copy")
    s = Storey(name="ground")
    w = Wall(name="north")
    body = Box(name="body", start=Point(x=0, y=0, z=0), end=Point(x=4000, y=300, z=2700))
    w.add(body)
    s.add(w)
    p.add_storey(s)
    proj = normalize_project_to_meters(p)
    child = proj.storeys[0].elements[0]._elements[0]
    with _no_storey_copies():
        assert detached_copy(child)._parent is None
