"""Tests for element ID uniqueness validation."""

from lite_step.models.project import Project, Storey
from lite_step.models.primitives import Point
from lite_step.models.elements import Wall, Box, Extrude
from lite_step.compiler.executor import validate_project


def _wall(id: str) -> Wall:
    """Create a minimal valid Wall with a Solid body."""
    wall = Wall(id=id)
    wall.add(Box(
        id=f"{id}_body",
        start=Point(x=0, y=0, z=0),
        end=Point(x=5000, y=3000, z=200),
        type="wall",
    ))
    return wall


def _building(*storeys_data) -> Project:
    """Create a Project from (name, elevation, elements) tuples."""
    proj = Project(name="Test")
    for name, elevation, elements in storeys_data:
        storey = Storey(name=name, elevation=elevation)
        for elem in elements:
            storey.add(elem)
        proj.add_storey(storey)
    return proj


def test_unique_ids_pass():
    """No errors when all element IDs are unique."""
    proj = _building((None, 0, [_wall("wall_north"), _wall("wall_south")]))
    errors = validate_project(proj)
    dup_errors = [e for e in errors if "Duplicate" in e]
    assert len(dup_errors) == 0


def test_duplicate_ids_no_longer_error():
    """DSL v2.1: id= is internal-only (never an emitted Name), so the
    there is no id-uniqueness check — a duplicate id= does not error.
    Identity uniqueness is canonical-name-based (test_name_identity)."""
    proj = _building((None, 0, [_wall("wall_north"), _wall("wall_north")]))
    errors = validate_project(proj)
    # No "Duplicate element ID" error; the walls are anonymous (no name=), so
    # no canonical-name collision either.
    assert [e for e in errors if "Duplicate element ID" in e] == []


def test_duplicate_ids_across_storeys_no_longer_error():
    """Same as above, across storeys — id= collisions are not checked."""
    proj = _building(
        ("ground", 0, [_wall("shared_wall")]),
        ("first", 3000, [_wall("shared_wall")]),
    )
    errors = validate_project(proj)
    assert [e for e in errors if "Duplicate element ID" in e] == []


def test_auto_generated_ids_are_unique():
    """Default UUID-based IDs should never collide."""
    wall1 = Wall()
    wall1.add(Box(start=Point(x=0, y=0, z=0), end=Point(x=5000, y=3000, z=200), type="wall"))
    wall2 = Wall()
    wall2.add(Box(start=Point(x=0, y=0, z=1000), end=Point(x=5000, y=3000, z=1200), type="wall"))
    proj = _building((None, 0, [wall1, wall2]))
    errors = validate_project(proj)
    dup_errors = [e for e in errors if "Duplicate" in e]
    assert len(dup_errors) == 0
