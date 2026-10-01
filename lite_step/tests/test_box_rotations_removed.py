"""``Box(rotations=)`` is a hard error; ``placement=Transform`` is the one path.

The field used to be accepted and rotate the emitted solid while every bounds
query (``world_aabb``) ignored it, so a rotated box reported its unrotated
extent. Removing it, with a message that names the replacement, is the loud
form of that failure.
"""
import pytest
from pydantic import ValidationError

from lite_step.models import Box, Point, Project, Transform


def _centred_box(**kw):
    return Box(start=Point(x=-200, y=-50, z=0), end=Point(x=200, y=50, z=100),
               name="b", **kw)


def test_box_rotations_raises_and_points_at_transform():
    with pytest.raises(ValidationError) as exc:
        Box(start=Point(x=0, y=0, z=0), end=Point(x=400, y=100, z=100),
            rotations=[("z", 9000)])
    msg = str(exc.value)
    assert "Box(rotations=" in msg
    assert "placement=Transform(" in msg
    assert "[('z', 9000)]" in msg  # shows the caller's own value in the fix


def test_box_rotations_empty_list_is_refused_too():
    # A present-but-empty argument is still the removed API, not a no-op.
    with pytest.raises(ValidationError, match="placement=Transform"):
        _centred_box(rotations=[])


def test_transform_rotation_swaps_box_extents_in_world_aabb():
    box = _centred_box()
    box.placement = Transform(origin=Point(x=1000, y=2000, z=0),
                              rotations=[("z", 9000)])
    Project(name="t").add(box)

    w = box.world_aabb()
    assert (w.size.x, w.size.y, w.size.z) == (100, 400, 100)
    # Centre stays at origin=: the box was authored centred on (0, 0).
    assert (w.min.x, w.max.x) == (950, 1050)
    assert (w.min.y, w.max.y) == (1800, 2200)
