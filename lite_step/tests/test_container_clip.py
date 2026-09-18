"""``.clip()`` on a CONTAINER reaches the parts under it.

Without this pass, ``slab.clip(...)`` appends to a ``_clips`` list nothing
downstream reads: byte-identical IFC, no half-space, no warning. The measured
consequence one level up was worse — authors moved the call to
``body.clip(...)``, which cut the concrete and left every bar where it was:
1007 m of 6941 m (14.5%) of a hospital arm's reinforcement outside its own
mitered sector.

Three things are asserted here and each one can fail on its own:

* the cut REACHES the parts (the repro, both spellings and a control);
* a linear primitive is TRIMMED rather than subtracted — it stays a ``Bar``,
  costs no CSG depth, and a bar trimmed to nothing LEAVES the tree;
* ``overrun`` offsets the path trim and ONLY the path trim. The boolean
  half-space stays flush at the plane whatever it says. That asymmetry is the
  feature: at a mitered joint the concrete is cut on the bisector while the
  reinforcement laps past it into the neighbour.
"""
from __future__ import annotations

import pytest

from lite_step.compiler.composite_clip import (
    CompositeClipError, iter_parts, trim_path,
)
from lite_step.compiler.executor import normalize_project_to_meters
from lite_step.ifc.generator import generate_ifc
from lite_step.models import (
    Bar, Box, HalfSpace, Point, Project, Slab,
)
from lite_step.models.project import Storey


# ── helpers ───────────────────────────────────────────────────────────────────

def _project(slab) -> Project:
    proj = Project(name="container clip")
    g = Storey(name="ground", elevation=0)
    g.add(slab)
    proj.add_storey(g)
    return proj


def _compile(slab) -> str:
    """Normalize + emit, and hand back the STEP text."""
    ifc = generate_ifc(normalize_project_to_meters(_project(slab)))
    assert ifc.success, ifc.error
    return ifc.ifc_content


def _plate(**slab_kwargs) -> tuple:
    """A 24 x 24 m plate, 275 mm thick, as (slab, body)."""
    s = Slab(name="plate", **slab_kwargs)
    body = Box(name="body", material="Concrete_C30-37",
               start=Point(x=0, y=-12000, z=0),
               end=Point(x=24000, y=12000, z=275))
    return s, body


def _bar(x0: int, x1: int, name: str = "b") -> Bar:
    """A straight bar running along +X at mid-depth."""
    return Bar(name=name, diameter=16,
               path=[Point(x=x0, y=0, z=100), Point(x=x1, y=0, z=100)])


def _paths_mm(elem) -> list:
    """Every surviving path under ``elem``, back in millimetres."""
    return [[(round(p.x * 1000), round(p.y * 1000), round(p.z * 1000))
             for p in part.path]
            for part in iter_parts(elem) if getattr(part, "path", None)]


# ── the repro ────────────────────────────────────────────────────────────

def test_a_container_clip_emits_what_the_primitive_clip_emits():
    """The headline. Container and primitive spellings now agree, and the
    no-clip control proves the count is not simply always 1.

    Counted on the emitted STEP rather than on ``_clips``: the whole defect was
    that the list held a half-space the file never saw.
    """
    def halfspaces(text):
        return text.count("IFCHALFSPACESOLID")

    s0, b0 = _plate()
    s0.add(b0)
    control = _compile(s0)

    s1, b1 = _plate()
    s1.add(b1)
    s1.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))   # CONTAINER
    container = _compile(s1)

    s2, b2 = _plate()
    b2.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))   # PRIMITIVE
    s2.add(b2)
    primitive = _compile(s2)

    assert halfspaces(control) == 0
    assert halfspaces(primitive) == 1
    assert halfspaces(container) == 1, (
        "a container clip emitted no half-space — this is, the silent "
        "no-op this pass exists to fix"
    )


def test_a_clip_on_a_container_with_no_geometry_raises():
    """The call cannot act, so it says so.

    Silence here is what WAS. An empty container is the one case where
    distributing has nothing to distribute to, and returning ``self`` quietly
    is indistinguishable from the bug.
    """
    s = Slab(name="empty")
    s.clip(origin=Point(x=0, y=0, z=0), normal=(1, 0, 0))
    with pytest.raises(CompositeClipError, match="nothing to cut"):
        normalize_project_to_meters(_project(s))


def test_the_cut_reaches_parts_added_after_the_clip_line():
    """Deferred, not eager — the property that makes the call order-free.

    ``Product(source)`` snapshots at construction, and a ``.no_carve()`` after
    it silently does nothing; five ground plates kept carving because of that.
    Resolving a clip eagerly would be the same trap in a second place.
    """
    s, body = _plate()
    s.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))
    s.add(body)                       # added AFTER the clip
    s.add(_bar(0, 24000, "late"))     # so is this

    norm = normalize_project_to_meters(_project(s))
    slab = norm.storeys[0].elements[0]
    assert _paths_mm(slab) == [[(0, 0, 100), (12000, 0, 100)]]


# ── a path is trimmed, not subtracted ─────────────────────────────────────────

def test_a_crossing_bar_is_trimmed_and_stays_a_bar():
    s, body = _plate()
    s.add(body)
    s.add(_bar(0, 24000))
    s.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))

    norm = normalize_project_to_meters(_project(s))
    slab = norm.storeys[0].elements[0]
    bars = [p for p in iter_parts(slab) if isinstance(p, Bar)]
    assert len(bars) == 1                       # still a Bar, not a boolean
    assert not bars[0]._clips, (
        "the bar took the half-space as a BOOLEAN — that costs one CSG level "
        "per bar (the hospital slab went to 12 of a 10 budget that way, and "
        "the viewer silently renders the uncarved solid past ~13)"
    )
    assert _paths_mm(slab) == [[(0, 0, 100), (12000, 0, 100)]]


def test_a_bar_wholly_on_the_removed_side_leaves_the_tree():
    """Dropped, not emitted at zero length.

    A zero-length ``IfcSweptDiskSolid`` is the silent failure this corpus keeps
    paying for; a bar with nothing left is not a short bar, it is no bar.
    """
    s, body = _plate()
    s.add(body)
    s.add(_bar(0, 8000, "kept"))
    s.add(_bar(16000, 24000, "gone"))
    s.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))

    norm = normalize_project_to_meters(_project(s))
    slab = norm.storeys[0].elements[0]
    bars = [p for p in iter_parts(slab) if isinstance(p, Bar)]
    assert [b.name for b in bars] == ["kept"]


def test_a_bar_wholly_on_the_kept_side_is_not_rebuilt():
    """Untouched means the same object, not an equal one.

    Rebuilding every uncut bar from arithmetic would churn coordinates through
    a float round-trip on models where nothing was supposed to change.

    Asserted at the trim rather than through a compile, because the normalizer
    deep-copies the whole tree — after it, every path is a new object whatever
    this pass did, so a compile cannot see the property at all.
    """
    path = [Point(x=0, y=0, z=100), Point(x=8000, y=0, z=100)]
    assert trim_path(path, Point(x=12000, y=0, z=0), (1, 0, 0)) is path


# ── overrun ───────────────────────────────────────────────────────────────────

def test_overrun_lets_a_crossing_bar_reach_exactly_that_far_past_the_plane():
    """The acceptance criterion that replaced the flush one.

    The retired criterion — "0.0% of bar length outside the kept sector" —
    asserted the BUG: it is the flush assertion, and flush is what a lapped
    joint must not be. This one can fail in both directions: a trim that
    silently did not happen leaves the bar at 24000, and one that over-reaches
    leaves it past 12500.
    """
    s, body = _plate()
    s.add(body)
    s.add(_bar(0, 24000))
    s.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0), overrun=500)

    norm = normalize_project_to_meters(_project(s))
    slab = norm.storeys[0].elements[0]
    assert _paths_mm(slab) == [[(0, 0, 100), (12500, 0, 100)]]


def test_overrun_never_extends_a_bar_that_ends_short_of_the_lap_line():
    """*At most* ``overrun``, and *exactly* only when the bar has the length.

    Padding a short bar out to the lap line would invent steel that is in no
    schedule and no delivery — the trim only ever shortens, which falls out of
    every emitted point being one of the path's own or interpolated between
    two of them.

    The LONG bar in this model is not decoration: on its own, "the short bar
    came out unchanged" is exactly what a pass that did nothing at all would
    produce, so the assertion could not fail. The pair makes it falsifiable in
    both directions at once.
    """
    s, body = _plate()
    s.add(body)
    s.add(_bar(0, 12200, "short"))            # crosses, but only by 200
    s.add(_bar(0, 24000, "long"))             # crosses with room to spare
    s.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0), overrun=500)

    norm = normalize_project_to_meters(_project(s))
    slab = norm.storeys[0].elements[0]
    assert _paths_mm(slab) == [
        [(0, 0, 100), (12200, 0, 100)],       # short: untouched, not stretched
        [(0, 0, 100), (12500, 0, 100)],       # long:  stopped at plane + lap
    ]


def test_overrun_is_measured_as_a_real_distance_on_an_unnormalized_normal():
    """``.clip()`` accepts any non-zero direction and the boolean does not care
    about its length — but ``overrun`` is a LENGTH. With ``(2,0,0)`` taken at
    face value a 500 mm lap would trim at 250.
    """
    s, body = _plate()
    s.add(body)
    s.add(_bar(0, 24000))
    s.clip(origin=Point(x=12000, y=0, z=0), normal=(2, 0, 0), overrun=500)

    norm = normalize_project_to_meters(_project(s))
    slab = norm.storeys[0].elements[0]
    assert _paths_mm(slab) == [[(0, 0, 100), (12500, 0, 100)]]


def test_the_boolean_half_space_stays_flush_whatever_the_overrun_says():
    """The asymmetry, asserted on the emitted geometry rather than on intent.

    Concrete is mitered on the bisector; steel laps past it. A solid that
    honoured ``overrun`` would push the concrete past the joint and into its
    neighbour — two slabs occupying the same 500 mm, and a takeoff that
    double-counts it.
    """
    def clip_plane_x(slab):
        norm = normalize_project_to_meters(_project(slab))
        body = [p for p in iter_parts(norm.storeys[0].elements[0])
                if isinstance(p, Box)][0]
        assert len(body._clips) == 1
        return body._clips[0].origin.x

    s_flush, b_flush = _plate()
    s_flush.add(b_flush)
    s_flush.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))

    s_lap, b_lap = _plate()
    s_lap.add(b_lap)
    s_lap.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0), overrun=500)

    assert clip_plane_x(s_flush) == clip_plane_x(s_lap) == pytest.approx(12.0)


def test_a_negative_overrun_is_refused():
    """Pulling a path SHORT of the plane is a shorter element, not a clip —
    and the trim cannot express it anyway (it only ever shortens along the
    path's own vertices)."""
    with pytest.raises(ValueError, match="overrun must be >= 0"):
        HalfSpace(origin=Point(x=0, y=0, z=0), normal=(1, 0, 0), overrun=-1)


# ── the pass leaves the existing path alone ───────────────────────────────────

def test_a_primitives_own_clip_is_left_for_the_generator():
    """The pre-existing path is untouched: a geometry-bearing element still
    consumes its own clips through the generator, and the pass must not
    re-home them onto itself and double-cut."""
    s, body = _plate()
    body.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))
    s.add(body)

    norm = normalize_project_to_meters(_project(s))
    box = [p for p in iter_parts(norm.storeys[0].elements[0])
           if isinstance(p, Box)][0]
    assert len(box._clips) == 1


def test_a_nested_container_is_not_cut_twice_by_the_same_plane():
    """Depth-first: the inner container consumes its own clip before the outer
    one distributes, so the inner parts see each plane exactly once."""
    inner = Slab(name="inner")
    inner.add(_bar(0, 24000, "b"))
    inner.clip(origin=Point(x=16000, y=0, z=0), normal=(1, 0, 0))

    outer, body = _plate()
    outer.add(body)
    outer.add(inner)
    outer.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))

    norm = normalize_project_to_meters(_project(outer))
    slab = norm.storeys[0].elements[0]
    # inner's own plane cuts to 16000, outer's to 12000 — the tighter wins,
    # and neither is applied twice (which would be invisible here but shows up
    # as a doubled boolean on a solid).
    assert _paths_mm(slab) == [[(0, 0, 100), (12000, 0, 100)]]


def test_the_pass_is_idempotent():
    """Re-running must not re-cut. ``_clips`` is cleared on consumption, which
    is what makes that true — asserted, because "cleared" is the kind of
    bookkeeping a refactor drops silently, and a second application would show
    up as a solid carrying two identical half-spaces (a wasted CSG level) with
    geometry that still looks right.

    ``resolve`` is called explicitly rather than leaning on the executor's own
    call, so this measures the property whatever the hook does.
    """
    from lite_step.compiler.composite_clip import resolve

    s, body = _plate()
    s.add(body)
    s.add(_bar(0, 24000))
    s.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))

    norm = normalize_project_to_meters(_project(s))
    slab = norm.storeys[0].elements[0]
    resolve(norm)
    resolve(norm)
    assert _paths_mm(slab) == [[(0, 0, 100), (12000, 0, 100)]]
    box = [p for p in iter_parts(slab) if isinstance(p, Box)][0]
    assert len(box._clips) == 1


# ── the report ────────────────────────────────────────────────────────────────

def test_the_report_names_the_overrun_even_when_it_is_flush(capsys):
    """A flush cut IS a construction joint. Stating it is the only way a
    reader can tell a lap that was applied from one that was silently
    dropped."""
    s, body = _plate()
    s.add(body)
    s.add(_bar(0, 24000))
    s.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0))
    normalize_project_to_meters(_project(s))
    assert "clip[Slab:plate]: 2 parts, 0 dropped entirely, 2 trimmed, flush" \
        in capsys.readouterr().err

    s2, body2 = _plate()
    s2.add(body2)
    s2.clip(origin=Point(x=12000, y=0, z=0), normal=(1, 0, 0), overrun=500)
    normalize_project_to_meters(_project(s2))
    assert "overrun=500mm" in capsys.readouterr().err


# ── the deferred miter derivation ─────────────────────────────────────────────

def _corner(extend_before: bool = None):
    """Two perpendicular wall leaves mitered at the origin.

    ``extend_before`` adds a return wing to the south leaf — a part that
    EXTENDS its run, so it moves the assembly's oriented long axis and
    therefore the bisector. ``True`` adds it before the ``miter()`` line,
    ``False`` after, ``None`` not at all.
    """
    from lite_step.models import Wall, miter

    wa, wb = Wall(name="south"), Wall(name="west")
    wa.add(Box(name="body", start=Point(x=0, y=0, z=0),
               end=Point(x=4000, y=300, z=2700)))
    wb.add(Box(name="body", start=Point(x=0, y=0, z=0),
               end=Point(x=300, y=4000, z=2700)))
    wing = lambda: Box(name="wing", start=Point(x=3800, y=300, z=0),
                       end=Point(x=4000, y=3000, z=2700))
    if extend_before is True:
        wa.add(wing())
    miter(wa, wb, at=Point(x=0, y=0, z=0), edge=(0, 0, 1))
    if extend_before is False:
        wa.add(wing())
    return wa, wb


def _derived_normal(wall) -> tuple:
    from lite_step.compiler.composite_clip import derive_pending_miters

    derive_pending_miters(wall)
    return tuple(round(c, 6) for c in wall._clips[0].normal)


def test_the_joint_angle_does_not_depend_on_where_the_miter_line_sits():
    """The acceptance criterion for deferring the derivation, and the reason
    it is worth its cost.

    Derived eagerly, this fixture gave (-0.831592, 0.555388, 0.0) with the wing
    added before the call and (-0.707107, 0.707107, 0.0) with it added after —
    45 degrees against 33.8, from identical source differing only in line
    order, with nothing in the output recording which you got.

    The control matters as much as the pair: a part that stays INSIDE the
    footprint must NOT move the angle, or "order-independent" would just mean
    "ignores its parts".
    """
    before = _derived_normal(_corner(extend_before=True)[0])
    after = _derived_normal(_corner(extend_before=False)[0])
    assert before == after, (
        f"the same joint derived two angles from the same source: {before} "
        f"with the wing added before the miter() line, {after} with it after"
    )
    # ...and the wing genuinely changes the angle, so the equality above is a
    # real agreement rather than both sides ignoring it.
    assert before != _derived_normal(_corner()[0])


def test_both_sides_of_a_joint_derive_from_untrimmed_geometry():
    """Pass 1 derives every joint before pass 2 trims anything.

    Distributing a clip TRIMS geometry, and the angle is derived from BOTH
    sides' footprints — so resolving one element's joint after its partner had
    been trimmed would read a footprint the author never wrote, and which side
    got the pristine reading would come down to tree order. Asserted by
    comparing against each side derived ALONE, with nothing else resolved.
    """
    from lite_step.compiler.composite_clip import resolve

    wa, wb = _corner()
    proj = _project(wa)
    proj.storeys[0].add(wb)
    norm = normalize_project_to_meters(proj)
    south, west = norm.storeys[0].elements[:2]

    # EXACTLY opposite, to the bit — no rounding. Comparing at 6 places
        # passes while the corpus carries a real
        # asymmetry ([0,-1,0] against [0,-1,-9.9e-05]); rounding to 6 places is
    # precisely the width that hides it.
    n_s = tuple(south._elements[0]._clips[0].normal)
    n_w = tuple(west._elements[0]._clips[0].normal)
    assert n_s == tuple(-c for c in n_w), (n_s, n_w)
    resolve(norm)                     # idempotent — no second derivation
    assert len(south._elements[0]._clips) == 1


def test_a_joint_is_derived_exactly_once():
    """The mechanism behind the assertion above, guarded directly.

    Antipodality is a CONSEQUENCE of deriving one plane per joint; asserting
    only the consequence lets a second derivation back in the moment it happens
    to agree. And a second derivation does not merely cost time — it
    tessellates the partner again, by which point the first side has taken its
    clip, so it measures a footprint the author never wrote. Measured on the
    corpus roof: 13 of 26 elements come back with a different vertex count on
    the second call, 8 -> 10, because a clipped box
    has more corners.
    """
    import lite_step.models.elements as E

    calls = []
    original = E.derive_miter_pair
    E.derive_miter_pair = lambda elem, req: (calls.append(1), original(elem, req))[1]
    try:
        wa, wb = _corner()
        proj = _project(wa)
        proj.storeys[0].add(wb)
        normalize_project_to_meters(proj)
    finally:
        E.derive_miter_pair = original
    assert len(calls) == 1, (
        f"one joint, {len(calls)} derivations — the second one reads a "
        f"footprint the first has already clipped"
    )


def test_a_model_reaching_the_generator_with_an_underived_joint_is_refused():
    """The cost of deferring, guarded rather than accepted.

    Skip the normalizer and the joint is simply never derived — the model
    compiles, validates and renders, and the corner is not cut. Nothing in the
    IFC records that a cut was asked for, so no downstream gate can find it.
    This is not hypothetical: it is how three test_displacement_clips cases
    failed while this was being built, and only their volume assertion caught
    it.
    """
    from lite_step.ifc.generator import generate_ifc

    wa, wb = _corner()
    proj = _project(wa)
    proj.storeys[0].add(wb)
    with pytest.raises(ValueError, match="never derived"):
        generate_ifc(proj, source_code=None)     # no normalize_project_to_meters


def test_a_negative_miter_overrun_is_refused_at_the_call():
    """At the CALL, not at the derivation.

    ``derive_miter_clips`` builds its HalfSpace via ``model_construct`` — its
    inputs are derived and the strict int-mm gate refuses the floats a
    derivation produces — so that object's own validators never run on this
    path. Without this check a negative overrun would ride all the way to a
    trim that cannot express it.
    """
    from lite_step.models import Wall, miter

    wa, wb = Wall(name="a"), Wall(name="b")
    with pytest.raises(ValueError, match=r"overrun=-1 must be >= 0"):
        miter(wa, wb, at=Point(x=0, y=0, z=0), edge=(0, 0, 1), overrun=-1)


# ── the composed case: bar in a sleeve, in mitered slabs ──────────────────────

def _sleeved_slab(name, p0, p1, box_end):
    """One concrete slab carrying a plastic sleeve with a bar inside it.

    The sleeve is a ``Pipe`` because the registry has no plastic *bar*: what
    matters for this fixture is that it is a PATH primitive 20 mm larger in
    radius than the bar it sheathes, so the miter must trim it rather than
    subtract it.
    """
    from lite_step.models import Bar, Pipe, Slab

    slab = Slab(name=name)
    slab.add(Box(name="body", material="Concrete_C30-37",
                 start=Point(x=0, y=0, z=0), end=box_end))
    bar = Bar(name="rebar", diameter=BAR_DIA, path=[p0, p1])
    sleeve = Pipe(name="sleeve", radius=SLEEVE_R, path=[p0, p1])
    sleeve.difference(bar)              # the bar carves the plastic
    slab.add(sleeve)
    return slab


BAR_DIA = 16                            # nominal, mm
SLEEVE_R = BAR_DIA // 2 + 20            # 2 cm bigger in RADIUS than the bar


def test_a_bar_in_a_sleeve_inside_mitered_slabs():
    """Everything this change touches, composed, on one model.

    Three mechanisms have to compose without any of them noticing the others:
    the miter distributes to a container's parts; the concrete takes its cut as
    a BOOLEAN; the sleeve takes the same cut as a PATH TRIM. The last two come
    from one half-space — the split is the part-kind, not the call.

    The bar is a consumed OPERAND of the sleeve, so it is reached through its
    consumer and is deliberately NOT trimmed as a part in its own right. That
    is what keeps a trimmed sleeve stub still hollow: trimming the cutter would
    shorten the HOLE rather than the solid.
    """
    from lite_step.models import Pipe, miter

    a = _sleeved_slab("a", Point(x=0, y=200, z=150), Point(x=6000, y=200, z=150),
                      Point(x=6000, y=400, z=300))
    b = _sleeved_slab("b", Point(x=200, y=0, z=150), Point(x=200, y=6000, z=150),
                      Point(x=400, y=6000, z=300))
    miter(a, b, at=Point(x=0, y=0, z=0), edge=(0, 0, 1))

    proj = _project(a)
    proj.storeys[0].add(b)
    norm = normalize_project_to_meters(proj)
    slab_a, slab_b = norm.storeys[0].elements[:2]

    for slab in (slab_a, slab_b):
        parts = {type(p).__name__: p for p in iter_parts(slab)}
        concrete, sleeve = parts["Box"], parts["Pipe"]
        assert len(concrete._clips) == 1, "the concrete takes the miter as a boolean"
        assert not sleeve._clips, (
            "the sleeve took the miter as a BOOLEAN — a path primitive must be "
            "trimmed, or every sleeve costs a CSG level it does not need"
        )
        assert len(sleeve._cuts) == 1, (
            "the bar stopped carving the sleeve — a trimmed stub must still be "
            "hollow, so the cutter is reached through its consumer and never "
            "trimmed as a part of its own"
        )

    # The bisector of +x and +y through the origin is the plane x=y, so a
    # sleeve running along y=200 is cut at x=200 — an exact number, not a
    # tolerance, because the trim interpolates on the path's own segment.
    assert _paths_mm(slab_a) == [[(200, 200, 150), (6000, 200, 150)]]
    assert _paths_mm(slab_b) == [[(200, 200, 150), (200, 6000, 150)]]

    ifc = generate_ifc(norm, source_code=None)
    assert ifc.success, ifc.error
    # TWO half-spaces: one miter clip per concrete body. The sleeves contribute
    # none, which is the point of the trim.
    #
    # This asserted FOUR until the container-miter suppression was fixed. The
    # extra two were the redundant auto-carve's clipped operand: the joint id
    # sat on the container while the displacement pass pairs the leaf SOLIDS,
    # so the pair was never recognised as authored and carved anyway. The
    # number moving is how that reached the emitted file — see
    # test_carve_miter_composition.py.
    assert ifc.ifc_content.count("IFCHALFSPACESOLID") == 2


def test_the_sleeve_is_wider_than_the_bar_it_sheathes():
    """Guards the fixture itself: a sleeve narrower than its bar would make
    every assertion above pass while modelling nothing physical."""
    assert SLEEVE_R - BAR_DIA / 2 == 20
