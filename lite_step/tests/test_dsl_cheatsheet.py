"""The API quick reference in ``agents/dsl-reference.md`` must match the models.

The block was GENERATED until; it is hand-written now. A signature
dump could not say the things that actually trip an author — that ``Sheet`` is a
FUNCTION returning a ``Sweep``, that ``Mesh`` has two mutually exclusive
spellings, that ``corner_min`` is a ``Point2D`` (the omission that cost the
doc-verification reader its only failed compile). It also cost a generator
script, a byte-pin, and a stripper in ``worker/cli.py`` keeping its own
maintainer comment out of the build prompt.

What is pinned, and what deliberately is not:

* **Pinned** — every authorable class is named, both closed vocabularies are
  complete, every method and verb advertised exists and accepts the keywords it
  advertises, and no internals leak.
* **Not pinned** — the prose, types and defaults. The compiler validates every
  constructor call at construction, so a wrong default is a loud failure on the
  first build rather than a silent lie.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "agents" / "dsl-reference.md"


def _block_in_doc() -> str:
    """The fenced API block under the quick-reference heading."""
    doc = DOC.read_text(encoding="utf-8")
    marker = "## 3. API quick reference"
    assert marker in doc, "the API quick-reference heading moved — repoint this"
    tail = doc.split(marker, 1)[1]
    assert "```python" in tail, "no fenced API block under the heading"
    return tail.split("```python", 1)[1].split("```", 1)[0].strip()


@pytest.mark.parametrize("method,keywords", [
    ("clip", ("origin", "normal")),
    ("opening", ("along", "along_center", "up", "inset")),
    ("height_at", ("x", "y", "round")),
    ("raycast", ("origin", "direction", "round")),
    ("heightmap_at", ("corner_min", "corner_max", "rows", "cols", "round")),
])
def test_methods_accept_the_keywords_the_sheet_advertises(method, keywords):
    """The methods half is hand-written, so it needs its own drift guard."""
    from lite_step.models import Box

    fn = getattr(Box, method, None)
    assert fn is not None, f"{method} vanished from the element API"
    params = inspect.signature(fn).parameters
    for kw in keywords:
        assert kw in params, f"the cheat sheet advertises {method}(..., {kw}=) — it does not exist"


def test_project_methods_the_sheet_advertises_exist():
    """Same drift guard, for the hand-written lines that are NOT on an
    element — the parametrized case above only ever looks at ``Box``."""
    from lite_step.models import Project

    fn = getattr(Project, "assert_carved", None)
    assert fn is not None, "assert_carved vanished from the Project API"
    params = inspect.signature(fn).parameters
    assert "target" in params and "by" in params
    assert params["by"].kind is inspect.Parameter.KEYWORD_ONLY, (
        "the cheat sheet advertises `by=` as keyword-only"
    )


@pytest.mark.parametrize("method", ["aggregate", "zone"])
def test_grouping_verbs_the_sheet_advertises_exist(method):
    """Same drift guard for the grouping lines — `name` positional,
    `members=` keyword-only, exactly as the sheet spells them."""
    from lite_step.models import Project

    fn = getattr(Project, method, None)
    assert fn is not None, f"{method} vanished from the Project API"
    params = inspect.signature(fn).parameters
    assert "name" in params and "members" in params
    assert params["members"].kind is inspect.Parameter.KEYWORD_ONLY, (
        f"the cheat sheet advertises {method}(name, members=) with members keyword-only"
    )


def test_product_lines_the_sheet_advertises_exist():
    """Product landed after the generator (WS-B) and was missing from the
    sheet entirely — the block claims 'if a constructor is not here, it
    does not exist', so absence is the lie this guards against."""
    from lite_step.models import Product

    block = _block_in_doc()
    assert "Product(source, *, name)" in block, "the Product line left the sheet"
    assert "occurrence" in block, "the occurrence line left the sheet"
    params = inspect.signature(Product).parameters
    assert "source" in params and "name" in params
    assert params["name"].kind is inspect.Parameter.KEYWORD_ONLY
    occ = inspect.signature(Product.occurrence).parameters
    assert "name" in occ and occ["name"].kind is inspect.Parameter.KEYWORD_ONLY


def test_cheatsheet_does_not_leak_internals():
    """`id=`, `storeys=` and module paths must never reach the agent."""
    block = _block_in_doc()
    for leak in ("id: str", "storeys", "sites:", "PydanticUndefined",
                 "lite_step.models", "collections.abc", "NoneType"):
        assert leak not in block, f"internal leaked into the cheat sheet: {leak!r}"


# ── the closed vocabularies ────────────────────────────────────────


def _baseline_materials() -> dict:
    """The DK registry as `dk.py` DEFINES it, not as the process currently holds it.

    ``MATERIALS`` is an extensible baseline, not a closed set: the documented
    ``create-material`` rule has models register products inline
    (``dk.MATERIALS[key] = MaterialDef(...)``), and the bridge skill does exactly
    that for ``Steel_S355``/``Asphalt_Pavement``/``RoadMarking_White``. So once any
    test compiles that example, the global registry carries keys the reference was
    never meant to list, and reading the live global made this test fail depending
    on which OTHER tests had run first.

    Loading the module under a private name gives a pristine copy without
    disturbing the one everything else is holding.
    """
    import importlib.util
    from pathlib import Path

    src = Path(__file__).resolve().parents[2] / "lite_step" / "materials" / "dk.py"
    spec = importlib.util.spec_from_file_location("_dk_baseline_for_test", src)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.MATERIALS


def test_every_material_key_is_listed():
    """The hand-written list had ALREADY drifted when this was added.

    ``Cavity_Ventilated`` and ``Cavity_Unventilated`` were missing from the
    doc's "DK registry keys:" sentence while the LayerSet section required
    them by name ("a ventilation gap is a registry material, not an authored
    flag"). The doc told the agent to use keys it did not list.
    """
    block = _block_in_doc()
    missing = [k for k in _baseline_materials() if k not in block]
    assert not missing, f"material keys absent from the cheat sheet: {missing}"


def test_every_whitelisted_ifc_class_is_listed():
    """A truncated whitelist is the expensive kind of incomplete.

    Prose ending with "..." means an unlisted class is learned by failing at
    construction, i.e. by burning a build. 37 names cost ~510 B.
    """
    from lite_step.models.elements import ELEMENT_IFC_CLASS_WHITELIST

    block = _block_in_doc()
    missing = [c for c in ELEMENT_IFC_CLASS_WHITELIST if c not in block]
    assert not missing, f"ifc classes absent from the cheat sheet: {missing}"


def test_the_reference_documents_what_ships_not_what_a_model_registered():
    """A model must not be able to write itself into the DSL reference.

    `MATERIALS` is an extensible baseline — the documented `create-material`
    rule has models register products inline, and the bridge skill registers
    Steel_S355 / Asphalt_Pavement / RoadMarking_White exactly that way. Both
    the generator and the earlier version of these tests read the LIVE global,
    so once any test in the same process compiled that example, the cheat
    sheet regenerated with those keys in it and the doc comparison failed —
    a green or red result depending purely on test ORDER.

    Registering a material here proves the baseline load is immune.
    """
    from lite_step.materials import dk

    # Clone a real entry rather than construct one — MaterialDef requires
    # render/geometry and this test is about the registry, not the model.
    sentinel = "Unobtainium_Test_Only"
    dk.MATERIALS[sentinel] = next(iter(dk.MATERIALS.values()))
    try:
        assert sentinel in dk.MATERIALS, "sentinel did not register — test is a no-op"
        assert sentinel not in _block_in_doc(), (
            "a runtime-registered material reached the reference — the doc "
            "lists the shipped baseline, not the live registry"
        )
        assert sentinel not in _baseline_materials()
    finally:
        dk.MATERIALS.pop(sentinel, None)


def test_every_authorable_model_class_is_in_the_cheat_sheet():
    """An unlisted constructor is an UNREACHABLE one.

    The reference says it outright, two lines above the block: *"Generated
    from the live models — if a constructor is not here, it does not exist."*
    The author reads that literally, so a class absent from the sheet is a class she
    will not use however well it works.

    That is not hypothetical. ``SpatialElement`` shipped implemented, tested
    and emitting conformant IFC — and absent from here, because the generator
    keeps a HAND-MAINTAINED name list and nothing compared it to what the
    package exports. Same failure as the LOD350 skill index: the
    thing worked and could not be reached.

    Asserted against the DOC rather than the generator's dict on purpose —
    the doc is what is inlined into the build prompt, so it is the artefact
    whose completeness matters.
    """
    from pydantic import BaseModel

    import lite_step.models as models

    block = _block_in_doc()

    # Exempt, and each for a stated reason — an exemption set without reasons
    # is where the next unreachable class hides.
    NOT_AUTHORABLE = {
        # The abstract base every container inherits. Not a thing in its own
        # right, so its absence is correct.
        "BimElement",
        # A frozen value object the compiler builds inside ``.clip(origin=,
        # normal=)`` (elements.py: ``self._clips.append(HalfSpace(...))``).
        # An author never writes ``HalfSpace(...)``; the verb they DO write,
        # ``e.clip()``, is listed. Exported for typing, not for authoring.
        "HalfSpace",
    }

    exported = {
        name for name in models.__all__
        if isinstance(getattr(models, name, None), type)
        and issubclass(getattr(models, name), BaseModel)
    }
    missing = sorted(exported - NOT_AUTHORABLE - {n for n in exported if n in block})
    assert not missing, (
        f"model class(es) exported but absent from the cheat sheet: {missing}.\n"
        f"the author is told a constructor that is not listed does not exist, so "
        f"these are unreachable. Add them to the API block in "
        f"agents/dsl-reference.md.")


# ── the OTHER direction: a real keyword the sheet never mentions ─────────────

def test_overrun_is_advertised_on_BOTH_verbs_that_take_it():
    """``overrun=`` is ONE concept spanning two verbs, so advertising it on one
    is the same as not advertising it.

    ``test_methods_accept_the_keywords_the_sheet_advertises`` above checks one
    direction only — that nothing on the sheet is fictional. The reverse is not
    generally checkable (the sheet is a curated cheat sheet, not an API dump,
    and plenty of real kwargs are deliberately left off it), but it IS checkable
    for a parameter that exists on two verbs for one reason: an author who reads
    it on ``.clip()`` and not on ``miter()`` concludes miters cannot lap.

    Measured when this landed: the engine had shipped ``miter(…, overrun=0)``
    and the sheet advertised it on ``.clip()`` alone. Nothing failed, and the
    parameter was unreachable to anyone reading the reference — which is every
    build agent.
    """
    from lite_step.models import Box, miter

    for fn, label in ((Box.clip, ".clip()"), (miter, "miter()")):
        assert "overrun" in inspect.signature(fn).parameters, (
            f"{label} lost overrun= — the sheet still advertises it"
        )

    sheet = _block_in_doc()
    for line_start, label in (("e.clip(", ".clip()"), ("miter(a, b", "miter()")):
        line = next((l for l in sheet.splitlines() if l.startswith(line_start)), None)
        assert line is not None, f"the sheet has no {label} signature line"
        assert "overrun" in line, (
            f"{label} takes overrun= and its cheat-sheet line does not say so: {line!r}. "
            f"One concept advertised on one of its two verbs is not advertised."
        )


# ── prose claims outside the fenced block ───────────────────────────────────
#
# ``_block_in_doc`` reads the API block only. The three claims below live in
# the prose, and each one exists because an author got it wrong: a positional
# single-argument constructor, a profile authored in world coordinates, and a
# heightmap given both base spellings. Prose is not pinned in general (the
# module docstring says so) — these are pinned because each states a BEHAVIOUR,
# so the behaviour changing must redden the doc rather than silently outdate it.


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.mark.parametrize("factory,keyword", [
    ("Element", "ifc_class"),
    ("Material", "key"),
    ("Space", "name"),
])
def test_single_argument_constructors_are_keyword_only(factory, keyword):
    """Rule 1 is general, but the TypeError names no parameter.

    ``BaseModel.__init__() takes 1 positional argument but 2 were given`` is
    what a positional call actually says — nothing in it points back at rule 1,
    which is why the one-argument constructors keep being written positionally.
    The doc spells each keyword; this proves the spelling is the real one and
    that the positional form is still refused.
    """
    import lite_step.models as models

    cls = getattr(models, factory)
    with pytest.raises(TypeError):
        cls("a-positional-argument")
    assert keyword in inspect.signature(cls).parameters, (
        f"the reference tells the author to write {factory}({keyword}=...) — "
        f"that parameter does not exist"
    )
    assert f'{factory}({keyword}=' in _doc(), (
        f"the keyword-only note lost its {factory} spelling"
    )


def test_heightmap_takes_exactly_one_base_plane():
    """`depth=` and `bottom_z=` are alternatives; listing both on one line
        reads as though both were required."""
    from lite_step.models import Mesh, Point2D

    flat = dict(heightmap=[[0, 0], [0, 0]],
                corner_min=Point2D(x=0, y=0), corner_max=Point2D(x=1000, y=1000))
    with pytest.raises(ValueError):
        Mesh(**flat, depth=500, bottom_z=-500)      # both
    with pytest.raises(ValueError):
        Mesh(**flat)                                # neither
    Mesh(**flat, depth=500)                         # relative — fine
    Mesh(**flat, bottom_z=-500)                     # absolute — fine
    with pytest.raises(ValueError):
        Mesh(**flat, bottom_z=500)                  # not below the lowest height

    doc = _doc()
    assert "exactly ONE of depth= / bottom_z=" in doc, (
        "the API block stopped saying the two base spellings are exclusive"
    )
    assert "strictly BELOW the lowest height" in doc, (
        "the doc stopped saying bottom_z must clear the terrain underneath — "
        "without it the author reads bottom_z as a trim plane"
    )


def test_revolve_profile_is_radial_and_the_doc_says_so():
    """A profile in world coordinates compiles into empty space.

    ``Revolve`` is the one case the compiler CAN catch, because a radial
    distance below zero is meaningless. ``Sweep`` cannot be caught — every
    offset is legal — so the doc carries that half.
    """
    from lite_step.models import Point, Point2D, Revolve

    axis = [Point(x=0, y=0, z=0), Point(x=0, y=0, z=1000)]
    with pytest.raises(ValueError, match="radial"):
        Revolve(profile=[Point2D(x=-1, y=0), Point2D(x=1000, y=0),
                         Point2D(x=1000, y=1000)], path=axis)

    doc = _doc()
    assert "the 2-point AXIS" in doc, "the doc stopped saying Revolve's path is an axis"
    assert "RADIAL distance" in doc, "the doc stopped naming the Revolve profile frame"


def test_reversing_a_sweep_path_mirrors_its_profile():
    """The claim behind "centre the profile": the across-axis reverses.

    Measured on a real failure — a 7 x 7 m house whose two roof slopes were
    swept from opposite eaves to one ridge with the SAME profile, authored
    ``x=0..7600`` (world extents) on a path at the roof's left verge. One
    slope landed on the building and the other 7.6 m beside it, so the render
    read as "half the roof is misplaced" when in fact BOTH halves were
    authored wrong and one was correct by accident.

    Nothing here can be validated at construction — every profile offset is a
    legal offset — so the reference carries the rule and this pins the
    behaviour the rule describes.
    """
    from lite_step.models import Element, Point, Point2D, Project, Storey, Sweep

    HALF, EAVE_Z, RIDGE_Z = 3800, 2550, 5195

    def _span(profile_x0, profile_x1, path_x):
        proj = Project(name="roof_sign")
        storey = Storey(name="ground", elevation=0)
        proj.add_storey(storey)
        spans = []
        for name, eave_y in (("south", -3800), ("north", 3800)):
            slope = Element(ifc_class="IfcCovering", predefined_type="ROOFING",
                            name=f"slope_{name}").add(
                Sweep(path=[Point(x=path_x, y=eave_y, z=EAVE_Z),
                            Point(x=path_x, y=0, z=RIDGE_Z)],
                      profile=[Point2D(x=profile_x0, y=0), Point2D(x=profile_x1, y=0),
                               Point2D(x=profile_x1, y=40), Point2D(x=profile_x0, y=40)],
                      color="roof"))
            storey.add(slope, carve="none")
            b = slope.world_aabb()
            spans.append((b.min.x, b.max.x))
        return spans

    # As authored in the failure: profile 0..7600, path on the left verge.
    south, north = _span(0, 2 * HALF, -HALF)
    assert south == pytest.approx((-3 * HALF, -HALF)), (
        "the mirrored slope stopped landing off the building — if the "
        "across-axis does not reverse with the path, the reference's "
        "reason for centring a profile is gone"
    )
    assert north == pytest.approx((-HALF, HALF)), (
        "the other slope stopped landing correctly-by-accident"
    )
    assert south != north, "both slopes cannot land in the same place"

    # The remedy the reference gives: profile centred, path up the middle.
    south, north = _span(-HALF, HALF, 0)
    assert south == north == pytest.approx((-HALF, HALF)), (
        "a centred profile on a centred path does not put both slopes on "
        "the building — the reference tells the author it does"
    )

    doc = _doc()
    assert "up the MIDDLE of what it sweeps" in doc, (
        "the doc stopped saying the path is the middle, not an edge"
    )
    assert "REVERSES with the path" in doc, (
        "the doc stopped saying the across-axis reverses — which is the "
        "reason centring the profile is the fix, not a style preference"
    )
