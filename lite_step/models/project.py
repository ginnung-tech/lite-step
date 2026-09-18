"""
Lite-STEP Project - the root authoring container + storeys.

Hierarchy:
  Project (the root; maps to IfcProject → IfcSite → IfcBuilding)
    ├── Site (sites — terrain / site-context, under IfcSite)
    └── Storey (multiple — under IfcBuilding)
          └── Elements (Wall, Box, ...)
"""

import copy

from pydantic import BaseModel, Field, PrivateAttr, field_validator
from typing import Any, List, Union, Optional
from uuid import uuid4

from .primitives import _reject_float_when_strict
from .elements import (
    reject_opening_children,
    Wall, Site,
    Box, Extrude, Sweep, Pipe, Revolve, Bar, Mesh, Element,
    # Semantic element classes for architecture sketching
    Column, Beam, Slab, Roof,
    # Planning primitives
    ReferencePoint, GuideLine, Space,
    # DSL v2.1 leaf-name grammar (shared with Storey — a storey name is a leaf)
    _LEAF_NAME_RE,
)


# Union type for all element types
ElementType = Union[
    Wall, Site, Box, Extrude, Sweep, Pipe, Revolve, Bar, Element,
    Mesh, Column, Beam, Slab, Roof,
    ReferencePoint, GuideLine, Space,
]


def _reject_storey_as_element(elements: tuple) -> None:
    """Refuse a ``Storey`` handed to an element-taking ``.add()``.

    ``proj.add(Storey(name="first", elevation=3000))`` must not be accepted
    in silence: a plain ``Project.add`` files the storey into the DEFAULT
    storey's
    ``elements`` list — a storey as a child element of another storey —
    and the model died much later, deep inside emission, with::

        AttributeError: 'Storey' object has no attribute 'ifc_name'

    an error naming neither the mistake nor the fix. ``.add()`` is the
    universal containment verb everywhere else in the DSL and ``add_storey``
    is one keystroke away, so reaching for the wrong one is the ordinary
    thing to do, not a careless thing to do. Per the roadmap's §5 process
    note: *a rule an author must be surprised by once to learn is a defect
    in the rule, not in the author.*

    Raised at the ``.add()`` call site, so the traceback points at the
    authoring line — the same contract as ``_validate_container_children``
    (which already refuses this on a ``Wall``/``Element``, naming
    ``.anchor()``; ``Project`` and ``Storey`` were the two containers with
    no table at all, which is why they let it through).

    Checked BEFORE anything is appended, so a rejected ``.add()`` never
    partially mutates the container.
    """
    for elem in elements:
        if not isinstance(elem, Storey):
            continue
        label = elem.name or elem.id
        raise ValueError(
            f"Storey {label!r} was passed to .add(), which takes ELEMENTS. A "
            f"storey is a container OF elements, not one of them — adding it "
            f"here files it under another storey, and the model fails much "
            f"later with \"'Storey' object has no attribute 'ifc_name'\". "
            f"Use proj.add_storey(Storey(name={label!r}, ...)) to add the "
            f"storey, then proj.storeys[n].add(...) — or the storey's own "
            f".add(...) — to put elements on it."
        )


class Storey(BaseModel):
    """
    Building storey at a specific elevation.

    At script execution time: integer elevation in millimeters
    After compilation: float elevation in meters (normalized by executor)

    Elements added to a storey are placed relative to the storey's elevation.

    DSL v2.1: ``name=`` is an optional lowercase LEAF, exactly like an element
    — a NAMED storey contributes a ``storey:<leaf>`` segment to every canonical
    path under it (``wall:north:storey:ground``); an ANONYMOUS storey
    (``name=None``, the default) contributes nothing (``wall:north``). The
    single-storey common case is therefore anonymous and segment-free. When a
    project has more than one storey the compiler requires every storey to be
    named (anonymous storeys would be indistinguishable scopes). ``Project``
    is NOT a segment — canonical paths stop at the storey.
    """
    id: str = Field(default_factory=lambda: f"storey_{uuid4().hex[:8]}")
    name: Optional[str] = None
    elevation: float = 0.0  # Elevation from building origin (mm at script time, m after compilation)
    elements: List[ElementType] = Field(default_factory=list)

    #: DSL v2.1 canonical name (``storey:<leaf>`` when named, ``None`` when
    #: anonymous), stamped by ``lite_step.compiler.naming.stamp_canonical_names``.
    #: Read by the emitter for the IfcBuildingStorey Name.
    _canonical_name: Optional[str] = PrivateAttr(default=None)

    #: DSL v1.5 RULE 2 — ``elevation`` is a mm dimension like every other
    #: geometry value, so a float is refused at construction under strict
    #: int-mm (``Storey(elevation=I(4207.5/3))``, not ``1402.5``). It was the
    #: one hole in RULE 2's "every mm dimension" claim: typed ``float`` with
    #: no gate, it accepted sub-mm drift silently and the executor divided it
    #: straight to metres. The meters-normalizer ASSIGNS
    #: ``storey.elevation = elevation / 1000.0`` and pydantic does not
    #: validate on assignment here (``validate_assignment`` is off), so the
    #: normalized float never meets this gate — same contract as ``Point``.
    _strict_int_mm = field_validator("elevation", mode="before")(
        _reject_float_when_strict
    )

    @field_validator("name", mode="after")
    @classmethod
    def _validate_leaf_name(cls, v: Optional[str]) -> Optional[str]:
        """A storey name is a leaf (``[a-z0-9_]+``) or None — same grammar as
        an element name (DSL v2.1)."""
        if v is None:
            return None
        if not _LEAF_NAME_RE.match(v):
            raise ValueError(
                f"Storey name={v!r} is not a valid leaf — DSL v2.1 storey names "
                f"are a single lowercase word matching [a-z0-9_]+ (no uppercase, "
                f"':', '.', or whitespace), e.g. Storey(name=\"ground\"). Leave "
                f"it off for an anonymous single storey."
            )
        return v

    def add(self, *elements: ElementType, carve: str = None) -> None:
        """Add one or more elements to this storey.

        ``carve`` names who loses material in the overlaps this add creates —
        ``"other"`` (default), ``"self"``, ``"none"``. See
        ``elements._add_children``. A storey is where most top-level solids
        meet each other, so this is the container the argument matters most
        on; omitting it here would have left the feature unreachable for the
        common case.

        Varargs to match the semantic containers (Wall/Roof/... all take
        ``*elements``) and the documented contract "all ``.add()`` methods
        accept varargs" — ``storey.add(w1, w2)`` / ``storey.add(*walls)``.

        Refuses a ``Storey`` — see
        :func:`_reject_storey_as_element`. This is the one container whose
        ``.add()`` has no type table (the ``BimElement`` containers refuse a
        ``Storey`` already, via ``_validate_container_children``), so a
        storey filed here as an element was accepted in silence and died
        inside emission.

        Refuses a ``Window``/``Door`` too, through the SAME
        :func:`elements.reject_opening_children` every other container calls —
        having no type table is why this one missed it. The message already
        said an opening "cannot be ADDED", so the rule was stated more widely
        than it was enforced. The compile-time backstop
        (``executor._opening_placement_errors``) DID catch it, so nothing
        misbuilt; what changes is that the author learns it on the line they
        wrote rather than at the end of a compile.
        """
        from .elements import _bump_generation

        from .elements import CARVE_DIRECTIONS, _stamp_carve

        if carve is not None and carve not in CARVE_DIRECTIONS:
            raise ValueError(
                f"add(carve={carve!r}) is not a direction — use "
                f"{', '.join(repr(d) for d in CARVE_DIRECTIONS)}."
            )
        _reject_storey_as_element(elements)
        reject_opening_children(self, elements)
        for elem in elements:
            elem._parent = self
        if carve is not None:
            for elem in elements:
                _stamp_carve(elem, carve)
        self.elements.extend(elements)
        _bump_generation()          # a world read taken before this is stale

    def __deepcopy__(self, memo=None):
        """Deep copy without following ``_parent`` — see
        ``BimElement.__deepcopy__``. A Storey's pointer reaches the Project
        and through it every other storey, so copying one storey would copy
        the whole model."""
        cls = type(self)
        new = cls.__new__(cls)
        if memo is not None:
            memo[id(self)] = new
        object.__setattr__(new, "__dict__", copy.deepcopy(self.__dict__, memo))
        object.__setattr__(new, "__pydantic_extra__",
                           copy.deepcopy(self.__pydantic_extra__, memo))
        object.__setattr__(new, "__pydantic_fields_set__",
                           copy.copy(self.__pydantic_fields_set__))
        private = getattr(self, "__pydantic_private__", None)
        if private is None:
            object.__setattr__(new, "__pydantic_private__", None)
        else:
            object.__setattr__(new, "__pydantic_private__", {
                k: (None if k == "_parent" else copy.deepcopy(v, memo))
                for k, v in private.items()
            })
        return new

    def __eq__(self, other: object) -> bool:
        """Value equality, ignoring ``_parent`` — see
        ``BimElement.__eq__`` for why this override is mandatory rather than
        cosmetic. A Storey pointing at its Project makes pydantic's equality
        walk Project -> storeys -> this Storey -> Project, without end."""
        if other.__class__ is not self.__class__:
            return NotImplemented
        if self.__dict__ != other.__dict__:
            return False
        mine = self.__pydantic_private__ or {}
        theirs = other.__pydantic_private__ or {}
        if mine.keys() != theirs.keys():
            return False
        return all(v == theirs[k] for k, v in mine.items() if k != "_parent")

    @property
    def parent(self) -> Optional["Project"]:
        """The :class:`Project` this storey belongs to, or ``None``.

        ``Storey`` is not a :class:`~lite_step.models.elements.BimElement`, so
        it carries its own copy of the back-reference rather than inheriting
        one. Same contract: walking ``.parent`` from any attached element
        reaches the ``Project``, and a chain that stops short means the
        element's world position is not decided yet.
        """
        return self._parent

    #: The Project this storey was added to. See ``BimElement._parent``.
    _parent: Optional[Any] = PrivateAttr(default=None)

    def authored_aabb(self, divisor: float = 1.0):
        """Extent of everything on this storey — see
        ``BimElement.authored_aabb``.

        ``Storey`` is not a ``BimElement`` (it is a plain ``BaseModel``), so
        this is a small duplicate rather than an inherited method. Worth it:
        "how big is this floor" is a question authors ask, and the alternative
        is every caller writing its own union over ``storey.elements`` — which
        is exactly the second, subtly different AABB this design exists to
        prevent.
        """
        from lite_step.models.bounds import Bounds, BoundsError
        from lite_step.compiler.extent import element_extent, _union

        box = _union(element_extent(e, divisor) for e in self.elements)
        if box is None:
            raise BoundsError(
                f"Storey '{self.name or self.id}' has no extent to report — it "
                f"holds no geometry."
            )
        return Bounds.from_aabb(box, divisor=divisor)

    class Config:
        extra = "forbid"


class Project(BaseModel):
    """
    The root authoring container — storeys, sites, and elements.

    Maps to the IFC hierarchy IfcProject → IfcSite → IfcBuilding →
    IfcBuildingStorey. ``.name`` is the IfcProject/IfcBuilding Name and the
    manifest root key.

    Usage in LLM scripts:
        proj = Project(name="My House")
        proj.add(Wall(...))  # Adds to default storey
        proj.add(Site(...))  # Routed to sites (under IfcSite), not a storey

        # Or with explicit (named) storeys — multi-storey requires names:
        proj.add_storey(Storey(name="ground", elevation=0))
        proj.add_storey(Storey(name="first", elevation=3000))
        proj.storeys[0].add(Wall(...))
    """
    name: str = "Project"
    storeys: List[Storey] = Field(default_factory=list)

    # DSL Site containers (terrain / site-context). A ``Site`` maps to the
    # real ``IfcSite`` and its children live under it — NOT inside a storey.
    # Routed here by ``add()`` so a site-only model (terrain, no building
    # elements) carries no phantom IfcBuilding/IfcBuildingStorey.
    sites: List[Site] = Field(default_factory=list)

    # Georeferencing (optional — set by site-context, not by LLM scripts).
    # Maps the local model origin (0,0,0) to a real-world position.
    site_longitude: Optional[float] = None   # WGS84 decimal degrees (bbox center)
    site_latitude: Optional[float] = None    # WGS84 decimal degrees (bbox center)
    site_elevation: Optional[float] = None   # meters above sea level (geoid)
    site_true_north: Optional[float] = None  # degrees from local Y-axis to true north (0 = Y points north)

    #: Idempotency guard for the geometric-displacement pass
    #: (``lite_step.compiler.displacement.apply_displacement``) — the pass
    #: appends occupant cut-operands to hosts and rewrites terrain meshes, which
    #: is NOT idempotent, so it must run at most once per project.
    _displacement_applied: bool = PrivateAttr(default=False)

    #: Visit count per object ``id()``, a by-product of
    #: ``lite_step.compiler.naming.stamp_canonical_names``'s walk. Any value
    #: above 1 means one object is attached in more than one place, which the
    #: compiler cannot represent (the canonical name, frame and anchor are
    #: single-valued per object). Read by ``naming.find_shared_elements``.
    #: Refreshed on every stamp, so it never outlives the tree it describes.
    #:
    #: ``None`` until stamped — NOT an empty dict. The two are different
    #: answers ("nobody has looked" vs "looked, found nothing"), and a default
    #: of ``{}`` would let an unstamped project report a clean bill of health.
    _visit_counts: Optional[dict] = PrivateAttr(default=None)

    #: ``(label, generation)`` per world query answered against this project.
    #: Compared with the final mutation generation at compile so a coordinate
    #: read before its element moved becomes a warning rather than a silently
    #: wrong building. See ``elements._MUTATION_GENERATION``.
    _world_queries: list = PrivateAttr(default_factory=list)

    #: Measurement dependency graph: ``{element.id: {element.id, ...}}``, an
    #: edge A→B meaning "A was placed from a measurement of B". Built at
    #: AUTHORING time, because by the time placements resolve the dependency
    #: is gone — ``a.placement`` is a literal ``Transform`` carrying no
    #: provenance back to what it was measured from. A cycle here is refused
    #: on the spot; see ``walkguard.MeasurementCycleError``.
    _measure_edges: dict = PrivateAttr(default_factory=dict)

    #: ``element.id → element`` for every node named in ``_measure_edges``,
    #: so a refusal can print ``mesh:mesh1 → mesh:mesh3`` instead of hashes.
    #: Holding the reference also keeps ids stable for the graph's lifetime.
    _measure_nodes: dict = PrivateAttr(default_factory=dict)

    #: World reads answered since the last placement assignment. Drained into
    #: edges by ``BimElement.__setattr__`` when a placement lands: a world read
    #: taken while positioning an element IS a dependency on it.
    #:
    #: Deliberately not conditioned on whether the value reached the result.
    #: Deciding that would mean evaluating author arithmetic to choose whether
    #: to refuse — the compiler guessing at intent, which is the thing this
    #: feature exists to stop. A read multiplied away is still a read, and the
    #: ring it closes is still refused.
    _pending_reads: list = PrivateAttr(default_factory=list)

    #: ``(loser, winner, mechanism)`` per carve this compile realised — written
    #: by ``displacement.apply_displacement``, read by the compile-time carve
    #: report and by :meth:`assert_carved`.
    _carve_pairs: list = PrivateAttr(default_factory=list)

    #: ``(target_fragment, cutter_fragment)`` per :meth:`assert_carved` call,
    #: collected at AUTHOR time and checked after the displacement pass.
    _carve_assertions: list = PrivateAttr(default_factory=list)

    #: ``(kind, name, member_fragments)`` per :meth:`aggregate` / :meth:`zone`
    #: call, collected at AUTHOR time and resolved at compile by
    #: ``lite_step.ifc.groupings.collect_groupings``. Plain strings, so
    #: ``normalize_project_to_meters``'s deepcopy carries them verbatim and
    #: there is nothing here for the mm -> m pass to miss.
    _groupings: list = PrivateAttr(default_factory=list)

    def add(self, *elements: ElementType, carve: str = None) -> None:
        """Add one or more elements. Varargs — ``proj.add(south, north)`` —
        matching the documented "all ``.add()`` methods accept varargs"
        contract.

        A ``Site`` is routed to ``self.sites`` (it maps to the real
        ``IfcSite``, not a storey). Every other element goes to the default
        (first) storey, which is created ONLY when there is at least one
        non-Site element — so ``proj.add(site)`` on its own leaves
        ``self.storeys == []`` and the model compiles to a building-less,
        site-only IFC (no phantom IfcBuilding/IfcBuildingStorey).

        DSL v2.1: the auto-created storey is ANONYMOUS (``name=None``) so the
        single-storey common case yields segment-free canonical paths
        (``wall:north``, not ``wall:north:storey:...``).

        A ``Storey`` is REFUSED here, naming ``add_storey()`` —
        see :func:`_reject_storey_as_element`. Checked before the default
        storey is auto-created, so a rejected call leaves the project
        exactly as it was rather than minting a phantom storey on the way
        to an error.

        A ``Window``/``Door`` is refused here too, and for the
        second half of that same reason. The forward to
        ``self.storeys[0].add(...)`` would reach the refusal anyway now that
        ``Storey.add`` carries it — but only AFTER this method has minted the
        default storey, so the check has to be on this side of that line as
        well. The message would also name ``Storey.add()`` rather than the
        method the author called.
        """
        _reject_storey_as_element(elements)
        reject_opening_children(self, elements)
        non_site = []
        for elem in elements:
            if isinstance(elem, Site):
                elem._parent = self
                self.sites.append(elem)
            else:
                non_site.append(elem)
        if non_site:
            if not self.storeys:
                self.add_storey(Storey(elevation=0))
            # Forwarded, not dropped: this is the wrapper trap the spec names
            # for skill helpers like place(), and it would be the same silent
            # loss of an exemption here.
            self.storeys[0].add(*non_site, carve=carve)

    def add_storey(self, storey: Storey) -> None:
        """Add a storey to the project."""
        storey._parent = self
        self.storeys.append(storey)

    def assert_carved(self, target: str, *, by: str) -> "Project":
        """State that ``by`` carves ``target``, and FAIL THE COMPILE if it does
        not.

            proj.add(roof)          # the roof frame stands first…
            proj.add(eave_wall)     # …so the wall birdsmouths each rafter foot
            proj.assert_carved("sweep:rafter_s_0", by="wall:eave_south")

        Displacement direction is decided by AUTHORING ORDER — the later-added
        solid carves what already stands — and that is a feature: the assembly
        tail of a model file reads as a construction narrative. What it is not
        is visible. Reordering two ``proj.add()`` lines silently changes the
        geometry, and nothing in a diff, a type, or a test could see it. This
        is the statement that can.

        It authors nothing. The displacement pass still INFERS the carve; this
        pins what was inferred, exactly as a test pins behaviour.

        **The contract is "this PAIR carves" — an observable outcome — and
        never "this MECHANISM carved it."** Do not read it as a claim about
        inference: a pair satisfies it whether the subtraction came from the
        inferred carve, from ``.difference()``, from ``.void()`` or from
        ``.opening()``. That is deliberate and load-bearing. A later
        workstream moves the exemption boundaries between those mechanisms
        (roadmap §6.5), and every assertion written today has to survive it —
        which it does only while the assertion is about the outcome. Anything
        that would make this method distinguish mechanisms breaks every
        assertion in the corpus at once, and must not be added.

        Both arguments are canonical-name fragments and resolve exactly like
        ``Anchor(host=)`` — whole ``type:leaf`` pairs from the tail, so
        ``wall:north`` matches both ``wall:north`` and ``box:body:wall:north``.
        A miss and an ambiguity are both compile errors, for the same reason
        they are there: an assertion that silently matched nothing would be
        worse than no assertion at all.

        Returns ``self``, so assertions chain under the ``proj.add(...)`` tail
        they document.
        """
        self._carve_assertions.append((target, by))
        return self

    def aggregate(self, name: str, *, members: List[str]) -> "Project":
        """Group elements ALREADY in the tree under one named parent.

            proj.aggregate("north_facade",
                           members=["wall:north:storey:ground",
                                    "wall:north:storey:first",
                                    "wall:gable_end"])

        Emits ``IfcRelAggregates`` with a geometry-less parent of the members'
        own IFC class — ``IfcWall = IfcWall + IfcWall``, the conventional
        stacked-wall export. Walls are authored PER STOREY; the facade is the
        aggregate (roadmap §3, the facade inversion).

        **Members are canonical-name STRINGS, and are never contained.** An
        object passed positionally to ``.add()``/``.anchor()`` is containment
        and stamps the child's canonical name; a string in a keyword is a
        reference into the tree and stamps nothing (§V.7). Containing the
        members would rewrite ``wall:north:storey:ground`` into a path under
        the facade — churning the manifest key and the patch atom — which is
        the exact identity churn the inversion exists to avoid. So the
        aggregate is a STATEMENT about the model, like
        :meth:`assert_carved`, not a container in it.

        The parent's IFC class is **derived** from the members and mixed
        classes are a compile error: an author cannot state a class that
        contradicts the members, and the failure is loud rather than a
        silently mis-typed parent. Fragments resolve exactly as
        ``Anchor(host=)`` does — whole ``type:leaf`` pairs from the tail —
        and a miss or an ambiguity is a compile error there for the same
        reason it is here.

        Geometry stays on the members. The parent carries none, or quantities
        double-count and a viewer draws the facade twice.

        Returns ``self``, so statements chain under the ``proj.add(...)`` tail
        they describe.
        """
        from lite_step.ifc.groupings import (
            AGGREGATE, validate_grouping_name, validate_members,
        )

        key = validate_grouping_name(name, AGGREGATE)
        self._groupings.append(
            (AGGREGATE, key, validate_members(members, AGGREGATE, key)))
        return self

    def zone(self, name: str, *, members: List[str]) -> "Project":
        """Group ``Space``s ALREADY in the tree into a named zone.

            proj.zone("thermal_north", members=["space:kitchen",
                                                "space:living"])

        Emits ``IfcZone`` + ``IfcRelAssignsToGroup``. Same shape as
        :meth:`aggregate` — a named, geometry-less grouping of tree members
        cited by canonical-name string, never contained — because §V.7
        settled that they ARE the same shape; the only differences are the
        emitted relation and the rule that a zone groups ``IfcSpace``s only.
        A non-Space member is a loud refusal: ``IfcZone`` is defined over
        spaces, and a zone of walls is a different relation rather than a
        lenient reading of this one.

        Unlike aggregates, zones may OVERLAP — a space belongs to one
        decomposition but any number of groups — which is why the two are
        separate verbs rather than one with a flag.

        Space-boundary inference (``IfcRelSpaceBoundary``, internal/external
        classification, adjacency) is deliberately NOT here; it is greenfield
        geometry work with its own workstream.
        """
        from lite_step.ifc.groupings import (
            ZONE, validate_grouping_name, validate_members,
        )

        key = validate_grouping_name(name, ZONE)
        self._groupings.append(
            (ZONE, key, validate_members(members, ZONE, key)))
        return self

    def authored_aabb(self, divisor: float = 1.0):
        """Extent of the WHOLE model — every storey and every site.

        The one bounding query that is unambiguous by construction: a Project
        is the root, so there is no placement still to come and no distinction
        between the authored and world stages. ``world_aabb()`` on a Project
        would answer identically, which is why it is not offered — two names
        for one number invite the reader to assume a difference.
        """
        from lite_step.models.bounds import Bounds, BoundsError
        from lite_step.compiler.extent import element_extent, _union

        boxes = [element_extent(e, divisor)
                 for s in self.storeys for e in s.elements]
        boxes += [element_extent(s, divisor) for s in self.sites]
        box = _union(boxes)
        if box is None:
            raise BoundsError(
                f"Project '{self.name}' has no extent to report — it holds no "
                f"geometry. An empty model has no bounding box, and a "
                f"degenerate one at the origin would silently anchor whatever "
                f"you author against it to the origin too."
            )
        return Bounds.from_aabb(box, divisor=divisor)

    def get_all_elements(self) -> List[ElementType]:
        """Get all elements from all storeys."""
        elements = []
        for storey in self.storeys:
            elements.extend(storey.elements)
        return elements

    @property
    def elements(self):
        """Refused on purpose, with the alternatives named.

        A Project holds ``sites`` and ``storeys``; it keeps NO list of its
        own. ``proj.add(wall)`` routes the wall into ``storeys[0]``, so there
        is no honest answer here — returning the storeys' contents would make
        a caller that walks storeys AND this property add everything twice,
        which is a worse failure than the omission it would be fixing.

        Every element CONTAINER answers ``.elements`` (see
        ``BimElement.elements``); a Project is not one, and raising here keeps
        ``hasattr(proj, "elements")`` correctly False while still telling a
        reader what to call instead.
        """
        raise AttributeError(
            "Project has no .elements — it holds sites and storeys, and "
            ".add() routes elements into storeys[0]. Use .storeys / .sites "
            "to walk the tree, or .get_all_elements() for every storey's "
            "elements flattened."
        )

    class Config:
        extra = "forbid"
