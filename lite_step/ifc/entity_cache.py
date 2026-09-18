"""
Entity Cache - Deduplicate commonly repeated IFC entities.

This module provides caching for IFC entities that are frequently
recreated with identical values, reducing file size and improving
generation performance.

Commonly duplicated entities:
- IfcDirection: Z-up direction [0,0,1] created for every extrusion
- IfcCartesianPoint: Origin points and repeated coordinates
- IfcRectangleProfileDef: Same profile dimensions across elements
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import ifcopenshell


def _round_key(value: float, precision: int = 6) -> float:
    """Round value to specified decimal places for consistent hashing."""
    return round(value, precision)


#: Decimal places a linear measure (coordinate span / extrusion depth /
#: profile dimension) is snapped to. Geometry is authored in int-mm and
#: normalized ``/1000`` to meters, so 6 places (micrometre precision) is
#: lossless while killing meter-space subtraction drift
#: (``3.2 - 3.0 == 0.20000000000000018`` -> ``0.2``;).
#: Bump when variable precision scales land (cm / mm / 0.1 mm).
GEOMETRY_DECIMALS = 6


def snap_to_precision(value: float, decimals: int = GEOMETRY_DECIMALS) -> float:
    """Snap a LINEAR measure to ``decimals`` places (multiplier rounding).

    Apply to the immediate output of a coordinate subtraction/addition — the
    dimension of a discrete element — so every downstream consumer (profile
    dims, extrusion ``Depth``, derived offsets) sees the clean value. Do NOT
    apply per-step inside a cumulative loop (rounding error would accumulate),
    and NOT to direction ratios (unit vectors need full precision).

    Snapping a direction was tried and reverted: ``_apply_clips`` re-normalizes
    every clip normal to unit length before emitting (OCC tessellates a
    non-unit plane to EMPTY geometry), so a snapped 45-degree normal comes back
    out at full width — the rounding cost precision in the DSL value and
    changed nothing in the file.
    """
    return round(value, decimals)


@dataclass(frozen=True)
class DirectionKey:
    """Hashable key for IfcDirection caching."""
    x: float
    y: float
    z: float

    @classmethod
    def from_coords(cls, x: float, y: float, z: float) -> "DirectionKey":
        """Create key with rounded coordinates for stable hashing."""
        return cls(
            x=_round_key(x),
            y=_round_key(y),
            z=_round_key(z)
        )


@dataclass(frozen=True)
class PointKey:
    """Hashable key for IfcCartesianPoint caching."""
    coords: Tuple[float, ...]

    @classmethod
    def from_coords(cls, coords: list | tuple) -> "PointKey":
        """Create key with rounded coordinates for stable hashing."""
        return cls(coords=tuple(_round_key(c) for c in coords))


@dataclass(frozen=True)
class RectProfileKey:
    """Hashable key for IfcRectangleProfileDef caching."""
    x_dim: float
    y_dim: float
    name: str

    @classmethod
    def from_dims(cls, x_dim: float, y_dim: float, name: str = "") -> "RectProfileKey":
        """Create key with rounded dimensions for stable hashing."""
        return cls(
            x_dim=_round_key(x_dim),
            y_dim=_round_key(y_dim),
            name=name or ""
        )


@dataclass(frozen=True)
class StyleChainKey:
    """Hashable key for surface style chain caching (color + transparency + name)."""
    r: float
    g: float
    b: float
    transparency: float
    name: Optional[str] = None

    @classmethod
    def from_color(cls, r: float, g: float, b: float, transparency: float = 0.0,
                   name: Optional[str] = None) -> "StyleChainKey":
        """Create key with rounded color values for stable hashing."""
        return cls(
            r=_round_key(r, 4),
            g=_round_key(g, 4),
            b=_round_key(b, 4),
            transparency=_round_key(transparency, 4),
            name=name,
        )


class EntityCache:
    """
    Cache for deduplicating frequently created IFC entities.
    
    Usage:
        model = ifcopenshell.file(schema=IFC_OUTPUT_SCHEMA)
        cache = EntityCache(model)
        
        # Instead of model.create_entity("IfcDirection", ...)
        direction = cache.get_or_create_direction(0.0, 0.0, 1.0)
        
        # Subsequent calls with same coords return cached entity
        same_dir = cache.get_or_create_direction(0.0, 0.0, 1.0)
        assert direction is same_dir
    
    Thread Safety:
        NOT thread-safe. Each thread/process should use its own cache.
    
    Statistics:
        Call get_stats() to see cache hit/miss ratios.
    """

    def __init__(self, model: "ifcopenshell.file"):
        """
        Initialize cache for the given IFC model.
        
        Args:
            model: ifcopenshell.file instance to create entities in
        """
        self.model = model

        # Entity caches keyed by frozen dataclass
        self._directions: Dict[DirectionKey, Any] = {}
        self._points: Dict[PointKey, Any] = {}
        self._rect_profiles: Dict[RectProfileKey, Any] = {}
        self._style_chains: Dict[StyleChainKey, Any] = {}  # -> IfcSurfaceStyle

        # unique_dims registry: (x_dim, y_dim) float pair -> owning salt name.
        # Used by get_or_create_rect_profile(unique_dims=True) to guarantee
        # differently-salted profiles never share the exact same dims (the
        # geometry-signature noise that defeats the SPA render worker's
        # content-based instancing — see the method docstring).
        self._unique_rect_dims: Dict[tuple, str] = {}

        # Registry materials (v1.5): one IfcMaterial (+ IfcMaterialProperties)
        # per key, one IfcRelAssociatesMaterial per key with products appended
        self._ifc_materials: Dict[str, Any] = {}
        self._material_rels: Dict[str, Any] = {}
        #: Entity ids the Product geometry purge has condemned. Collected here
        #: and deleted in one pass at serialization rather than one
        #: `file.remove` at a time — see `deduplicator._drop_purged`.
        self._purged_ids: set = set()
        #: material key -> the products associated with it, accumulated in
        #: Python and written onto the relationship once. See
        #: ``associate_material`` for why this is not appended entity-side.
        self._material_members: Dict[str, list] = {}

        # Layer buildups (v15.2): one IfcMaterialLayerSet per distinct
        # buildup content, one IfcMaterialLayerSetUsage (+ its association
        # rel) per (buildup, direction, sense) with products appended.
        self._layer_sets: Dict[tuple, Any] = {}
        self._layer_usage_rels: Dict[tuple, Any] = {}

        # Which IFC product each container child emitted, keyed by the
        # CONTAINER's id(): {id(container): [(dsl child, product), ...]}.
        # Recorded as emission runs, and read only by ``ifc.voids`` — see
        # ``record_child_product``.
        self._child_products: Dict[int, list] = {}

        # Statistics (use Any to allow float for hit ratios in get_stats)
        self._stats: Dict[str, Any] = {
            "directions_hits": 0,
            "directions_misses": 0,
            "points_hits": 0,
            "points_misses": 0,
            "rect_profiles_hits": 0,
            "rect_profiles_misses": 0,
            "style_chains_hits": 0,
            "style_chains_misses": 0,
        }

    def record_child_product(self, container: Any, child: Any, product: Any) -> None:
        """Record that ``child`` (a DSL element) emitted ``product`` under
        ``container``.

        The pairing a container's ``.void()`` distribution needs, stated by the
        emitter that knows it rather than re-derived afterwards. Inferring it
        by zipping ``container.elements`` against the product list is what
        :mod:`lite_step.ifc.voids` documents as wrong in three separate silent
        ways — a nested container has no AABB, a ``Mesh`` child emits no
        product, and a layered container emits SLICES that are not in
        ``.elements`` at all.

        Keyed on ``id()``: the DSL tree is alive for the whole compile, and a
        shared element is a compile ERROR (``naming.find_shared_elements``), so
        one object is one place in the tree.
        """
        self._child_products.setdefault(id(container), []).append((child, product))

    def child_products(self, container: Any) -> list:
        """The recorded ``(dsl child, product)`` pairs for ``container``."""
        return self._child_products.get(id(container), [])

    def get_or_create_direction(self, x: float, y: float, z: float) -> Any:
        """
        Get cached IfcDirection or create new one.
        
        Args:
            x, y, z: Direction ratios (will be rounded for cache lookup)
            
        Returns:
            IfcDirection entity
        """
        key = DirectionKey.from_coords(x, y, z)
        
        if key in self._directions:
            self._stats["directions_hits"] += 1
            return self._directions[key]
        
        self._stats["directions_misses"] += 1
        direction = self.model.create_entity(
            "IfcDirection",
            DirectionRatios=[float(x), float(y), float(z)]
        )
        self._directions[key] = direction
        return direction

    def get_or_create_point(self, coords: list | tuple) -> Any:
        """
        Get cached IfcCartesianPoint or create new one.
        
        Args:
            coords: List/tuple of coordinates (2D or 3D)
            
        Returns:
            IfcCartesianPoint entity
        """
        key = PointKey.from_coords(coords)
        
        if key in self._points:
            self._stats["points_hits"] += 1
            return self._points[key]
        
        self._stats["points_misses"] += 1
        point = self.model.create_entity(
            "IfcCartesianPoint",
            Coordinates=[float(c) for c in coords]
        )
        self._points[key] = point
        return point

    # Nudge step for ``unique_dims``: 20 micrometres (0.02 mm). Chosen to
    # survive BOTH downstream flattening layers:
    #   * the STEP deduplicator normally rounds floats to 0.1 mm before
    #     merging — rect profiles are exempted from that rounding
    #     (deduplicator.DEDUP_NO_FLOAT_ROUND_TYPES) precisely so this noise
    #     reaches the file; and
    #   * the SPA render stack tessellates into float32 vertex buffers —
    #     at facade scale (~6-13 m coordinates) one float32 ulp is ~1 um,
    #     so a nudge must be >~10 um to change the actual vertex bytes.
    #     (An earlier 10 nm draft survived nothing: rounded away in STEP,
    #     and quantised away in float32 even if it hadn't been.)
    # 0.02 mm is still 2 orders below build tolerance and below the 0.1 mm
    # display/QTO precision — invisible everywhere that matters.
    _UNIQUE_DIMS_STEP = 2e-5

    def _claim_unique_dims(self, x_dim: float, y_dim: float, name: str,
                           salt: str = "x") -> Tuple[float, float]:
        """Return ``(x_dim, y_dim)`` not shared with any DIFFERENTLY-named
        unique-dims profile in this model.

        Deterministic: repeat calls with the same (dims, name) return the same
        nudged value (the first claim wins the exact dims; later distinct names
        walk up in 10 nm steps until free). Same-name repeats reuse the claim,
        so multiple identical slices within one wall still share one profile.

        ``salt`` names the dimension the nudge lands in. It is a REQUIRED
        judgement by the caller, not a detail: the nudge is sub-tolerance noise
        only in a dimension nothing derives from. A layered wall sliced along
        local X has its layer THICKNESS in ``XDim``, and salting that perturbs
        the buildup itself — measured on a mitered ring, a 43 mm cavity read
        42.98 mm in Blender and each slice sat 0.01 mm proud of the miter plane
        it had been clipped to. Salt the length; never the thickness.

        A zero step returns immediately. Without that guard the loop cannot
        advance and spins forever the moment two walls share slice dims —
        which is what happens to anyone who tries to switch the salt off by
        setting ``_UNIQUE_DIMS_STEP = 0.0``.
        """
        x = float(x_dim)
        y = float(y_dim)
        if not self._UNIQUE_DIMS_STEP:
            return x, y
        i = 0 if salt == "x" else 1
        dims = [x, y]
        while True:
            key = (dims[0], dims[1])
            owner = self._unique_rect_dims.get(key)
            if owner is None:
                self._unique_rect_dims[key] = name
                return dims[0], dims[1]
            if owner == name:
                return dims[0], dims[1]
            dims[i] += self._UNIQUE_DIMS_STEP

    def get_or_create_rect_profile(
        self,
        x_dim: float,
        y_dim: float,
        name: str = "",
        profile_type: str = "AREA",
        unique_dims: bool = False,
        salt: str = "x",
    ) -> Any:
        """
        Get cached IfcRectangleProfileDef or create new one.

        Note: Only caches profiles with the same name AND dimensions.
        Unnamed profiles with matching dimensions will share the same entity —
        pass a distinct ``name`` to force a BYTE-distinct profile (a name-only
        split survives the STEP serializer's identical-entity purge).

        ``unique_dims=True`` (requires ``name``) additionally guarantees the
        emitted ``(XDim, YDim)`` float pair is unique among differently-named
        unique-dims profiles, by nudging XDim in 10 nm steps (deterministic,
        sub-tolerance — geometry-signature noise). Why: a mirror pair (west/
        east facade leaf) with byte-IDENTICAL dims tessellates to identical
        vertex data, which the SPA's multi-threaded fragments render worker
        groups into one instanced mesh; a race in that worker then sometimes
        drops an instance (the missing-cladding bug — nondeterministic, which
        is why usually ONE wall of the two mirror pairs vanishes rather than
        one per pair). A name-only split (v18.0.4) proved insufficient: the
        worker keys on geometry content, not on ProfileName. Distinct dims →
        distinct vertex data → no shared-geometry group → nothing to drop.
        Scoped to the layered-wall slice path; deliberate instancing (rafters,
        repeated frames) keeps sharing exact dims and stays instanced.

        Args:
            x_dim: Width (X dimension) in meters
            y_dim: Depth (Y dimension) in meters
            name: Optional profile name (empty string if not needed)
            profile_type: Profile type (default "AREA")
            unique_dims: Enforce cross-name dims uniqueness (needs ``name``)

        Returns:
            IfcRectangleProfileDef entity
        """
        if unique_dims and name:
            x_dim, y_dim = self._claim_unique_dims(x_dim, y_dim, name, salt=salt)

        key = RectProfileKey.from_dims(x_dim, y_dim, name)

        if key in self._rect_profiles:
            self._stats["rect_profiles_hits"] += 1
            return self._rect_profiles[key]

        self._stats["rect_profiles_misses"] += 1

        # Build kwargs - only include name if provided
        kwargs: Dict[str, Any] = {
            "ProfileType": profile_type,
            "XDim": float(x_dim),
            "YDim": float(y_dim),
        }
        if name:
            kwargs["ProfileName"] = name

        profile = self.model.create_entity("IfcRectangleProfileDef", **kwargs)
        self._rect_profiles[key] = profile
        return profile

    def get_or_create_style_chain(self, r: float, g: float, b: float, transparency: float = 0.0,
                                  name: Optional[str] = None) -> Any:
        """
        Get cached IfcSurfaceStyle or create full style chain.

        Chain: IfcColourRgb -> IfcSurfaceStyleShading/Rendering -> IfcSurfaceStyle

        IFC4 assigns IfcSurfaceStyle directly to IfcStyledItem.Styles
        (IfcPresentationStyleAssignment is deprecated in IFC4).

        The IfcStyledItem (linking a specific solid to the style) is NOT cached
        since it must be unique per geometry item.

        Args:
            r, g, b: Color components in 0.0-1.0 range
            transparency: 0.0 = fully opaque, 1.0 = fully transparent
            name: overrides the IfcSurfaceStyle Name; when omitted the default
                'Glass' (translucent) / 'Color' (opaque) is kept. Terrain passes
                'Terrain' for an opaque, correctly-named ground surface. The name
                is part of the cache key so distinct-named styles don't collide.

        Returns:
            IfcSurfaceStyle entity (shared across same-colored elements)
        """
        key = StyleChainKey.from_color(r, g, b, transparency, name)

        if key in self._style_chains:
            self._stats["style_chains_hits"] += 1
            return self._style_chains[key]

        self._stats["style_chains_misses"] += 1

        colour_rgb = self.model.create_entity(
            "IfcColourRgb", Name=None, Red=float(r), Green=float(g), Blue=float(b)
        )

        if transparency > 0.001:
            surface_rendering = self.model.create_entity(
                "IfcSurfaceStyleRendering",
                SurfaceColour=colour_rgb,
                Transparency=float(transparency),
                ReflectanceMethod="FLAT"
            )
            surface_style = self.model.create_entity(
                "IfcSurfaceStyle", Name=name or "Glass", Side="BOTH",
                Styles=[surface_rendering]
            )
        else:
            surface_shading = self.model.create_entity(
                "IfcSurfaceStyleShading", SurfaceColour=colour_rgb
            )
            surface_style = self.model.create_entity(
                "IfcSurfaceStyle", Name=name or "Color", Side="BOTH",
                Styles=[surface_shading]
            )

        self._style_chains[key] = surface_style
        return surface_style

    def get_or_create_ifc_material(self, key: str, definition: Any = None) -> Any:
        """
        Get cached IfcMaterial for a registry key, or create it.

        On first creation, the registry definition's Psets (product/EPD data)
        are attached as IFC4 material-level property sets
        (IfcMaterialProperties, one per Pset name).

        ``Category`` carries the registry's material FAMILY (concrete, masonry,
        timber, …). This is the attribute 2D drawing tools hatch on — Bonsai
        derives the SVG class ``layer-material-category-<Category>`` from it —
        so leaving it null (as this did before v21.2) means no material-aware
        hatching is reachable in any downstream drawing. It is a DIFFERENT
        attribute from ``IfcMaterialLayer.Category``, which carries the layer
        ROLE (``MaterialDef.function``); see ``get_or_create_layer_set``.

        Args:
            key: Material key — becomes IfcMaterial.Name (e.g. "Glass_VIG")
            definition: Optional MaterialDef from lite_step.materials; None
                for open-vocabulary keys (plain named material, no Psets)

        Returns:
            IfcMaterial entity (shared across all elements with the key)
        """
        if key in self._ifc_materials:
            return self._ifc_materials[key]

        ifc_material = self.model.create_entity(
            "IfcMaterial", Name=key,
            Category=getattr(definition, "category", None) if definition is not None else None,
        )
        if definition is not None:
            for pset_name, props in definition.psets.items():
                prop_entities = [
                    self.model.create_entity(
                        "IfcPropertySingleValue",
                        Name=prop_name,
                        NominalValue=self._wrap_pset_value(value),
                    )
                    for prop_name, value in props.items()
                ]
                self.model.create_entity(
                    "IfcMaterialProperties",
                    Name=pset_name,
                    Properties=prop_entities,
                    Material=ifc_material,
                )
        self._ifc_materials[key] = ifc_material
        return ifc_material

    def associate_material(self, product: Any, ifc_material: Any) -> Any:
        """
        Associate a product with an IfcMaterial via IfcRelAssociatesMaterial.

        One relationship per material key; subsequent products append to its
        RelatedObjects (keeps the STEP output lean).

        **Members accumulate in a PYTHON list and reach the entity once**, in
        :meth:`flush_material_associations`. The obvious spelling —
        ``rel.RelatedObjects = list(rel.RelatedObjects) + [product]`` — reads
        the whole aggregate out of the C++ layer, appends one, and writes it
        all back, so appending N members costs O(N^2). Measured on
        unitized-curtain-wall: 15,000 calls, 74.3M ``entity_instance.walk``
        steps, ~203 s = 30% of the compile.

        Order is preserved exactly (plain append), so the emitted aggregate is
        the same sequence the incremental version produced.
        """
        import ifcopenshell

        key = ifc_material.Name
        rel = self._material_rels.get(key)
        if rel is None:
            rel = self.model.create_entity(
                "IfcRelAssociatesMaterial",
                GlobalId=ifcopenshell.guid.new(),
                RelatedObjects=[product],
                RelatingMaterial=ifc_material,
            )
            self._material_rels[key] = rel
            self._material_members[key] = [product]
        else:
            self._material_members[key].append(product)
        return rel

    def record_purged(self, ids) -> None:
        """Condemn entity ids for deletion at serialization.

        Ids rather than handles on purpose: the handles stay alive in the model
        until the file is written, and holding them would only invite a later
        pass to follow one.
        """
        self._purged_ids.update(ids)

    @property
    def purged_ids(self) -> set:
        return self._purged_ids

    def flush_material_associations(self) -> None:
        """Write each accumulated member list onto its relationship.

        Idempotent, so a caller that flushes twice cannot double a list — the
        assignment REPLACES rather than appends. Only the relationships whose
        membership actually grew are touched; a single-member association was
        already written correctly at creation.
        """
        for key, rel in self._material_rels.items():
            members = self._material_members.get(key) or []
            if len(members) > 1:
                rel.RelatedObjects = members

    def _layer_set_key(self, layer_set: Any, scale: float = 1.0) -> tuple:
        """Content key for buildup sharing: identical buildups (each layer's
        material key + thickness, plus the set name) share ONE
        IfcMaterialLayerSet across elements. Category/IsVentilated derive from
        the registry, so the key (material key + thickness) already captures
        them.

        ``scale`` participates because two elements can author the SAME buildup
        and stretch it differently onto their bodies — they then describe
        different LayerThicknesses and must not share one set."""
        return (
            layer_set.name,
            tuple((m.key, m.thickness_mm) for m in layer_set.layers),
            scale,
        )

    def get_or_create_layer_set(self, layer_set: Any, scale: float = 1.0) -> Any:
        """Get cached IfcMaterialLayerSet for a DSL ``LayerSet``, or create it.

        Each layer ``Material`` becomes an ``IfcMaterialLayer(Material=<shared
        per-key IfcMaterial>, LayerThickness=<meters>, Category, IsVentilated)``
        — everything DERIVED from the registry (never re-authored): Psets ride
        the shared IfcMaterial; ``Category`` = the material's ``function``;
        ``IsVentilated`` = a cavity material's ``ventilated``. Layer order is
        the authored list order (outer→inner).
        """
        key = self._layer_set_key(layer_set, scale)
        if key in self._layer_sets:
            return self._layer_sets[key]

        from lite_step.materials import registry_definition

        # Thicknesses are derived from SNAPPED CUMULATIVE boundaries, not
        # snapped independently: rounding each layer on its own lets the error
        # accumulate, so a stretched buildup's layers stop summing to the
        # extent they were stretched onto (271 -> 280 mm summed to 280.001).
        # This mirrors ``layers.layer_spans``, which snaps its LAST span to the
        # far face for the same reason — the two must agree layer for layer.
        _cum, _bounds = 0.0, []
        for mat in layer_set.layers:
            _cum += mat.thickness_mm / 1000.0 * scale
            _bounds.append(snap_to_precision(_cum))
        # Snap the difference too — subtracting two 6-dp values reintroduces
        # binary noise (0.258 - 0.108 = 0.15000000000000002), which would
        # churn every corpus byte for nothing.
        thicknesses = [snap_to_precision(b - a)
                       for a, b in zip([0.0] + _bounds, _bounds)]

        ifc_layers = []
        for mat, thickness in zip(layer_set.layers, thicknesses):
            mdef = registry_definition(mat.key)
            ifc_material = self.get_or_create_ifc_material(mat.key, mdef)
            category = mdef.function if mdef is not None else None
            is_ventilated = None
            if mdef is not None and mdef.geometry.form == "cavity":
                is_ventilated = mdef.geometry.ventilated
            ifc_layers.append(self.model.create_entity(
                "IfcMaterialLayer",
                Material=ifc_material,
                LayerThickness=thickness,
                IsVentilated=is_ventilated,
                Category=category,
            ))
        ifc_set = self.model.create_entity(
            "IfcMaterialLayerSet",
            MaterialLayers=ifc_layers,
            LayerSetName=layer_set.name,
        )
        self._layer_sets[key] = ifc_set
        return ifc_set

    def associate_layer_usage(self, product: Any, layer_set: Any,
                              direction_axis: str,
                              offset_from_reference_line: float,
                              thickness_scale: float = 1.0,
                              outer_at_max: bool = False) -> Any:
        """Associate a product with its buildup via the full occurrence chain
        ``IfcRelAssociatesMaterial → IfcMaterialLayerSetUsage →
        IfcMaterialLayerSet → IfcMaterialLayer``.

        ``direction_axis``: ``"axis1"`` / ``"axis2"`` / ``"axis3"`` — the axis
        the buildup is actually stacked along. Passing ``"axis2"`` unconditionally
    while slicing along whichever horizontal axis is thin
        so half the walls of a ring described themselves wrongly to any
        consumer that walks the buildup.
        ``outer_at_max`` (from ``LayerSet.outward``): the first-listed (outer)
        layer sits at the through-axis MAX face → the layers run from that
        reference toward MIN, so ``DirectionSense`` is NEGATIVE; POSITIVE when
        the outer layer sits at MIN.

        ``offset_from_reference_line``: signed distance, in metres and along
        the through axis, from the product's ``ObjectPlacement`` to the face
        the FIRST-listed layer starts at. **Required, not defaulted** — it was
        hardcoded ``0.0`` across four call sites and stayed wrong in all of
        them. Our products carry an identity-rotation
        placement at the body's AABB *centre*, so with ``0.0`` a consumer
        re-derives every layer boundary half a wall thickness off: on a
        108/43/190 buildup it cuts at +108 mm and +151 mm from the centre —
        both inside the insulation, 43 mm apart. That pair of planes is the
        reported phantom band, and its width tracks the middle layer's
        thickness because it *is* that thickness. See
        ``lite_step/tests/test_material_layers.py`` for the derivation test.

        The usage (the placement) is keyed per (buildup, direction, sense,
        offset) — two walls facing opposite ways, or of different total
        thickness, get DISTINCT usages instead of colliding (which dropped a
        side), while the ``IfcMaterialLayerSet`` (the type) stays shared
        across all of them.
        """
        import ifcopenshell

        ifc_set = self.get_or_create_layer_set(layer_set, thickness_scale)
        offset = snap_to_precision(offset_from_reference_line)
        usage_key = (self._layer_set_key(layer_set, thickness_scale),
                     direction_axis, outer_at_max, offset)
        rel = self._layer_usage_rels.get(usage_key)
        if rel is None:
            usage = self.model.create_entity(
                "IfcMaterialLayerSetUsage",
                ForLayerSet=ifc_set,
                LayerSetDirection={"axis1": "AXIS1", "axis2": "AXIS2"}.get(
                    direction_axis, "AXIS3"),
                DirectionSense="NEGATIVE" if outer_at_max else "POSITIVE",
                OffsetFromReferenceLine=offset,
            )
            rel = self.model.create_entity(
                "IfcRelAssociatesMaterial",
                GlobalId=ifcopenshell.guid.new(),
                RelatedObjects=[product],
                RelatingMaterial=usage,
            )
            self._layer_usage_rels[usage_key] = rel
        else:
            rel.RelatedObjects = list(rel.RelatedObjects) + [product]
        return rel

    def _wrap_pset_value(self, value: Any) -> Any:
        """Wrap a Python Pset value in the matching IFC simple type."""
        if isinstance(value, bool):  # bool before int — bool is an int subclass
            return self.model.create_entity("IfcBoolean", wrappedValue=value)
        if isinstance(value, int):
            return self.model.create_entity("IfcInteger", wrappedValue=value)
        if isinstance(value, float):
            return self.model.create_entity("IfcReal", wrappedValue=value)
        return self.model.create_entity("IfcLabel", wrappedValue=str(value))

    def get_z_up_direction(self) -> Any:
        """
        Convenience method for the commonly used Z-up direction [0,0,1].
        
        Returns:
            Cached IfcDirection for Z-up extrusion
        """
        return self.get_or_create_direction(0.0, 0.0, 1.0)

    def get_origin_3d(self) -> Any:
        """
        Convenience method for the origin point [0,0,0].
        
        Returns:
            Cached IfcCartesianPoint at origin
        """
        return self.get_or_create_point([0.0, 0.0, 0.0])

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with hit/miss counts and ratios for each entity type
        """
        stats = dict(self._stats)
        
        # Calculate ratios
        for entity_type in ["directions", "points", "rect_profiles", "style_chains"]:
            hits = stats[f"{entity_type}_hits"]
            misses = stats[f"{entity_type}_misses"]
            total = hits + misses
            stats[f"{entity_type}_hit_ratio"] = hits / total if total > 0 else 0.0
            stats[f"{entity_type}_total"] = total
            stats[f"{entity_type}_unique"] = misses  # Misses = unique entities created
        
        return stats

    def get_summary(self) -> str:
        """
        Get a human-readable summary of cache effectiveness.
        
        Returns:
            Formatted string with cache statistics
        """
        stats = self.get_stats()
        lines = [
            "EntityCache Statistics:",
            f"  Directions:  {stats['directions_unique']:4d} unique / {stats['directions_total']:4d} total ({stats['directions_hit_ratio']*100:.1f}% cache hits)",
            f"  Points:      {stats['points_unique']:4d} unique / {stats['points_total']:4d} total ({stats['points_hit_ratio']*100:.1f}% cache hits)",
            f"  RectProfiles:{stats['rect_profiles_unique']:4d} unique / {stats['rect_profiles_total']:4d} total ({stats['rect_profiles_hit_ratio']*100:.1f}% cache hits)",
            f"  StyleChains: {stats['style_chains_unique']:4d} unique / {stats['style_chains_total']:4d} total ({stats['style_chains_hit_ratio']*100:.1f}% cache hits)",
        ]
        return "\n".join(lines)

    def clear(self):
        """Clear all caches. Useful if reusing cache across multiple models (not recommended)."""
        self._directions.clear()
        self._points.clear()
        self._rect_profiles.clear()
        self._unique_rect_dims.clear()
        self._style_chains.clear()
        # Reset stats
        for key in self._stats:
            self._stats[key] = 0
