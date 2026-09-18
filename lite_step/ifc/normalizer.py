"""
IFC Normalizer: Parse IFC files into structured ExtractedElement lists.

Walks the IFC spatial hierarchy, resolves nested placements to absolute
world coordinates, classifies geometry representations, and converts
from IFC conventions (Z-up, meters) to DSL conventions (Z-up, mm integers).

Supports IFC2X3, IFC4, and attempts best-effort parsing of IFC2X2 and IFC4X3
by rewriting schema headers to the nearest supported version.
"""

import logging
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    import ifcopenshell
    import numpy as np
    IFC_AVAILABLE = True
except ImportError:
    IFC_AVAILABLE = False
    ifcopenshell = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Spatial structure types — not real "elements" for reconstruction
SPATIAL_TYPES = frozenset({
    "IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcSpace",
    "IfcRelContainedInSpatialStructure", "IfcRelAggregates",
})

# Schema rewrite map: unsupported → closest supported
_SCHEMA_REWRITE = {
    "IFC2X2_FINAL": "IFC2X3",
    "IFC2X2": "IFC2X3",
    "IFC4X3_RC1": "IFC4",
    "IFC4X3_RC2": "IFC4",
    "IFC4X3_ADD1": "IFC4",
    "IFC4X3_ADD2": "IFC4",
    "IFC4X3": "IFC4",
}


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class NormalizationResult:
    """Full result from IFC normalization, including metadata."""
    elements: List["ExtractedElement"]
    total_products_found: int = 0
    skipped_types: Dict[str, int] = field(default_factory=dict)
    schema: str = "UNKNOWN"
    schema_rewritten: bool = False
    parse_error: Optional[str] = None
    #: Project-level groupings — ``proj.aggregate()`` / ``proj.zone()``
    #: statements recovered from the file. Kept OUT of ``elements`` on
    #: purpose: a grouping parent carries no geometry, so anything
    #: reconstructing elements would either invent a body for it or
    #: double-count its members. See :func:`_extract_groupings`.
    groupings: List["ExtractedGrouping"] = field(default_factory=list)


@dataclass
class PlacementInfo:
    """Absolute placement in DSL coordinates (Z-up, mm integers)."""
    translation: Tuple[int, int, int]   # (x, y, z) mm
    rotation: Tuple[int, int, int]      # Euler angles x1000


@dataclass
class GeometryInfo:
    """Classified geometry extracted from IFC representation."""
    mode: str  # "box", "contour", "boolean", "mesh", "unknown"
    # Box mode
    start: Optional[Tuple[int, int, int]] = None   # mm, Z-up
    end: Optional[Tuple[int, int, int]] = None      # mm, Z-up
    # Contour mode
    contour: Optional[List[Tuple[int, int]]] = None  # 2D vertices, mm
    thickness: Optional[int] = None                   # mm
    extrusion_direction: Optional[Tuple[float, float, float]] = None
    # Boolean mode
    boolean_operands: Optional[List["GeometryInfo"]] = None
    boolean_operation: Optional[str] = None  # "difference" or "union"
    # Mesh mode (IfcFacetedBrep). mesh_vertices is the absolute-world
    # vertex list in DSL coords (Z-up, mm integers); mesh_faces is a
    # list of triangle index triples into mesh_vertices. Quad+ faces
    # are fan-triangulated. start/end always also populated as a
    # bbox fallback so cold-import can render even when the
    # reconstructor only knows boxes.
    mesh_vertices: Optional[List[Tuple[int, int, int]]] = None
    mesh_faces: Optional[List[Tuple[int, int, int]]] = None


@dataclass
class ExtractedGrouping:
    """A project-level grouping recovered from an IFC file.

    The import counterpart of ``proj.aggregate()`` / ``proj.zone()``. Before
    this existed the walk was ``IfcRelContainedInSpatialStructure``-only, so a
    geometry-less aggregate parent was **silently invisible** on import — its
    name, its identity and any facade Psets dropped with no warning, which is
    the silent-degradation failure this codebase refuses.
    """
    kind: str                   # "aggregate" (IfcRelAggregates) | "zone" (IfcZone)
    name: str                   # the parent's IFC Name — the author's key
    guid: str                   # the parent entity's GlobalId
    ifc_type: str               # IfcWall, IfcZone, ...
    members: List[str] = field(default_factory=list)   # member IFC Names, file order
    member_guids: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedElement:
    """A normalized element extracted from an IFC file."""
    ifc_type: str               # IfcWall, IfcColumn, etc.
    name: str                   # = element.id from IFC Name field
    guid: str                   # IFC GlobalId
    storey_idx: int             # Which storey (by elevation order)
    storey_elevation: int       # mm
    placement: PlacementInfo
    geometry: GeometryInfo
    children: List["ExtractedElement"] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Coordinate transforms (inverse of generator's dsl_to_ifc_point)
# ---------------------------------------------------------------------------

def _ifc_to_dsl_point(x_ifc: float, y_ifc: float, z_ifc: float) -> Tuple[int, int, int]:
    """
    Convert IFC coordinates (Z-up, meters) to DSL coordinates (Z-up, mm integers).

    Both IFC and DSL use Z-up. Only unit conversion needed (meters → mm).
    """
    return (
        int(round(x_ifc * 1000)),
        int(round(y_ifc * 1000)),
        int(round(z_ifc * 1000)),
    )


def _ifc_to_dsl_direction(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """Convert IFC direction vector to DSL direction (identity, no swap)."""
    return (x, y, z)


# ---------------------------------------------------------------------------
# Placement resolution
# ---------------------------------------------------------------------------

def _get_axis2placement3d_matrix(placement) -> "np.ndarray":
    """Build 4x4 matrix from IfcAxis2Placement3D."""
    loc = placement.Location.Coordinates if placement.Location else (0, 0, 0)

    # Z axis (Axis attribute, defaults to [0,0,1])
    if placement.Axis:
        z_axis = np.array(placement.Axis.DirectionRatios, dtype=np.float64)
    else:
        z_axis = np.array([0, 0, 1], dtype=np.float64)
    z_axis = z_axis / (np.linalg.norm(z_axis) or 1.0)

    # X axis (RefDirection attribute, defaults to [1,0,0])
    if placement.RefDirection:
        x_axis = np.array(placement.RefDirection.DirectionRatios, dtype=np.float64)
    else:
        x_axis = np.array([1, 0, 0], dtype=np.float64)
    # Orthonormalize: X must be perpendicular to Z
    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    norm = np.linalg.norm(x_axis)
    if norm > 1e-10:
        x_axis = x_axis / norm
    else:
        x_axis = np.array([1, 0, 0], dtype=np.float64)

    # Y axis = Z cross X
    y_axis = np.cross(z_axis, x_axis)

    mat = np.eye(4, dtype=np.float64)
    mat[0:3, 0] = x_axis
    mat[0:3, 1] = y_axis
    mat[0:3, 2] = z_axis
    mat[0:3, 3] = [loc[0], loc[1], loc[2]]
    return mat


def _resolve_placement(product) -> "np.ndarray":
    """
    Resolve nested IfcLocalPlacement chain to absolute 4x4 matrix.

    Walks PlacementRelTo chain and composes transformations.
    """
    if not hasattr(product, "ObjectPlacement") or product.ObjectPlacement is None:
        return np.eye(4, dtype=np.float64)

    placement = product.ObjectPlacement
    matrices = []

    while placement is not None:
        if hasattr(placement, "RelativePlacement") and placement.RelativePlacement is not None:
            rel = placement.RelativePlacement
            if rel.is_a("IfcAxis2Placement3D"):
                matrices.append(_get_axis2placement3d_matrix(rel))
            else:
                # IfcAxis2Placement2D fallback
                loc = rel.Location.Coordinates if rel.Location else (0, 0)
                mat = np.eye(4, dtype=np.float64)
                mat[0, 3] = loc[0]
                mat[1, 3] = loc[1] if len(loc) > 1 else 0
                matrices.append(mat)

        placement = getattr(placement, "PlacementRelTo", None)

    # Compose from root (last) to leaf (first)
    result = np.eye(4, dtype=np.float64)
    for mat in reversed(matrices):
        result = result @ mat

    return result


def _matrix_to_placement_info(mat: "np.ndarray") -> PlacementInfo:
    """
    Extract PlacementInfo from absolute 4x4 matrix.

    Converts IFC meters to DSL mm integers (both Z-up).
    Decomposes rotation to Euler angles * 1000.
    """
    # Translation: IFC (meters) -> DSL (mm) — both Z-up
    tx, ty, tz = mat[0, 3], mat[1, 3], mat[2, 3]
    translation = _ifc_to_dsl_point(tx, ty, tz)

    # Rotation decomposition from 3x3 submatrix
    R = mat[0:3, 0:3].copy()
    rotation = _decompose_rotation(R)

    return PlacementInfo(translation=translation, rotation=rotation)


def _decompose_rotation(R: "np.ndarray") -> Tuple[int, int, int]:
    """
    Decompose 3x3 rotation matrix to Euler XYZ angles.

    Returns angles in Lite-STEP convention (degrees * 1000).
    Both IFC and DSL use Z-up — direct mapping, no axis swap.
    """
    # Extract Euler angles (XYZ convention) from rotation matrix
    # R = Rx(a) * Ry(b) * Rz(c)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    if sy > 1e-6:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0

    # Convert radians to degrees, then to Lite-STEP convention (*1000)
    rx_deg = math.degrees(rx)
    ry_deg = math.degrees(ry)
    rz_deg = math.degrees(rz)

    dsl_rx = int(round(rx_deg * 1000))
    dsl_ry = int(round(ry_deg * 1000))
    dsl_rz = int(round(rz_deg * 1000))

    return (dsl_rx, dsl_ry, dsl_rz)


# ---------------------------------------------------------------------------
# Geometry classification
# ---------------------------------------------------------------------------

def _classify_geometry(product, abs_matrix: "np.ndarray") -> GeometryInfo:
    """
    Classify and extract geometry from an IFC product's representation.

    Checks Body representation items and categorizes as:
    - box: IfcExtrudedAreaSolid + IfcRectangleProfileDef
    - contour: IfcExtrudedAreaSolid + IfcArbitraryClosedProfileDef
    - boolean: IfcBooleanResult chain
    - mesh: IfcFacetedBrep (bounding box fallback)
    - unknown: anything else
    """
    if not hasattr(product, "Representation") or product.Representation is None:
        return GeometryInfo(mode="unknown")

    for rep in product.Representation.Representations:
        if rep.RepresentationIdentifier in ("Body", "Sketch"):
            for item in rep.Items:
                return _classify_item(item, abs_matrix)

    return GeometryInfo(mode="unknown")


def _classify_item(item, abs_matrix: "np.ndarray") -> GeometryInfo:
    """Classify a single representation item."""
    if item.is_a("IfcExtrudedAreaSolid"):
        return _extract_extruded_solid(item, abs_matrix)
    elif item.is_a("IfcBooleanResult"):
        return _extract_boolean(item, abs_matrix)
    elif item.is_a("IfcFacetedBrep"):
        return _extract_brep(item, abs_matrix)
    elif item.is_a("IfcBooleanClippingResult"):
        return _extract_boolean(item, abs_matrix)
    else:
        return GeometryInfo(mode="unknown")


def _extract_extruded_solid(solid, abs_matrix: "np.ndarray") -> GeometryInfo:
    """Extract geometry from IfcExtrudedAreaSolid."""
    profile = solid.SweptArea
    depth = solid.Depth

    # Get the solid's local placement (Position)
    local_mat = np.eye(4, dtype=np.float64)
    if solid.Position is not None:
        local_mat = _get_axis2placement3d_matrix(solid.Position)

    # Combined matrix: absolute placement * solid's local position
    combined = abs_matrix @ local_mat

    if profile.is_a("IfcRectangleProfileDef"):
        return _extract_box_from_rect_profile(profile, depth, solid, combined)
    elif profile.is_a("IfcArbitraryClosedProfileDef"):
        return _extract_contour_from_arbitrary_profile(profile, depth, solid, combined)
    else:
        # Unknown profile type — treat as box approximation
        return GeometryInfo(mode="unknown")


def _rect_profile_axes(profile) -> Tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """``(centre, u, v)`` of an ``IfcRectangleProfileDef`` in the sweep plane.

    ``IfcParameterizedProfileDef.Position`` is an ``IfcAxis2Placement2D``: the
    rectangle is centred on its ``Location`` with ``XDim`` along its
    ``RefDirection``. Both are OPTIONAL and both default to the identity, which
    is what every path of our own generator emits — so reading them costs
    nothing on our files and is the difference between right and wrong on a
    third-party one, where an offset or rotated profile is ordinary.

    All three vectors come back in the solid's own 3D coordinates (z = 0).
    """
    centre = np.zeros(3, dtype=np.float64)
    u = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    position = getattr(profile, "Position", None)
    if position is not None:
        if position.Location is not None:
            coords = position.Location.Coordinates
            centre = np.array(
                [coords[0], coords[1] if len(coords) > 1 else 0.0, 0.0],
                dtype=np.float64,
            )
        if position.RefDirection is not None:
            ratios = position.RefDirection.DirectionRatios
            candidate = np.array(
                [ratios[0], ratios[1] if len(ratios) > 1 else 0.0, 0.0],
                dtype=np.float64,
            )
            norm = np.linalg.norm(candidate)
            if norm > 1e-10:
                u = candidate / norm

    # v completes a right-handed 2D frame in the sweep plane.
    v = np.array([-u[1], u[0], 0.0], dtype=np.float64)
    return centre, u, v


def _extract_box_from_rect_profile(
    profile, depth: float, solid, combined_matrix: "np.ndarray"
) -> GeometryInfo:
    """
    Extract box geometry from IfcRectangleProfileDef + extrusion.

    Read off the file, not off our own emitter's convention. An
    ``IfcExtrudedAreaSolid`` sweeps its profile from the plane ``Position``
    puts it on, and the swept span runs ``[0, Depth]`` along
    ``ExtrudedDirection`` — the profile plane is the START face, never the
    middle. ``combined_matrix`` already carries ``Position``, so the box is
    that span forward from the combined origin.

    Spanning ``±Depth/2`` about the combined origin double-counts the offset
        our generator writes into ``Position`` (``z = -h/2`` on
    the wall/solid paths) and reported every box ``Depth/2`` low along the
    extrusion axis — 3 m wall at world z 0..3000 came back -1500..1500. Note
    the convention is NOT uniform even among our own paths (openings write
    ``z = 0``; a contour cut operand writes ``z = min``), and a third-party
    file is free to write anything: spanning ``[0, Depth]`` from whatever the
    file says is the one rule correct for all of them.
    """
    x_dim = profile.XDim   # width along the profile's own X
    y_dim = profile.YDim   # depth along the profile's own Y

    # Profile frame, in the solid's local coordinates.
    prof_centre, prof_u, prof_v = _rect_profile_axes(profile)
    half_u = prof_u * (x_dim / 2.0)
    half_v = prof_v * (y_dim / 2.0)

    # Extrusion vector, in the solid's local coordinates. Depth is measured
    # ALONG the direction, so the ratios are normalized before scaling.
    ext_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    ext_ratios = getattr(solid, "ExtrudedDirection", None)
    if ext_ratios is not None and ext_ratios.DirectionRatios:
        candidate = np.array(ext_ratios.DirectionRatios, dtype=np.float64)
        if candidate.shape == (3,):
            norm = np.linalg.norm(candidate)
            if norm > 1e-10:
                ext_local = candidate / norm
    sweep = ext_local * float(depth)

    rotation = combined_matrix[0:3, 0:3]
    origin = combined_matrix[0:3, 3]

    # The 8 corners: profile rectangle at the start face, and the same
    # rectangle swept forward by the full depth.
    corners = []
    for su in (-1, 1):
        for sv in (-1, 1):
            for sw in (0.0, 1.0):
                local = prof_centre + su * half_u + sv * half_v + sw * sweep
                corners.append(origin + rotation @ local)

    corners = np.array(corners)
    min_corner = corners.min(axis=0)
    max_corner = corners.max(axis=0)

    # Convert to DSL coordinates (Z-up, mm)
    start = _ifc_to_dsl_point(min_corner[0], min_corner[1], min_corner[2])
    end = _ifc_to_dsl_point(max_corner[0], max_corner[1], max_corner[2])

    return GeometryInfo(mode="box", start=start, end=end)


def _extract_contour_from_arbitrary_profile(
    profile, depth: float, solid, combined_matrix: "np.ndarray"
) -> GeometryInfo:
    """Extract contour geometry from IfcArbitraryClosedProfileDef."""
    outer_curve = profile.OuterCurve
    if outer_curve is None:
        return GeometryInfo(mode="unknown")

    # Extract 2D points from the polyline
    points_2d = []
    if outer_curve.is_a("IfcPolyline"):
        for pt in outer_curve.Points:
            coords = pt.Coordinates
            points_2d.append((coords[0], coords[1]))
    elif outer_curve.is_a("IfcIndexedPolyCurve"):
        if hasattr(outer_curve, "Points") and outer_curve.Points:
            coord_list = outer_curve.Points
            if hasattr(coord_list, "CoordList"):
                for coords in coord_list.CoordList:
                    points_2d.append((coords[0], coords[1]))
    else:
        return GeometryInfo(mode="unknown")

    if len(points_2d) < 3:
        return GeometryInfo(mode="unknown")

    # Convert 2D profile points to mm integers
    contour_mm = [(int(round(x * 1000)), int(round(y * 1000))) for x, y in points_2d]

    thickness_mm = int(round(depth * 1000))

    # Extrusion direction in world space
    ext_dir_local = np.array([0, 0, 1, 0], dtype=np.float64)
    ext_dir_world = combined_matrix @ ext_dir_local
    ext_dir = _ifc_to_dsl_direction(
        ext_dir_world[0], ext_dir_world[1], ext_dir_world[2]
    )

    return GeometryInfo(
        mode="contour",
        contour=contour_mm,
        thickness=thickness_mm,
        extrusion_direction=ext_dir,
    )


def _extract_boolean(item, abs_matrix: "np.ndarray") -> GeometryInfo:
    """Extract boolean operation geometry."""
    _op_map = {"DIFFERENCE": "difference", "UNION": "union", "INTERSECTION": "intersection"}
    op = _op_map.get(item.Operator, "union")

    operands = []
    if hasattr(item, "FirstOperand") and item.FirstOperand:
        operands.append(_classify_item(item.FirstOperand, abs_matrix))
    if hasattr(item, "SecondOperand") and item.SecondOperand:
        operands.append(_classify_item(item.SecondOperand, abs_matrix))

    # Use the first operand's geometry as the primary
    primary = operands[0] if operands else GeometryInfo(mode="unknown")

    return GeometryInfo(
        mode="boolean",
        start=primary.start,
        end=primary.end,
        boolean_operands=operands,
        boolean_operation=op,
    )


def _extract_brep(brep, abs_matrix: "np.ndarray") -> GeometryInfo:
    """Extract IfcFacetedBrep as full triangulated mesh + bbox fallback.

    Walks CfsFaces → Bounds → Polygon, dedups vertices, fan-triangulates
    any face with more than 3 vertices, and applies abs_matrix to lift
    local coordinates into world space. Always populates start/end so
    consumers that only handle bbox keep working unchanged.
    """
    vertices_ifc: List[Tuple[float, float, float]] = []
    vert_index: Dict[Tuple[float, float, float], int] = {}
    faces: List[Tuple[int, int, int]] = []

    def _add_vertex(coords: Tuple[float, float, float]) -> int:
        idx = vert_index.get(coords)
        if idx is None:
            idx = len(vertices_ifc)
            vert_index[coords] = idx
            vertices_ifc.append(coords)
        return idx

    try:
        outer = brep.Outer
        if outer and hasattr(outer, "CfsFaces"):
            for face in outer.CfsFaces:
                for bound in face.Bounds:
                    loop = bound.Bound
                    if not hasattr(loop, "Polygon"):
                        continue
                    poly_indices: List[int] = []
                    for pt in loop.Polygon:
                        coords = tuple(float(c) for c in pt.Coordinates)
                        # Apply abs_matrix to lift to world coords
                        world = abs_matrix @ np.array(
                            [coords[0], coords[1], coords[2], 1.0],
                            dtype=np.float64,
                        )
                        world_coords = (float(world[0]), float(world[1]), float(world[2]))
                        poly_indices.append(_add_vertex(world_coords))
                    # Fan-triangulate the loop (handles tris, quads, n-gons)
                    if len(poly_indices) >= 3:
                        a = poly_indices[0]
                        for i in range(1, len(poly_indices) - 1):
                            faces.append((a, poly_indices[i], poly_indices[i + 1]))
    except Exception:
        # Pass through to the "no vertices" path below
        pass

    if not vertices_ifc:
        return GeometryInfo(mode="unknown")

    # Always compute bbox over the (already world-space) vertices
    pts = np.array(vertices_ifc, dtype=np.float64)
    min_pt = pts.min(axis=0)
    max_pt = pts.max(axis=0)
    start_raw = _ifc_to_dsl_point(min_pt[0], min_pt[1], min_pt[2])
    end_raw = _ifc_to_dsl_point(max_pt[0], max_pt[1], max_pt[2])
    start = tuple(min(s, e) for s, e in zip(start_raw, end_raw))
    end = tuple(max(s, e) for s, e in zip(start_raw, end_raw))

    # Convert vertices to DSL coords (mm integers, Z-up). Dedup may
    # collide post-rounding for very close points; keep the
    # vertex_index → DSL mapping order-stable so face indices remain
    # valid.
    mesh_vertices: List[Tuple[int, int, int]] = [
        _ifc_to_dsl_point(v[0], v[1], v[2]) for v in vertices_ifc
    ]

    return GeometryInfo(
        mode="mesh",
        start=start,
        end=end,
        mesh_vertices=mesh_vertices,
        mesh_faces=faces if faces else None,
    )


# ---------------------------------------------------------------------------
# Opening extraction (Window/Door children of walls)
# ---------------------------------------------------------------------------

def _extract_openings(product) -> List[ExtractedElement]:
    """Extract IfcOpeningElement children and their Door/Window fills."""
    children = []

    # Walk IfcRelVoidsElement to find openings
    if not hasattr(product, "HasOpenings"):
        return children

    for rel in product.HasOpenings:
        opening = rel.RelatedOpeningElement
        if opening is None:
            continue

        abs_mat = _resolve_placement(opening)
        geom = _classify_geometry(opening, abs_mat)
        placement_info = _matrix_to_placement_info(abs_mat)

        # Check for fill elements (Door/Window)
        fills = []
        if hasattr(opening, "HasFillings"):
            for fill_rel in opening.HasFillings:
                fill = fill_rel.RelatedBuildingElement
                if fill is not None:
                    fill_mat = _resolve_placement(fill)
                    fill_geom = _classify_geometry(fill, fill_mat)
                    fill_placement = _matrix_to_placement_info(fill_mat)
                    fills.append(ExtractedElement(
                        ifc_type=fill.is_a(),
                        name=fill.Name or f"fill_{fill.GlobalId[:8]}",
                        guid=fill.GlobalId,
                        storey_idx=0,
                        storey_elevation=0,
                        placement=fill_placement,
                        geometry=fill_geom,
                    ))

        children.append(ExtractedElement(
            ifc_type="IfcOpeningElement",
            name=opening.Name or f"opening_{opening.GlobalId[:8]}",
            guid=opening.GlobalId,
            storey_idx=0,
            storey_elevation=0,
            placement=placement_info,
            geometry=geom,
            children=fills,
        ))

    return children


# ---------------------------------------------------------------------------
# Aggregated children (Column/Beam/Roof child Solids)
# ---------------------------------------------------------------------------

def _is_spatially_contained(product) -> bool:
    """True when ``product`` sits in an ``IfcRelContainedInSpatialStructure``.

    **Not the discriminator** — see :func:`_is_grouping_parent`. Under
    IFC 4.3's *Spatial Containment* rule a decomposition part has
    ``ContainedInStructure`` NIL, and so does an aggregate member, so this
    predicate now returns ``False`` for both and separates nothing. It
    survives as a cheap "was this already picked up by the containment walk?"
    test for the LEGACY shape (the emission, still in stored models).
    """
    return bool(getattr(product, "ContainedInStructure", None))


def _is_grouping_parent(parent, members) -> bool:
    """**The discriminator between the two things ``IfcRelAggregates`` means.**

    ``IfcRelAggregates`` carries one relation for two situations we must tell
    apart on import:

    * *Aggregated geometry of ONE element* — a window's mitered frame members,
      a wall's detail solids. Fold them in as CHILDREN of the host.
    * *An authored grouping of peer products* — ``proj.aggregate()``'s facade,
      and equally a third-party ``IfcElementAssembly``. Report it as an
      :class:`ExtractedGrouping`, with the members staying top-level elements. It discriminated on **spatial containment**: peers were contained,
    decomposition parts were not. That was retired, and it is worth being
    exact about why, because the honest answer is uncomfortable: **IFC does
    not distinguish these two cases at all.** The schema says so in as many
    words — *"Any subtype of IfcElement can be an element assembly, with
    IfcElementAssembly as a special focus subtype"* — and under the
    conformant shape both the facade's members and a window's frame members
    have ``ContainedInStructure`` NIL. There is no relation, no flag and no
    entity type that tells a facade of walls from a wall of parts. Waiting for
    one is waiting for something that is not coming.

    So the discriminator moves off the relationship and onto two facts about
    the parent that survive the shape change. **Both are required.**

    1. **The parent owns no geometry** — ``Representation is None``. With a
       body, the parent's own solid is what stands in the model and its parts
       refine something that already exists, so emitting them as elements too
       would double-count. Fold them in.

    2. **The parts are the same KIND of thing as the whole** — every member's
       IFC class equals the parent's, or the parent is an
       ``IfcElementAssembly`` (the schema's own focus subtype for "a whole
       made of peer parts", where mixed member classes are the point).

    Condition 2 is not decoration, and leaving it out is not theoretical: it
    was measured. **This compiler already emits body-less parents for ordinary
    single elements** — a DSL ``Column``/``Roof`` becomes an ``IfcColumn`` /
    ``IfcRoof`` with no ``Representation`` whose geometry hangs off it as
    aggregated ``IfcBuildingElementProxy`` children. On condition 1 alone
    every column in every model imported as a *grouping*, so a carport came
    back as five anonymous proxies and zero columns —
    ``test_integration.py::test_element_names_preserved`` caught it. What
    tells that from a facade is that a column is not made of columns:
    ``proj.aggregate()`` DERIVES the parent's class from its members and
    refuses mixed classes (§3, ``IfcWall = IfcWall + IfcWall``), so for every
    file this compiler writes, parent class == member class, always.

    Deliberately NOT used as a signal: whether the members carry a ``Name``.
    It separates today's fixtures perfectly and is a trap — a DSL author
    naming the ``Box`` inside a ``Column`` would silently flip that column
    into a grouping, from an authoring choice with nothing to do with it.

    This is strictly better founded than the containment test it replaces: it
    measures the failure mode directly rather than by proxy, it is stable
    under BOTH emission shapes (so files written before that still import
    identically), and it does not depend on a distinction the schema declines
    to make.

    What it costs: a third-party ``IfcElementAssembly`` with no own body now
    comes back as ``proj.aggregate()`` rather than as a host element with
    children. That is the right answer — it IS the same construct — and it
    needs no marker in the file to survive the round trip, because the
    members keep their own names, classes and world placements either way.
    """
    if parent is None or not members:
        return False
    if parent.is_a() in _SPATIAL_AGGREGATE_PARENTS:
        return False                      # the spatial hierarchy itself
    if not parent.is_a("IfcProduct"):
        return False                      # nothing a product walk could reach
    if getattr(parent, "Representation", None) is not None:
        return False                      # carries geometry: one element's parts
    if parent.is_a("IfcElementAssembly"):
        return True                       # the schema's own name for this
    same_class = all(m.is_a() == parent.is_a() for m in members)
    if not same_class:
        logger.debug(
            "IfcRelAggregates %s: body-less %s over %s — parts of one element, "
            "not a grouping of peers", parent.GlobalId, parent.is_a(),
            sorted({m.is_a() for m in members}),
        )
    return same_class


def _grouping_parents(model) -> Dict[int, Any]:
    """``{entity id of a grouping parent: its IfcRelAggregates}``.

    Computed ONCE, before any walk, and read by all of them — the element
    walk, the aggregated-children walk and :func:`_extract_groupings` each ask
    the same object the same question. Three walks each re-deriving "is this a
    grouping?" is how they start to disagree, and a disagreement here means
    either a model with no walls or a model with its walls twice.
    """
    out: Dict[int, Any] = {}
    for rel in model.by_type("IfcRelAggregates"):
        parent = rel.RelatingObject
        members = [m for m in (rel.RelatedObjects or []) if m is not None]
        if not _is_grouping_parent(parent, members):
            continue
        if parent.id() in out:
            logger.warning(
                "IfcRelAggregates: %s (%s) is the parent of TWO aggregations; "
                "IFC allows one Decomposes per product and readers disagree "
                "about the second — keeping the first, ignoring %s",
                parent.Name, parent.GlobalId, rel.GlobalId,
            )
            continue
        out[parent.id()] = rel
    return out


def _extract_aggregated_children(
    product,
    storey_idx: int,
    storey_elevation: int,
    grouping_parents: Dict[int, Any],
    seen: Optional[Set[int]] = None,
) -> List[ExtractedElement]:
    """Extract children linked via IfcRelAggregates, to any depth.

    Only *decomposition parts* — see :func:`_is_grouping_parent`. The entry
    call is only ever reached for a product that HAS a ``Representation`` (a
    grouping parent is never extracted as an element), so by the discriminator
    its children are parts of a body that already exists.

    **The walk recurses.** ``IfcRelAggregates`` nests, and this compiler emits
    nesting: an ``Element`` anchored into a container becomes its own product
    that aggregates its own solids (``generator._create_element_generic``), so
    a depth-2 authoring tree is a depth-2 IFC tree. A one-level walk exported
    that file correctly and re-imported it with the grandchildren gone and no
    diagnostic — ``ExtractedElement.children`` already nested, only the walk
    did not. Depth is therefore taken from the file, not assumed.

    **The discriminator is re-run at EVERY level.** ``grouping_parents`` is the
    model-wide decision from :func:`_grouping_parents`, and a child of a
    decomposition part can be a grouping parent in its own right — a
    third-party ``IfcElementAssembly`` hanging off an element is exactly that
    shape. Consulting the map only at the top would fold such a parent (and,
    through it, its peer members) into its grandparent's geometry, while
    :func:`_extract_groupings` reports the same aggregation as a grouping and
    ``normalize_ifc_full`` lifts the members as top-level elements — the model
    would come back holding its members twice, once nested and once loose. So
    the check belongs on the child, at each level, not on the root.

    ``seen`` carries the entity ids already folded in on this walk and makes
    the recursion total. A cyclic ``IfcRelAggregates`` is not something this
    compiler can write, but this parses THIRD-PARTY files, and the walk had no
    guard at all; without one, A-aggregates-B-aggregates-A recurses until
    Python's stack gives out, and ``_try_extract_element``'s ``except
    Exception`` turns that into the whole root element silently vanishing.
    ``Decomposes`` is ``[0:1]`` in the schema, so a product legitimately has at
    most one aggregation parent: a second visit is always a malformed file and
    is refused loudly rather than duplicated.

    The spatial-containment skip below is retained for the LEGACY shape: files
    written before that put the members in storeys, so a contained child
    here is a peer the containment walk already extracted, and nesting it as
    well emitted it twice.
    """
    children: List[ExtractedElement] = []

    if not hasattr(product, "IsDecomposedBy"):
        return children

    if seen is None:
        # The root is in the set from the start, or a cycle back onto it
        # terminates one level too late and nests the root under its own part.
        seen = {product.id()}

    for rel in product.IsDecomposedBy:
        for child in rel.RelatedObjects:
            if not child.is_a("IfcProduct"):
                continue
            if child.id() in seen:
                logger.warning(
                    "IfcRelAggregates %s: %s (%s) is already an ancestor or "
                    "sibling in this decomposition — cyclic or multiply-"
                    "parented aggregation, which IFC forbids (Decomposes is "
                    "[0:1]); the branch is dropped rather than walked",
                    rel.GlobalId, child.is_a(), child.GlobalId,
                )
                continue
            if child.id() in grouping_parents:
                logger.debug(
                    "aggregated child %s (%s) is a grouping parent — reported "
                    "as a grouping with its members as peer elements, not "
                    "folded into %s", child.GlobalId, child.is_a(),
                    product.GlobalId,
                )
                continue
            if _is_spatially_contained(child):
                logger.debug(
                    "aggregated child %s is spatially contained — a peer "
                    "product, not a decomposition part; left to the "
                    "containment walk", child.GlobalId,
                )
                continue
            seen.add(child.id())
            abs_mat = _resolve_placement(child)
            geom = _classify_geometry(child, abs_mat)
            placement_info = _matrix_to_placement_info(abs_mat)

            grandchildren = _extract_openings(child)
            grandchildren.extend(_extract_aggregated_children(
                child, storey_idx, storey_elevation, grouping_parents, seen))

            children.append(ExtractedElement(
                ifc_type=child.is_a(),
                name=child.Name or f"child_{child.GlobalId[:8]}",
                guid=child.GlobalId,
                storey_idx=storey_idx,
                storey_elevation=storey_elevation,
                placement=placement_info,
                geometry=geom,
                children=grandchildren,
            ))

    return children


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def normalize_ifc(ifc_content: str) -> List[ExtractedElement]:
    """Backward-compatible wrapper — returns just the element list."""
    return normalize_ifc_full(ifc_content).elements


def _detect_schema(ifc_content: str) -> str:
    """Extract FILE_SCHEMA from IFC STEP header."""
    m = re.search(r"FILE_SCHEMA\s*\(\s*\(\s*'([^']+)'\s*\)\s*\)", ifc_content)
    return m.group(1).upper() if m else "UNKNOWN"


def _rewrite_schema_header(ifc_content: str, original: str, target: str) -> str:
    """Rewrite FILE_SCHEMA header to a supported schema."""
    return ifc_content.replace(f"'{original}'", f"'{target}'", 1)


def normalize_ifc_full(ifc_content: str) -> NormalizationResult:
    """
    Parse an IFC file and return normalized elements with metadata.

    Walks the spatial hierarchy, resolves placements, classifies geometry,
    and converts coordinates to DSL conventions (Z-up, mm integers).

    Supports IFC2X3, IFC4, and attempts best-effort parsing of IFC2X2
    and IFC4X3 by rewriting schema headers.
    """
    if not IFC_AVAILABLE:
        # Plan §3.D E.5: user-uploaded IFC — parse failure is user input, not infra.
        logger.warning("ifcopenshell not available")
        return NormalizationResult(elements=[], parse_error="ifcopenshell not available")

    schema = _detect_schema(ifc_content)
    schema_rewritten = False

    # Rewrite unsupported schema headers to nearest supported version
    if schema in _SCHEMA_REWRITE:
        target = _SCHEMA_REWRITE[schema]
        ifc_content = _rewrite_schema_header(ifc_content, schema, target)
        schema_rewritten = True
        logger.info("Rewrote schema %s → %s for parsing", schema, target)

    # Write to temp file for ifcopenshell
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ifc")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(ifc_content)
        model = ifcopenshell.open(tmp_path)
    except Exception as e:
        # Plan §3.D E.5: user-uploaded IFC — parse failure is user input, not infra.
        logger.warning("Failed to parse IFC: %s", e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return NormalizationResult(
            elements=[], schema=schema, schema_rewritten=schema_rewritten,
            parse_error=str(e),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # --- Build storey map ---
    storeys = model.by_type("IfcBuildingStorey")
    storey_info = []
    for s in storeys:
        elev = 0.0
        if s.ObjectPlacement:
            mat = _resolve_placement(s)
            elev = mat[2, 3]  # Z in IFC = elevation
        storey_info.append((s, elev))

    storey_info.sort(key=lambda x: x[1])

    storey_map = {}  # IfcBuildingStorey id -> (idx, elevation_mm)
    for idx, (s, elev) in enumerate(storey_info):
        elev_mm = int(round(elev * 1000))
        storey_map[s.id()] = (idx, elev_mm)

    # --- Walk containment relations (primary path) ---
    elements = []
    seen_guids = set()
    skipped = Counter()

    # Decided once, read by every walk below — see :func:`_is_grouping_parent`.
    grouping_parents = _grouping_parents(model)
    #: ``{grouping parent entity id: (storey_idx, storey_elev_mm)}`` — where a
    #: member falls back to when its own elevation matches no storey.
    parent_storeys: Dict[int, Tuple[int, int]] = {}
    #: How much the containment walk got out of the file. Gates the fallback
    #: below, and it counts grouping parents even though they yield no
    #: element: a facade whose parent is the ONLY contained product left
    #: ``elements`` empty, fired the fallback, and swept in every un-contained
    #: IfcAnnotation drawing — a model that gained two "elements" purely by
    #: being aggregated.
    contained_yield = 0

    for rel in model.by_type("IfcRelContainedInSpatialStructure"):
        container = rel.RelatingStructure
        if container is None:
            continue

        if container.is_a("IfcBuildingStorey"):
            s_idx, s_elev = storey_map.get(container.id(), (0, 0))
        else:
            # Elements in IfcBuilding, IfcSite, etc. — virtual storey 0
            s_idx, s_elev = 0, 0

        for product in rel.RelatedElements:
            if not product.is_a("IfcProduct"):
                continue
            guid = product.GlobalId
            if guid in seen_guids:
                continue
            seen_guids.add(guid)

            if product.id() in grouping_parents:
                # IFC 4.3 puts the WHOLE in the storey, so the
                # containment walk now reaches the body-less grouping parent.
                # Extracting it would fabricate a body for a thing that has
                # none — the measured 100 mm placeholder box — and swallow its
                # members as children. It is reported as a grouping instead.
                parent_storeys[product.id()] = (s_idx, s_elev)
                contained_yield += 1
                continue

            extracted = _try_extract_element(product, s_idx, s_elev, skipped,
                                             grouping_parents)
            if extracted is not None:
                elements.append(extracted)
                contained_yield += 1

    # --- Fallback: walk all IfcProduct if containment found nothing ---
    if not contained_yield:
        for product in model.by_type("IfcProduct"):
            if product.is_a() in SPATIAL_TYPES:
                continue
            if product.id() in grouping_parents:
                continue          # body-less grouping parent — never an element
            guid = product.GlobalId
            if guid in seen_guids:
                continue
            seen_guids.add(guid)

            extracted = _try_extract_element(product, 0, 0, skipped,
                                             grouping_parents)
            if extracted is not None:
                elements.append(extracted)

    # --- Members of a grouping parent (IFC 4.3: ContainedInStructure NIL) ---
    # Under the conformant shape the members are in no containment relation at
    # all, so neither walk above can see them. They are PEER elements, not
    # decomposition parts, so they are lifted here rather than nested under
    # the parent — which is what makes the DSL tree survive the round trip:
    # ``storey.add(wall)`` twice plus a ``proj.aggregate()``, exactly what was
    # authored. Files written before that contained their members, so
    # those are already in ``seen_guids`` and this loop is a no-op for them.
    #
    # AFTER the fallback on purpose: a file whose only containment relation is
    # the facade's would otherwise leave ``elements`` non-empty and suppress
    # the fallback, silently dropping every un-contained product that is NOT
    # an aggregate member.
    for parent_id, rel in grouping_parents.items():
        fallback = parent_storeys.get(parent_id)
        for member in (rel.RelatedObjects or []):
            if member is None or not member.is_a("IfcProduct"):
                continue
            if member.GlobalId in seen_guids:
                continue
            seen_guids.add(member.GlobalId)
            s_idx, s_elev = _storey_for_member(
                member, storey_info, storey_map, fallback)
            extracted = _try_extract_element(member, s_idx, s_elev, skipped,
                                             grouping_parents)
            if extracted is not None:
                elements.append(extracted)
            else:
                logger.warning(
                    "aggregate member %s (%s) yielded no element — the "
                    "grouping will list a member the model does not contain",
                    member.Name, member.GlobalId,
                )

    # --- Spatial elements with geometry (Site topography, Space volumes) ---
    # IfcSite and IfcSpace stay in SPATIAL_TYPES so the generic product
    # fallback walk above keeps skipping them as bare containers. But
    # third-party exports do attach terrain meshes to IfcSite and
    # volume geometry to IfcSpace — lift those into the elements list
    # when they carry a non-trivial Representation. Site uses
    # storey_idx=-1 as the "site-level" sentinel; Space inherits its
    # parent storey via the IsDecomposedBy / Decomposes chain.
    for site in model.by_type("IfcSite"):
        guid = site.GlobalId
        if guid in seen_guids:
            continue
        if site.Representation is None:
            continue
        seen_guids.add(guid)
        extracted = _try_extract_element(site, -1, 0, skipped, grouping_parents)
        if extracted is not None and extracted.geometry.mode != "unknown":
            elements.append(extracted)

    for space in model.by_type("IfcSpace"):
        guid = space.GlobalId
        if guid in seen_guids:
            continue
        if space.Representation is None:
            continue
        seen_guids.add(guid)
        s_idx, s_elev = _find_containing_storey(space, storey_map)
        extracted = _try_extract_element(space, s_idx, s_elev, skipped,
                                         grouping_parents)
        if extracted is not None and extracted.geometry.mode != "unknown":
            elements.append(extracted)

    # --- Project-level groupings (aggregates + zones) ---
    groupings = _extract_groupings(model, grouping_parents)

    total_found = len(seen_guids)
    logger.info(
        "Normalized %d elements from IFC (%d total products, %d skipped, "
        "%d groupings)",
        len(elements), total_found, sum(skipped.values()), len(groupings),
    )

    return NormalizationResult(
        elements=elements,
        total_products_found=total_found,
        skipped_types=dict(skipped),
        schema=schema,
        schema_rewritten=schema_rewritten,
        groupings=groupings,
    )


# ---------------------------------------------------------------------------
# Project-level groupings — aggregates of PEER products, and zones
# ---------------------------------------------------------------------------

#: Classes that ARE the spatial hierarchy. ``IfcRelAggregates`` builds that
#: hierarchy too (Project -> Site -> Building -> Storey), and those relations
#: are structure rather than an authored grouping.
_SPATIAL_AGGREGATE_PARENTS = frozenset({
    "IfcProject", "IfcSite", "IfcBuilding", "IfcBuildingStorey",
})


def _extract_groupings(
    model, grouping_parents: Optional[Dict[int, Any]] = None,
) -> List[ExtractedGrouping]:
    """Recover ``proj.aggregate()`` / ``proj.zone()`` statements from a file.

    Two shapes, one list, because roadmap §V.7 settled that they are one idea:

    * **Peer aggregate** — an ``IfcRelAggregates`` whose ``RelatingObject``
      is a non-spatial product carrying NO ``Representation``. That single
      fact is the whole discriminator; :func:`_is_grouping_parent` explains
      why it replaced the containment test and why nothing else can do
      the job.
    * **Zone** — an ``IfcZone`` and the ``IfcRelAssignsToGroup`` that lists
      its spaces. ``IfcZone`` is an ``IfcGroup``, not an ``IfcProduct``, so
      no product walk could ever have seen it.

    Returned SEPARATELY from ``elements`` rather than folded into it: a
    grouping parent has no geometry, so a reconstructor treating it as an
    element would have to invent a body for it (the measured "fabricated
    100 mm placeholder box") and would double-count its members.

    ``grouping_parents`` is passed in by :func:`normalize_ifc_full` so the
    element walk and this pass read the identical decision; it is recomputed
    when this function is called on its own (tests, callers with only a
    model).
    """
    groupings: List[ExtractedGrouping] = []
    if grouping_parents is None:
        grouping_parents = _grouping_parents(model)

    for rel in grouping_parents.values():
        parent = rel.RelatingObject
        members = [m for m in (rel.RelatedObjects or []) if m is not None]

        groupings.append(ExtractedGrouping(
            kind="aggregate",
            name=parent.Name or f"aggregate_{parent.GlobalId[:8]}",
            guid=parent.GlobalId,
            ifc_type=parent.is_a(),
            members=[m.Name or f"elem_{m.GlobalId[:8]}" for m in members],
            member_guids=[m.GlobalId for m in members],
            properties=_extract_properties(parent),
        ))

    for zone in model.by_type("IfcZone"):
        members = []
        for rel in (getattr(zone, "IsGroupedBy", None) or []):
            members.extend(m for m in (rel.RelatedObjects or []) if m is not None)
        groupings.append(ExtractedGrouping(
            kind="zone",
            name=zone.Name or f"zone_{zone.GlobalId[:8]}",
            guid=zone.GlobalId,
            ifc_type=zone.is_a(),
            members=[m.Name or f"elem_{m.GlobalId[:8]}" for m in members],
            member_guids=[m.GlobalId for m in members],
            properties=_extract_properties(zone),
        ))

    return groupings


def _storey_for_member(
    member, storey_info, storey_map, fallback: Optional[Tuple[int, int]]
) -> Tuple[int, int]:
    """The storey an un-contained aggregate member belongs to.

    IFC 4.3 puts the containment on the WHOLE, so the file does not record
        which storey each member of a multi-storey facade was authored in. It does
    still record where the member IS: the storey whose elevation the member's
    own base sits at or above is the one it stands in, which is the same rule
    ``IfcBuildingStorey`` states for itself.

    Recovering it matters for identity, not geometry — the placement is
    resolved to absolute world coordinates either way — but a member filed in
    the wrong storey re-compiles under a different canonical name, and a
    changed canonical name is a changed manifest key and a changed patch atom.

    ``fallback`` is the parent's own container: what a member below every
    storey elevation gets, with a warning, rather than a guess dressed up as
    an answer.
    """
    try:
        z = float(_resolve_placement(member)[2, 3])
    except Exception as exc:                              # pragma: no cover
        logger.warning("aggregate member %s: placement unresolvable (%s) — "
                       "filing it in the parent's storey", member.GlobalId, exc)
        return fallback or (0, 0)

    found = None
    for storey, elevation in storey_info:                 # ascending elevation
        if z + 1e-6 >= elevation:
            found = storey_map.get(storey.id())
        else:
            break
    if found is None:
        logger.warning(
            "aggregate member %s sits at z=%.3f, below every storey elevation "
            "— filing it in %s", member.GlobalId, z,
            "the parent's storey" if fallback else "storey 0",
        )
        return fallback or (0, 0)
    return found


def _find_containing_storey(
    product, storey_map: Dict[int, Tuple[int, int]]
) -> Tuple[int, int]:
    """Walk Decomposes chain to find the enclosing IfcBuildingStorey.

    IfcSpace is usually aggregated under a storey via IfcRelAggregates.
    Returns (storey_idx, storey_elev_mm) — defaults to (0, 0) when no
    storey is found, which is the same fallback the primary walk uses.

    **This is NOT one of the peer-aggregation consumers, despite reading like
    one** (roadmap §3 names it as a third). A containment-first check was
    written here and then removed as unreachable: its only caller passes an
    ``IfcSpace``, and ``IfcSpace`` is an ``IfcSpatialElement`` — the
    ``ContainedInStructure`` inverse is declared on ``IfcElement`` and does
    not exist on it in IFC4X3_ADD2, so there is nothing for containment-first
    to read. A space also cannot be a peer-aggregate member here
    (:mod:`lite_step.ifc.groupings` refuses aggregating spatial-structure
    classes and points at ``proj.zone()``), so this walk never climbs to a
    grouping parent. Do not re-add the check without a caller that passes an
    ``IfcElement``.
    """
    parent = product
    # Bound the walk so a circular reference can't hang us.
    for _ in range(16):
        if not hasattr(parent, "Decomposes") or not parent.Decomposes:
            break
        # Decomposes is an inverse relation; pick the first relating object.
        rel = parent.Decomposes[0]
        if not hasattr(rel, "RelatingObject") or rel.RelatingObject is None:
            break
        parent = rel.RelatingObject
        if parent.is_a("IfcBuildingStorey"):
            return storey_map.get(parent.id(), (0, 0))
    return (0, 0)


def _try_extract_element(
    product, storey_idx: int, storey_elev: int, skipped: Counter,
    grouping_parents: Dict[int, Any],
) -> Optional[ExtractedElement]:
    """Try to extract a single product. Returns None and counts skip if not mappable.

    ``grouping_parents`` is REQUIRED rather than defaulted, because the
    aggregation walk below needs it at every level and a ``None`` default would
    spell "silently fold grouping parents into their host" — the exact silent
    loss this signature exists to prevent. Every caller is in
    :func:`normalize_ifc_full`, which has already computed it once.
    """
    try:
        abs_mat = _resolve_placement(product)
        geom = _classify_geometry(product, abs_mat)
        placement_info = _matrix_to_placement_info(abs_mat)

        children = _extract_openings(product)
        children.extend(
            _extract_aggregated_children(
                product, storey_idx, storey_elev, grouping_parents)
        )

        return ExtractedElement(
            ifc_type=product.is_a(),
            name=product.Name or f"elem_{product.GlobalId[:8]}",
            guid=product.GlobalId,
            storey_idx=storey_idx,
            storey_elevation=storey_elev,
            placement=placement_info,
            geometry=geom,
            children=children,
            properties=_extract_properties(product),
        )
    except Exception as exc:
        ifc_type = product.is_a() if hasattr(product, "is_a") else "Unknown"
        skipped[ifc_type] += 1
        logger.debug("Skipped %s: %s", ifc_type, exc)
        return None


def _extract_properties(product) -> Dict[str, Any]:
    """Extract IFC PropertySet values as a flat dict."""
    props = {}
    if not hasattr(product, "IsDefinedBy"):
        return props

    for rel in product.IsDefinedBy:
        if not rel.is_a("IfcRelDefinesByProperties"):
            continue
        pset = rel.RelatingPropertyDefinition
        if not pset.is_a("IfcPropertySet"):
            continue
        # Skip our own metadata
        if pset.Name == "LITESTEP_META":
            continue
        for prop in pset.HasProperties:
            if hasattr(prop, "NominalValue") and prop.NominalValue is not None:
                val = prop.NominalValue
                props[prop.Name] = val.wrappedValue if hasattr(val, "wrappedValue") else str(val)

    return props
