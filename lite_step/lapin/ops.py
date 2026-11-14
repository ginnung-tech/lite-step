"""A compiled Lite-STEP IFC as a list of Lapin operations.

The input is the IFC the compiler wrote, not the Python model: the file
already holds every solid placed and every carve as a boolean, and each
element's ``Name`` is its canonical name. Reading it means the operations
describe exactly what was emitted.

One operation per element, of one of two kinds:

``primitive``
    A box or a cylinder standing on the XY plane, turned about Z only. In
    Lapin this is a Studio primitive: the user can resize and move it.
    ``tools`` names the elements subtracted from it (Lapin keeps a subtract
    as a live relation between two elements, and the tool stays in the
    model, as a carving element does in Lite-STEP).

``body``
    Anything else, as a triangle mesh in world coordinates with every carve
    already applied. In Lapin this is a script-owned body: it can be moved,
    scheduled and exported, and reshaped only by the script.

An element is a ``primitive`` with tools only when every cutter in its
boolean chain is itself an uncarved primitive element of the model, to
within :data:`MATCH_TOLERANCE`. Lapin subtracts elements, and a mesh does
not cut there, so anything else falls back to ``body``. A fallback loses
editability, never shape.

All lengths are metres, all angles degrees: Lapin's units.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

FORMAT = "lite-step.lapin-ops"
VERSION = 1

#: Two solids are the same solid when every number differs by less than this (metres).
MATCH_TOLERANCE = 0.001
#: Numbers are rounded to this many decimals (0.1 mm) so a hash does not move with float noise.
_DECIMALS = 4
_PARALLEL = 1e-6


@dataclass(frozen=True)
class Op:
    """One element as Lapin makes it."""

    path: str
    kind: str  # "primitive" | "body"
    ifc_class: str
    #: The storey's name (``None`` when it has none) and its elevation in metres. Lapin's
    #: storey is found by the name, else by the elevation.
    storey: Optional[str]
    elevation: Optional[float]
    args: dict
    tools: tuple = ()

    @property
    def hash(self) -> str:
        """Changes when, and only when, what Lapin must build changes."""
        payload = [self.kind, self.ifc_class, self.storey, self.elevation, self.args, list(self.tools)]
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {
            "path": self.path, "kind": self.kind, "ifcClass": self.ifc_class, "storey": self.storey,
            "elevation": self.elevation, "args": self.args, "tools": list(self.tools), "hash": self.hash,
        }


@dataclass
class LapinOps:
    """The operations for one model, with what could not be mapped."""

    ops: list = field(default_factory=list)
    #: ``(name, elevation in metres)`` for every storey that holds an element, lowest first.
    storeys: list = field(default_factory=list)
    #: ``(path, reason)`` for every element left out. Never silent: the CLI prints these.
    skipped: list = field(default_factory=list)
    #: Things worth telling the author that are not errors (a name made up, a fallback's reason).
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        """How many elements became what. The first thing to read about a model."""
        primitives = [o for o in self.ops if o.kind == "primitive"]
        return {
            "elements": len(self.ops),
            "primitive": len(primitives),
            "primitive_carved": sum(1 for o in primitives if o.tools),
            "body": sum(1 for o in self.ops if o.kind == "body"),
            "skipped": len(self.skipped),
        }

    def to_json(self) -> str:
        return json.dumps({
            "format": FORMAT, "version": VERSION,
            "storeys": [{"name": n, "elevation": e} for n, e in self.storeys],
            "ops": [o.to_dict() for o in self.ops],
            "skipped": [{"path": p, "reason": r} for p, r in self.skipped],
            "notes": list(self.notes),
        }, separators=(",", ":"))


def ops_from_json(text: str) -> LapinOps:
    """The list :meth:`LapinOps.to_json` wrote. Refuses another format or a newer version."""
    data = json.loads(text)
    if data.get("format") != FORMAT:
        raise ValueError(f"not a {FORMAT} document: format={data.get('format')!r}")
    if data.get("version") != VERSION:
        raise ValueError(f"{FORMAT} version {data.get('version')!r} is not version {VERSION}")
    ops = []
    for o in data["ops"]:
        op = Op(o["path"], o["kind"], o["ifcClass"], o["storey"], o["elevation"], o["args"], tuple(o["tools"]))
        if op.hash != o["hash"]:
            raise ValueError(f"{op.path}: the hash does not match the operation (edited by hand?)")
        ops.append(op)
    return LapinOps(
        ops=ops,
        storeys=[(s["name"], s["elevation"]) for s in data["storeys"]],
        skipped=[(s["path"], s["reason"]) for s in data["skipped"]],
        notes=list(data["notes"]),
    )


# --------------------------------------------------------------------------- solids


@dataclass(frozen=True)
class _Solid:
    """A box or cylinder as Lapin's ``studio.primitive`` takes it."""

    shape: str
    at: tuple  # the middle of the base
    sizes: tuple  # box: width, length, height; cylinder: radius, height
    rotation: float  # degrees about Z, in [0, 90) for a box, 0 for a cylinder

    def numbers(self) -> tuple:
        return (*self.at, *self.sizes)

    def same_as(self, other: "_Solid") -> bool:
        if self.shape != other.shape:
            return False
        if any(abs(a - b) >= MATCH_TOLERANCE for a, b in zip(self.numbers(), other.numbers())):
            return False
        # An angle's effect grows with the solid: compare the sweep at its far corner.
        reach = max(self.sizes[:2]) if self.shape == "box" else 0.0
        turn = abs(self.rotation - other.rotation) % 90.0
        return math.radians(min(turn, 90.0 - turn)) * reach < MATCH_TOLERANCE

    def args(self) -> dict:
        r = lambda v: round(float(v), _DECIMALS) + 0.0  # noqa: E731 - "+ 0.0" turns -0.0 into 0.0
        out = {"shape": self.shape, "at": [r(v) for v in self.at], "rotation": round(float(self.rotation), 6) + 0.0}
        if self.shape == "box":
            out.update(width=r(self.sizes[0]), length=r(self.sizes[1]), height=r(self.sizes[2]))
        else:
            out.update(radius=r(self.sizes[0]), height=r(self.sizes[1]))
        return out


def _placement(entity) -> np.ndarray:
    import ifcopenshell.util.placement as placement

    return np.eye(4) if entity is None else np.array(placement.get_axis2placement(entity), dtype=float)


def _points(m: np.ndarray, pts: Iterable) -> np.ndarray:
    pts = np.array([[*p, 1.0] for p in pts], dtype=float)
    return (m @ pts.T).T[:, :3]


def _polygon(curve) -> Optional[list]:
    """A closed curve's corners when it is made of straight segments only, else ``None``."""
    if curve.is_a("IfcPolyline"):
        pts = [tuple(p.Coordinates) for p in curve.Points]
    elif curve.is_a("IfcIndexedPolyCurve"):
        if any(not s.is_a("IfcLineIndex") for s in curve.Segments or ()):
            return None
        coords = curve.Points.CoordList
        if curve.Segments:
            order = [i for s in curve.Segments for i in s.wrappedValue]
            pts = [tuple(coords[i - 1]) for i in order]
        else:
            pts = [tuple(c) for c in coords]
    else:
        return None
    out = []
    for p in pts:
        p = (float(p[0]), float(p[1]))
        if not out or math.dist(out[-1], p) > 1e-9:
            out.append(p)
    if len(out) > 1 and math.dist(out[0], out[-1]) <= 1e-9:
        out.pop()
    return out


def _rectangle(profile) -> Optional[list]:
    """A profile's four corners in its solid's XY plane when it is a plain rectangle."""
    if profile.is_a() == "IfcRectangleProfileDef":
        x, y = profile.XDim / 2.0, profile.YDim / 2.0
        corners = [(-x, -y, 0.0), (x, -y, 0.0), (x, y, 0.0), (-x, y, 0.0)]
        return [tuple(p) for p in _points(_placement(profile.Position), corners)]
    if profile.is_a() == "IfcArbitraryClosedProfileDef":
        pts = _polygon(profile.OuterCurve)
        if pts is None or len(pts) != 4:
            return None
        a, b, c, d = (np.array(p) for p in pts)
        square = abs(np.dot(b - a, d - a)) <= _PARALLEL * np.linalg.norm(b - a) * np.linalg.norm(d - a)
        closed = np.linalg.norm((a + c) - (b + d)) <= 1e-9
        return [(p[0], p[1], 0.0) for p in pts] if square and closed else None
    return None


def _box(origin: np.ndarray, edges: list) -> Optional[_Solid]:
    """Three edges from one corner as a Lapin box, or ``None`` when it does not stand on Z."""
    lengths = [float(np.linalg.norm(e)) for e in edges]
    if min(lengths) < 1e-9:
        return None
    units = [e / n for e, n in zip(edges, lengths)]
    for i in range(3):
        for j in range(i + 1, 3):
            if abs(float(np.dot(units[i], units[j]))) > _PARALLEL:
                return None  # sheared by a transform: not a box
    up = [i for i in range(3) if abs(abs(units[i][2]) - 1.0) < _PARALLEL]
    if len(up) != 1:
        return None
    k = up[0]
    flat = [i for i in range(3) if i != k]
    centre = origin + sum(edges) / 2.0
    # The box is its own mirror through both flat axes, so either axis may be "width":
    # take the one whose direction lies in [0, 90) degrees, which makes the answer unique.
    angles = [math.degrees(math.atan2(units[i][1], units[i][0])) % 180.0 for i in flat]
    w = 0 if angles[0] < 90.0 - 1e-9 else 1
    rotation = angles[w]
    if rotation > 90.0 - 1e-9:
        rotation = 0.0
        w = 1 - w
    return _Solid(
        "box",
        (float(centre[0]), float(centre[1]), float(centre[2] - lengths[k] / 2.0)),
        (lengths[flat[w]], lengths[flat[1 - w]], lengths[k]),
        rotation,
    )


def _solid(item, world: np.ndarray, scale: float) -> Optional[_Solid]:
    """A representation item as a Lapin primitive, or ``None``."""
    if item.is_a() == "IfcBlock":
        m = world @ _placement(item.Position)
        x, y, z = item.XLength, item.YLength, item.ZLength
        o, a, b, c = _points(m, [(0, 0, 0), (x, 0, 0), (0, y, 0), (0, 0, z)]) * scale
        return _box(o, [a - o, b - o, c - o])
    if item.is_a() != "IfcExtrudedAreaSolid":
        return None
    direction = np.array(item.ExtrudedDirection.DirectionRatios, dtype=float)
    direction = direction / np.linalg.norm(direction)
    if abs(abs(direction[2]) - 1.0) > _PARALLEL:
        return None  # an oblique extrusion is a sheared prism
    m = world @ _placement(item.Position)
    lift = (0.0, 0.0, float(item.Depth) * (1.0 if direction[2] > 0 else -1.0))
    profile = item.SweptArea
    corners = _rectangle(profile)
    if corners is not None:
        o, a, _, b = _points(m, corners) * scale
        top = _points(m, [tuple(np.array(corners[0]) + lift)])[0] * scale
        return _box(o, [a - o, b - o, top - o])
    if profile.is_a() == "IfcCircleProfileDef":
        centre = _points(_placement(profile.Position), [(0.0, 0.0, 0.0)])[0]
        base, top, rim = _points(m, [centre, tuple(centre + lift), tuple(centre + (profile.Radius, 0.0, 0.0))]) * scale
        axis = top - base
        height = float(np.linalg.norm(axis))
        if height < 1e-9 or abs(abs(axis[2]) / height - 1.0) > _PARALLEL:
            return None
        low = base if base[2] <= top[2] else top
        return _Solid("cylinder", tuple(float(v) for v in low), (float(np.linalg.norm(rim - base)), height), 0.0)
    return None


def _chain(item) -> Optional[tuple]:
    """``(base item, [cutters])`` for a chain of differences; ``None`` for any other boolean."""
    cutters = []
    while item.is_a("IfcBooleanResult"):
        if item.Operator != "DIFFERENCE":
            return None
        cutters.append(item.SecondOperand)
        item = item.FirstOperand
    return item, cutters


# --------------------------------------------------------------------------- the model


def _body_item(product) -> Optional[tuple]:
    """``(item, transform)`` for an element whose body is one item, mapped or not."""
    import ifcopenshell.util.placement as placement

    bodies = [r for r in product.Representation.Representations if r.RepresentationIdentifier == "Body"]
    if len(bodies) != 1 or len(bodies[0].Items) != 1:
        return None
    item = bodies[0].Items[0]
    transform = np.eye(4)
    while item.is_a("IfcMappedItem"):
        inner = item.MappingSource.MappedRepresentation.Items
        if len(inner) != 1:
            return None
        transform = transform @ np.array(placement.get_mappeditem_transformation(item), dtype=float)
        item = inner[0]
    return item, transform


def _storey_of(product):
    import ifcopenshell.util.element as element

    container = element.get_container(product)
    seen = set()
    while container is not None and container.id() not in seen:
        if container.is_a("IfcBuildingStorey"):
            return container
        seen.add(container.id())
        container = element.get_aggregate(container) or element.get_container(container)
    return None


def _named(products: list, notes: list) -> list:
    """Each product with the name it is tracked by. A name must be there, and unique."""
    used: dict = {}
    unnamed: dict = {}
    out = []
    for p in products:
        name = (p.Name or "").strip()
        if not name:
            n = unnamed[p.is_a()] = unnamed.get(p.is_a(), 0) + 1
            name = f"unnamed#{n}:{p.is_a()}"
            notes.append(f"{name}: the element has no name; this one is its place in the file, "
                         f"so it moves when elements before it are added or removed")
        count = used[name] = used.get(name, 0) + 1
        if count > 1:
            notes.append(f"{name}: {count} elements share this name; the later ones are tracked as {name}#N")
            name = f"{name}#{count}"
        out.append((name, p))
    return out


def _meshes(model, products: list, openings: bool) -> dict:
    """``{product id: (vertices, faces)}`` in world metres, every boolean applied.

    ``openings`` says whether ``IfcOpeningElement`` voids are subtracted. It is on only
    for elements that have openings of their own: ifcopenshell also cuts an element with
    the openings of whatever it is a part of, so a window's frame, aggregated under the
    wall it fills, would lose the half that sits inside the wall's opening.
    """
    import multiprocessing

    import ifcopenshell.geom

    if not products:
        return {}
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    settings.set("weld-vertices", True)
    settings.set("disable-opening-subtractions", not openings)
    out = {}
    threads = max(1, min(4, multiprocessing.cpu_count() // 2))
    iterator = ifcopenshell.geom.iterator(settings, model, threads, include=products)
    if not iterator.initialize():
        return out
    while True:
        shape = iterator.get()
        verts = np.round(np.array(shape.geometry.verts, dtype=float).reshape(-1, 3), _DECIMALS) + 0.0
        faces = np.array(shape.geometry.faces, dtype=int).reshape(-1, 3)
        if len(faces):
            out[shape.id] = (verts.tolist(), faces.tolist())
        if not iterator.next():
            return out


def ops_from_ifc(model: Any) -> LapinOps:
    """The Lapin operations for a compiled model.

    ``model`` is an ``ifcopenshell.file``, or a path to one. The caller keeps
    the file open for as long as this runs.
    """
    import ifcopenshell
    import ifcopenshell.util.placement as placement
    import ifcopenshell.util.unit as unit

    if not isinstance(model, ifcopenshell.file):
        model = ifcopenshell.open(str(model))
    scale = float(unit.calculate_unit_scale(model))
    result = LapinOps()

    products = [
        p for p in model.by_type("IfcElement")
        if not p.is_a("IfcFeatureElement") and p.Representation is not None
        and any(r.RepresentationIdentifier == "Body" for r in p.Representation.Representations)
    ]
    named = _named(products, result.notes)

    # Pass 1: what each element is on its own (its base solid and what is cut from it).
    plain: dict = {}  # path -> _Solid, for elements that are an uncarved primitive
    drafts = []
    for path, p in named:
        base = cutters = None
        found = None if p.HasOpenings else _body_item(p)
        if found is not None and p.ObjectPlacement is not None:
            item, transform = found
            world = np.array(placement.get_local_placement(p.ObjectPlacement), dtype=float) @ transform
            chain = _chain(item)
            if chain is not None:
                base = _solid(chain[0], world, scale)
                cutters = [(_solid(c, world, scale), c) for c in chain[1]]
        drafts.append((path, p, base, cutters))
        if base is not None and not cutters:
            plain[path] = base

    # Pass 2: a carved element stays a primitive only if every cutter is a plain element.
    meshed = []
    ops: dict = {}
    for path, p, base, cutters in drafts:
        tools = []
        why = None
        if base is None:
            why = "its shape is not a box or cylinder standing on Z"
        else:
            for solid, entity in cutters:
                match = None if solid is None else next(
                    (other for other, s in plain.items() if other != path and s.same_as(solid)), None)
                if match is None:
                    why = (f"a cutter ({entity.is_a()}) is not an uncarved box or cylinder element of the model")
                    break
                if match not in tools:
                    tools.append(match)
        storey = _storey_of(p)
        where = (None, None)
        if storey is not None:
            where = (storey.Name, round(float(storey.Elevation or 0.0) * scale, _DECIMALS) + 0.0)
            if where not in result.storeys:
                result.storeys.append(where)
        if why is None:
            ops[path] = Op(path, "primitive", p.is_a(), *where, base.args(), tuple(tools))
        else:
            meshed.append((path, p, where))
            if base is not None:
                result.notes.append(f"{path}: sent as a body because {why}")

    meshes = _meshes(model, [p for _, p, _ in meshed if p.HasOpenings], openings=True)
    meshes.update(_meshes(model, [p for _, p, _ in meshed if not p.HasOpenings], openings=False))
    for path, p, where in meshed:
        mesh = meshes.get(p.id())
        if mesh is None:
            result.skipped.append((path, "ifcopenshell produced no geometry for it"))
            continue
        ops[path] = Op(path, "body", p.is_a(), *where, {"vertices": mesh[0], "faces": mesh[1]})

    result.ops = [ops[path] for path, *_ in drafts if path in ops]
    result.storeys.sort(key=lambda s: (s[1], s[0] or ""))  # lowest first
    return result
