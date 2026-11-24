"""The model as triangles, each knowing its element, for the drone to look at.

Built once per model file (`ifcopenshell.geom`, world coordinates, metres) and cached by the file's sha256,
because tessellating takes far longer than looking. Left out on purpose: voids (`IfcOpeningElement`),
air volumes (`IfcSpace`) and annotations: a camera must see the structure, not the room it encloses.
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

SKIPPED_CLASSES = ("IfcOpeningElement", "IfcSpace", "IfcAnnotation", "IfcGrid", "IfcVirtualElement")
_CACHE_VERSION = 1


@dataclass(frozen=True)
class Element:
    index: int
    guid: str
    name: str | None
    ifc_class: str

    @property
    def label(self) -> str:
        """The canonical name when there is one, else the GlobalId."""
        return self.name or self.guid


@dataclass
class Scene:
    vertices: np.ndarray  # (V, 3) float32, metres
    triangles: np.ndarray  # (T, 3) int32
    tri_element: np.ndarray  # (T,) int32 -> elements
    elements: list[Element]
    parent: list[int] = field(default_factory=list)  # aggregation parent, -1 at the top

    def __post_init__(self) -> None:
        if not self.parent:
            self.parent = [-1] * len(self.elements)
        if len(self.parent) != len(self.elements):
            raise ValueError("every element needs a parent entry")
        if len(self.triangles) != len(self.tri_element):
            raise ValueError("every triangle needs exactly one element")
        if len(self.triangles) and int(self.tri_element.max()) >= len(self.elements):
            raise ValueError("a triangle points at an element that does not exist")

    # -- building ----------------------------------------------------------------------------------

    @classmethod
    def from_meshes(cls, meshes: Iterable[tuple[str, str | None, str, Any, Any]]) -> Scene:
        """From `(guid, name, ifc_class, vertices (N,3), faces (M,3))`. Meshes sharing a guid are one element."""
        verts, tris, owner, elements, by_guid, offset = [], [], [], [], {}, 0
        for guid, name, ifc_class, v, f in meshes:
            v = np.asarray(v, dtype=np.float32).reshape(-1, 3)
            f = np.asarray(f, dtype=np.int32).reshape(-1, 3)
            if len(f) == 0:
                continue
            idx = by_guid.get(guid)
            if idx is None:
                idx = by_guid[guid] = len(elements)
                elements.append(Element(idx, guid, name, ifc_class))
            verts.append(v)
            tris.append(f + offset)
            owner.append(np.full(len(f), idx, dtype=np.int32))
            offset += len(v)
        if not elements:
            return cls(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32), np.zeros(0, np.int32), [])
        return cls(np.concatenate(verts), np.concatenate(tris), np.concatenate(owner), elements)

    @classmethod
    def from_ifc(cls, path: str | Path, cache_dir: str | Path | None = None) -> Scene:
        """Tessellate an IFC file; with `cache_dir`, reuse the scene cached for the same bytes."""
        path = Path(path)
        cache = None
        if cache_dir is not None:
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            cache = Path(cache_dir) / f"scene-v{_CACHE_VERSION}-{sha}.npz"
            if cache.exists():
                try:
                    return cls.load(cache)
                except (OSError, ValueError, KeyError) as exc:
                    logger.warning("[survey.scene] cached scene unreadable, rebuilding: %s (%s)", cache, exc)
        import ifcopenshell
        import ifcopenshell.geom

        f = ifcopenshell.open(str(path))
        settings = ifcopenshell.geom.settings()
        settings.set("use-world-coords", True)
        meshes, shaped, skipped = [], [], 0
        it = ifcopenshell.geom.iterator(settings, f, max(1, (os.cpu_count() or 2) - 1))
        if it.initialize():
            while True:
                shape = it.get()
                product = f.by_id(shape.id)
                if any(product.is_a(c) for c in SKIPPED_CLASSES):
                    skipped += 1
                else:
                    g = shape.geometry
                    shaped.append(product)
                    meshes.append((product.GlobalId, getattr(product, "Name", None), product.is_a(),
                                   np.asarray(g.verts, dtype=np.float32), np.asarray(g.faces, dtype=np.int32)))
                if not it.next():
                    break
        scene = cls.from_meshes(meshes)
        scene.attach_parents(shaped, _aggregation_parent)
        logger.info("[survey.scene] %s: %d elements, %d triangles, %d skipped", path.name, len(scene.elements),
                    len(scene.triangles), skipped)
        if cache is not None:
            try:
                scene.save(cache)
            except OSError as exc:
                logger.warning("[survey.scene] could not cache the scene: %s (%s)", cache, exc)
        return scene

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez_compressed(tmp, vertices=self.vertices, triangles=self.triangles, tri_element=self.tri_element,
                            guids=np.array([e.guid for e in self.elements]),
                            names=np.array([e.name or "" for e in self.elements]),
                            classes=np.array([e.ifc_class for e in self.elements]),
                            parent=np.asarray(self.parent, dtype=np.int32))
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> Scene:
        with np.load(path, allow_pickle=False) as d:
            els = [Element(i, str(g), str(n) or None, str(c)) for i, (g, n, c) in enumerate(zip(d["guids"], d["names"], d["classes"]))]
            return cls(d["vertices"], d["triangles"], d["tri_element"], els, [int(x) for x in d["parent"]])

    # -- structure -----------------------------------------------------------------------------------

    def attach_parents(self, products: Iterable[Any], parent_of: Callable[[Any], Any | None]) -> None:
        """Add every aggregation level above the shaped products as elements, linked to their parents."""
        index = {e.guid: e.index for e in self.elements}
        parent = list(self.parent)
        for product in products:
            child = index.get(product.GlobalId)
            node, seen = product, {product.id()}
            while child is not None:
                up = parent_of(node)
                if up is None or up.id() in seen:
                    break
                seen.add(up.id())
                if up.GlobalId not in index:
                    index[up.GlobalId] = len(self.elements)
                    self.elements.append(Element(len(self.elements), up.GlobalId, getattr(up, "Name", None), up.is_a()))
                    parent.append(-1)
                if parent[child] != -1:
                    break
                parent[child] = index[up.GlobalId]
                child, node = index[up.GlobalId], up
        self.parent = parent

    def ancestors(self, index: int) -> list[int]:
        out, p = [], self.parent[index]
        while p >= 0 and p not in out:
            out.append(p)
            p = self.parent[p]
        return out

    def group_of(self, index: int, depth: int) -> int:
        """The level an element is reported at: negative counts up from the element (-1 itself, -2 what it
        is a part of: a member of a crossbow), positive counts down from the top."""
        chain = [index, *self.ancestors(index)]
        if depth < 0:
            return chain[min(-depth - 1, len(chain) - 1)]
        return chain[::-1][min(depth, len(chain) - 1)]

    def bounds_of(self, element_indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds of the triangles of the given elements and everything below them."""
        want = set(int(i) for i in element_indices)
        rows = [i for i in range(len(self.elements)) if i in want or want.intersection(self.ancestors(i))]
        mask = np.isin(self.tri_element, np.asarray(rows, dtype=np.int32))
        if not mask.any():
            raise ValueError("no triangles for those elements")
        pts = self.vertices[np.unique(self.triangles[mask])]
        return pts.min(axis=0).astype(np.float64), pts.max(axis=0).astype(np.float64)


def _aggregation_parent(e: Any) -> Any | None:
    for rel in getattr(e, "Decomposes", None) or []:
        if rel.is_a("IfcRelAggregates") and rel.RelatingObject.is_a("IfcElement"):
            return rel.RelatingObject
    return None
