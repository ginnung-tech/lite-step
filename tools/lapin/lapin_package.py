"""Build a `.lapin` (Format 3) package from a compiled Lite-STEP model.

Step 1 of the Lite-STEP -> Lapin parametric pipeline:
- Emits valid Format 3 `.lapin` ZIP packages (`manifest.json`, `project/document.json`,
  `model/<module>/<class>/<storey>.arrow`, `model/relations.arrow`).
- Extracts spatial hierarchy and triangulated element bodies (`LapinSharedBody`)
  with stable IFC GlobalIds matching Lite-STEP's `stable_ids`.
- Embeds `param()` metadata in `source/parametric.json` (`foreign`) and a starter
  `LapinScript` (`scripts/<model>_params.py`) so Lapin keeps the parameter contract
  while Step 2+ adds cascading pure-Python mesh adaptation.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import pprint
import struct
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zipfile

from .arrow_ipc import ArrowColumn, write_arrow_file

_FORMAT = "lapin-project"
_VERSION = 3
_APP = "lite-step 0.1.0"
_PRODUCT = "lapin"

# Tested Lapin app versions (`engine.hello` -> `version`): Format 3 is Lapin's
# internal save format without a public stability contract, so untested
# versions are refused loudly unless checked/added here.
TESTED_LAPIN_VERSIONS: Tuple[str, ...] = ("0.1.1238", "1.0.0")


class UnsupportedLapinVersionError(ValueError):
    """Raised when the target Lapin version has not been tested for Format 3."""


def validate_lapin_version(version: Optional[str]) -> str:
    """Ensure ``version`` is a Lapin version tested against this Format 3 writer."""
    normalized = (version or "").strip()
    if normalized not in TESTED_LAPIN_VERSIONS:
        tested = ", ".join(TESTED_LAPIN_VERSIONS)
        found = normalized or "unknown"
        raise UnsupportedLapinVersionError(
            f"Lapin version {found!r} is not tested with Lite-STEP's internal Format 3 .lapin writer "
            f"(tested: {tested}). Pass format=ifc / --format ifc instead."
        )
    return normalized

_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_$"
_LAPIN_NS = uuid.UUID("6f2c1a8e-5d3b-4c71-9a0e-2b8f4e7d1c55")

_IDENTITY_XF = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

# Class -> Lapin module mapping (unknown classes fall back to "core").
_CLASS_MODULES: Dict[str, str] = {
    "IfcWall": "architecture",
    "IfcWallStandardCase": "architecture",
    "IfcSlab": "architecture",
    "IfcDoor": "architecture",
    "IfcWindow": "architecture",
    "IfcRoof": "architecture",
    "IfcCurtainWall": "architecture",
    "IfcStair": "architecture",
    "IfcStairFlight": "architecture",
    "IfcRailing": "architecture",
    "IfcCovering": "architecture",
    "IfcBeam": "structure",
    "IfcColumn": "structure",
    "IfcMember": "structure",
    "IfcFooting": "structure",
    "IfcPile": "structure",
    "IfcPlate": "structure",
    "LapinScript": "scripting",
}


def _module_of(ifc_class: str) -> str:
    return _CLASS_MODULES.get(ifc_class, "core")


def _compress_guid(guid: uuid.UUID) -> str:
    n = guid.int
    out = [_ALPHABET[n >> 126]]
    for k in range(1, 22):
        out.append(_ALPHABET[(n >> (126 - 6 * k)) & 63])
    return "".join(out)


def _global_id(name: str, namespace: str = "lite-step") -> str:
    ns = uuid.uuid5(_LAPIN_NS, namespace)
    return _compress_guid(uuid.uuid5(ns, name))


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_segment(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_")


@dataclass(frozen=True)
class _Entity:
    id: str
    cls: str
    name: str
    predefined_type: Optional[str]
    attrs: Dict[str, Any]
    storey_id: Optional[str] = None


@dataclass(frozen=True)
class _Relation:
    id: str
    cls: str
    relating: str
    related: List[str]
    attrs: Optional[Dict[str, Any]] = None


def _encode_stored_body(
    verts: Sequence[float],
    faces: Sequence[int],
    color: str = "#c9ced6",
    part_name: str = "Body",
    material_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Encode triangles into Lapin's `StoredBodies` base64 layout (`v64`/`t64`)."""
    rounded = [round(float(v), 4) for v in verts]
    v_bytes = struct.pack(f"<{len(rounded)}d", *rounded) if rounded else b""
    t_ints = [int(i) for i in faces]
    t_bytes = struct.pack(f"<{len(t_ints)}i", *t_ints) if t_ints else b""
    part: Dict[str, Any] = {
        "name": part_name,
        "color": color.lower(),
        "v64": base64.b64encode(v_bytes).decode("ascii"),
        "t64": base64.b64encode(t_bytes).decode("ascii"),
    }
    if material_id:
        part["material"] = material_id
    stored = {"parts": [part]}
    canonical = json.dumps(stored, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return stored, _sha256_hex(canonical)


def _extract_style(shape_geometry) -> Tuple[str, Optional[float], Optional[str]]:
    """Return ``(hex_color, transparency, style_name)`` for the first shape material."""
    color = "#c9ced6"
    transparency: Optional[float] = None
    style_name: Optional[str] = None
    try:
        mats = getattr(shape_geometry, "materials", None) or ()
        if not mats:
            return color, transparency, style_name
        mat = mats[0]
        diffuse = getattr(mat, "diffuse", None)
        if diffuse is not None:
            r = max(0, min(255, int(round(diffuse.r() * 255))))
            g = max(0, min(255, int(round(diffuse.g() * 255))))
            b = max(0, min(255, int(round(diffuse.b() * 255))))
            color = f"#{r:02x}{g:02x}{b:02x}"
        raw_trans = getattr(mat, "transparency", None)
        if raw_trans is not None:
            val = float(raw_trans)
            if math.isfinite(val) and 0.001 < val < 1.0:
                transparency = round(val, 3)
        raw_name = getattr(mat, "name", None)
        if isinstance(raw_name, str) and raw_name and raw_name != "DefaultMaterial":
            style_name = raw_name
    except (AttributeError, TypeError, ValueError):
        pass
    return color, transparency, style_name


def _arrow_kind(values: Sequence[Any]) -> str:
    present = [v for v in values if v is not None]
    if not present:
        return "Json"
    if all(isinstance(v, bool) for v in present):
        return "Bool"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in present):
        return "Number"
    if all(isinstance(v, str) for v in present):
        return "Text"
    if all(
        isinstance(v, list)
        and all(isinstance(n, (int, float)) and not isinstance(n, bool) for n in v)
        for v in present
    ):
        return "Numbers"
    return "Json"


def _attr_column(key: str, entities: Sequence[_Entity]) -> ArrowColumn:
    vals = [e.attrs.get(key) for e in entities]
    present_null = any(key in e.attrs and e.attrs[key] is None for e in entities)
    kind = "Json" if present_null else _arrow_kind(vals)
    meta = (("lapin.kind", kind),)
    col_name = f"attr:{key}"
    if kind == "Number":
        return ArrowColumn(col_name, "float64", [None if v is None else float(v) for v in vals], nullable=True, metadata=meta)
    if kind == "Bool":
        return ArrowColumn(col_name, "bool", [None if v is None else bool(v) for v in vals], nullable=True, metadata=meta)
    if kind == "Numbers":
        return ArrowColumn(col_name, "list_float64", [None if v is None else [float(x) for x in v] for v in vals], nullable=True, metadata=meta)
    if kind == "Text":
        return ArrowColumn(col_name, "utf8", [None if v is None else str(v) for v in vals], nullable=True, metadata=meta)
    serialized = [
        None if key not in e.attrs else ("null" if e.attrs[key] is None else json.dumps(e.attrs[key], separators=(",", ":"), ensure_ascii=False))
        for e in entities
    ]
    return ArrowColumn(col_name, "utf8", serialized, nullable=True, metadata=meta)


def _serialize_entities(entities: Sequence[_Entity], order_of: Dict[str, int]) -> bytes:
    keys = sorted({k for e in entities for k in e.attrs.keys()})
    columns: List[ArrowColumn] = [
        ArrowColumn("id", "utf8", [e.id for e in entities], nullable=False),
        ArrowColumn("class", "utf8", [e.cls for e in entities], nullable=False),
        ArrowColumn("name", "utf8", [e.name for e in entities], nullable=False),
        ArrowColumn("predefinedType", "utf8", [e.predefined_type for e in entities], nullable=True),
        ArrowColumn("order", "int32", [order_of[e.id] for e in entities], nullable=False),
    ]
    columns.extend(_attr_column(key, entities) for key in keys)
    return write_arrow_file(columns, len(entities))


def _serialize_relations(relations: Sequence[_Relation], order_of: Dict[str, int]) -> bytes:
    columns = [
        ArrowColumn("id", "utf8", [r.id for r in relations], nullable=False),
        ArrowColumn("class", "utf8", [r.cls for r in relations], nullable=False),
        ArrowColumn("relating", "utf8", [r.relating for r in relations], nullable=False),
        ArrowColumn("related", "list_utf8", [r.related for r in relations], nullable=False),
        ArrowColumn(
            "attrs",
            "utf8",
            [None if r.attrs is None else json.dumps(r.attrs, separators=(",", ":"), ensure_ascii=False) for r in relations],
            nullable=True,
        ),
        ArrowColumn("order", "int32", [order_of[r.id] for r in relations], nullable=False),
    ]
    return write_arrow_file(columns, len(relations))


def _starter_script_code(model_name: str, params_meta: List[Dict[str, Any]]) -> str:
    payload = pprint.pformat(params_meta, indent=2, sort_dicts=False)
    return (
        f'"""Lite-STEP parametric metadata for `{model_name}`.\n\n'
        "Generated by `lite-step lapin push` (Step 1). Pure-Python cascading geometry\n"
        "evaluation and `script.watch` hooks are added in the next step.\n"
        '"""\n\n'
        f"MODEL = {model_name!r}\n"
        f"PARAMS = {payload}\n"
    )


class _PackageBuilder:
    """Stateful accumulator for building a Format 3 `.lapin` entity/relation model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.entities: List[_Entity] = []
        self.relations: List[_Relation] = []
        self.used_ids: set[str] = set()
        self.shared_by_hash: Dict[str, str] = {}
        self.materials_by_style: Dict[Tuple[str, float, str], str] = {}

    def unique_id(self, candidate: Optional[str], fallback_key: str) -> str:
        if candidate and len(candidate) == 22 and candidate not in self.used_ids:
            self.used_ids.add(candidate)
            return candidate
        gid = _global_id(f"{self.model_name}:{fallback_key}")
        while gid in self.used_ids:
            fallback_key += "#"
            gid = _global_id(f"{self.model_name}:{fallback_key}")
        self.used_ids.add(gid)
        return gid

    def add_rel(self, cls: str, relating: str, related: List[str], candidate_id: Optional[str] = None) -> None:
        rid = self.unique_id(candidate_id, f"rel:{cls}:{relating}:{','.join(related)}")
        self.relations.append(_Relation(id=rid, cls=cls, relating=relating, related=related))

    def ensure_transparent_material(self, color: str, transparency: Optional[float], style_name: Optional[str]) -> Optional[str]:
        if transparency is None:
            return None
        name = style_name or f"IFC {color[1:].upper()} {round(100 * (1 - transparency))}%"
        key = (color.lower(), transparency, name)
        if key in self.materials_by_style:
            return self.materials_by_style[key]
        mat_id = self.unique_id(None, f"material:{key[0]}:{key[1]}:{key[2]}")
        self.materials_by_style[key] = mat_id
        self.entities.append(_Entity(
            id=mat_id,
            cls="IfcMaterial",
            name=name,
            predefined_type=None,
            attrs={"color": color.lower(), "transparency": transparency},
        ))
        return mat_id

    def add_spatial_hierarchy(self, ifc: Any, unit_scale: float) -> Tuple[str, Dict[int, str], List[str]]:
        proj_ifc = next(iter(ifc.by_type("IfcProject")), None)
        proj_id = self.unique_id(proj_ifc.GlobalId if proj_ifc else None, "project")
        proj_name = proj_ifc.Name if proj_ifc and proj_ifc.Name else self.model_name
        self.entities.append(_Entity(proj_id, "IfcProject", proj_name, None, {"lengthUnit": "METRE"}))

        site_ifc = next(iter(ifc.by_type("IfcSite")), None)
        site_id = self.unique_id(site_ifc.GlobalId if site_ifc else None, "site")
        self.entities.append(_Entity(site_id, "IfcSite", site_ifc.Name if site_ifc and site_ifc.Name else "Site", None, {}))
        self.add_rel("IfcRelAggregates", proj_id, [site_id])

        bld_ifc = next(iter(ifc.by_type("IfcBuilding")), None)
        bld_id = self.unique_id(bld_ifc.GlobalId if bld_ifc else None, "building")
        self.entities.append(_Entity(bld_id, "IfcBuilding", bld_ifc.Name if bld_ifc and bld_ifc.Name else "Building", None, {}))
        self.add_rel("IfcRelAggregates", site_id, [bld_id])

        storey_map, storey_ids = self._add_storeys(ifc, unit_scale)
        self.add_rel("IfcRelAggregates", bld_id, storey_ids)
        return proj_id, storey_map, storey_ids

    def _add_storeys(self, ifc: Any, unit_scale: float) -> Tuple[Dict[int, str], List[str]]:
        ifc_storeys = sorted(
            ifc.by_type("IfcBuildingStorey"),
            key=lambda s: float(getattr(s, "Elevation", 0.0) or 0.0),
        )
        if not ifc_storeys:
            sid = self.unique_id(None, "storey:Ground floor")
            self.entities.append(_Entity(sid, "IfcBuildingStorey", "Ground floor", None, {"elevation": 0.0, "height": 3.0}))
            return {}, [sid]

        elevations = [round(float(getattr(s, "Elevation", 0.0) or 0.0) * unit_scale, 6) for s in ifc_storeys]
        storey_map: Dict[int, str] = {}
        storey_ids: List[str] = []
        for i, storey in enumerate(ifc_storeys):
            sid = self.unique_id(storey.GlobalId, f"storey:{storey.Name or i}")
            elev = elevations[i]
            height = round(elevations[i + 1] - elev, 6) if i + 1 < len(elevations) else 3.0
            if height <= 0:
                height = 3.0
            self.entities.append(_Entity(sid, "IfcBuildingStorey", storey.Name or f"Level {i}", None, {"elevation": elev, "height": height}))
            storey_map[storey.id()] = sid
            storey_ids.append(sid)
        return storey_map, storey_ids

    def collect_products(self, ifc: Any, storey_map: Dict[int, str], storey_ids: Sequence[str]) -> int:
        import ifcopenshell.geom
        import ifcopenshell.util.element

        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        iterator = ifcopenshell.geom.iterator(settings, ifc, max(1, (os.cpu_count() or 2) - 1))

        contained_by_storey: Dict[str, List[str]] = {sid: [] for sid in storey_ids}
        seen_keys: Dict[str, int] = {}
        element_count = 0
        fallback_storey_id = storey_ids[0]

        if iterator.initialize():
            while True:
                shape = iterator.get()
                product = ifc.by_id(shape.id)
                verts = list(shape.geometry.verts)
                faces = list(shape.geometry.faces)
                if verts and faces and product.is_a("IfcElement") and not product.is_a("IfcOpeningElement"):
                    elem_id, storey_id = self._append_product(
                        product, shape.geometry, verts, faces,
                        ifcopenshell.util.element.get_container(product),
                        storey_map, fallback_storey_id, seen_keys,
                    )
                    contained_by_storey[storey_id].append(elem_id)
                    element_count += 1
                if not iterator.next():
                    break

        for sid, elem_ids in contained_by_storey.items():
            if elem_ids:
                self.add_rel("IfcRelContainedInSpatialStructure", sid, elem_ids)
        return element_count

    def _append_product(
        self,
        product: Any,
        shape_geometry: Any,
        verts: List[float],
        faces: List[int],
        container: Any,
        storey_map: Dict[int, str],
        fallback_storey_id: str,
        seen_keys: Dict[str, int],
    ) -> Tuple[str, str]:
        cls = product.is_a()
        name = product.Name or cls[3:]
        predefined = getattr(product, "PredefinedType", None)
        storey_id = storey_map[container.id()] if container is not None and container.id() in storey_map else fallback_storey_id

        key = f"{storey_id}/{cls}/{name}"
        seen_keys[key] = seen_keys.get(key, 0) + 1
        if seen_keys[key] > 1:
            key = f"{key}#{seen_keys[key]}"

        color, transparency, style_name = _extract_style(shape_geometry)
        mat_id = self.ensure_transparent_material(color, transparency, style_name)
        stored, content_hash = _encode_stored_body(verts, faces, color=color, material_id=mat_id)
        shared_id = self.shared_by_hash.get(content_hash)
        if shared_id is None:
            shared_id = self.unique_id(None, f"shared:{content_hash}")
            self.shared_by_hash[content_hash] = shared_id
            self.entities.append(_Entity(
                id=shared_id,
                cls="LapinSharedBody",
                name="Shared body",
                predefined_type=None,
                attrs={"body": stored, "contentHash": content_hash},
            ))

        elem_id = self.unique_id(getattr(product, "GlobalId", None), f"elem:{key}")
        self.entities.append(_Entity(
            id=elem_id,
            cls=cls,
            name=name,
            predefined_type=predefined if isinstance(predefined, str) else None,
            attrs={
                "bodyOf": shared_id,
                "xf": list(_IDENTITY_XF),
                "owner": f"lite-step:{self.model_name}",
                "externalAttrs": {"liteStep": key},
            },
            storey_id=storey_id,
        ))
        return elem_id, storey_id


def _extract_param_script(
    builder: _PackageBuilder,
    source_path: Optional[Path],
    proj_id: str,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, bytes, str]]]:
    params_meta: List[Dict[str, Any]] = []
    if source_path is not None and source_path.suffix.lower() == ".py" and source_path.is_file():
        from lite_step.params import read_params

        for p in read_params(source_path.read_text(encoding="utf-8")):
            if p.editable:
                params_meta.append({
                    "name": p.name,
                    "value": p.value,
                    "unit": p.unit,
                    "label": p.label,
                    "line": p.line,
                    "target": proj_id,
                })

    script_texts: List[Tuple[str, bytes, str]] = []
    if params_meta:
        script_name = f"{_safe_segment(builder.model_name)}_params.py"
        script_code = _starter_script_code(builder.model_name, params_meta)
        script_bytes = script_code.encode("utf-8")
        script_id = builder.unique_id(None, f"script:{script_name}")
        builder.entities.append(_Entity(
            id=script_id,
            cls="LapinScript",
            name=script_name,
            predefined_type=None,
            attrs={
                "code": script_code,
                "language": "python",
                "sha256": _sha256_hex(script_bytes),
                "description": f"Lite-STEP parameters for {builder.model_name}",
            },
        ))
        script_texts.append((f"scripts/{script_name}", script_bytes, "scripting"))
    return params_meta, script_texts


def _write_lapin_zip(
    output_path: Path,
    manifest: Dict[str, Any],
    parts: List[Tuple[Dict[str, Any], bytes]],
    foreign_entries: List[Tuple[Dict[str, Any], bytes]],
) -> None:
    manifest_bytes = json.dumps(manifest, indent=2, ensure_ascii=False).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".saving")
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", manifest_bytes)
            for part_meta, part_bytes in parts:
                zf.writestr(part_meta["path"], part_bytes)
            for foreign_meta, foreign_bytes in foreign_entries:
                zf.writestr(foreign_meta["path"], foreign_bytes)
        os.replace(temp_path, output_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def build_lapin_package(
    ifc_path: Path,
    output_path: Path,
    source_path: Optional[Path] = None,
    lapin_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a compiled IFC4X3 file (and optional Lite-STEP `.py` source) to a Format 3 `.lapin` zip."""
    import ifcopenshell
    import ifcopenshell.util.unit

    if lapin_version is not None:
        validate_lapin_version(lapin_version)
    print(
        f"warning: writing Lapin internal Format {_VERSION} (.lapin) directly "
        f"(tested with Lapin {', '.join(TESTED_LAPIN_VERSIONS)}); "
        "Format 3 has no stability contract for external writers — use format=ifc for stable interchange",
        file=sys.stderr,
    )

    ifc = ifcopenshell.open(str(ifc_path))
    model_name = source_path.stem if source_path is not None else ifc_path.stem
    unit_scale = float(ifcopenshell.util.unit.calculate_unit_scale(ifc) or 1.0)

    builder = _PackageBuilder(model_name)
    proj_id, storey_map, storey_ids = builder.add_spatial_hierarchy(ifc, unit_scale)
    element_count = builder.collect_products(ifc, storey_map, storey_ids)
    params_meta, script_texts = _extract_param_script(builder, source_path, proj_id)

    order_of: Dict[str, int] = {e.id: idx for idx, e in enumerate(builder.entities)}
    for idx, rel in enumerate(builder.relations):
        order_of[rel.id] = idx

    parts: List[Tuple[Dict[str, Any], bytes]] = []
    classes: Dict[str, str] = {}

    doc_bytes = b"{}"
    parts.append(({"path": "project/document.json", "sha256": _sha256_hex(doc_bytes), "size": len(doc_bytes)}, doc_bytes))

    grouped: Dict[Tuple[str, str, str], List[_Entity]] = {}
    for entity in builder.entities:
        mod = _module_of(entity.cls)
        classes[entity.cls] = mod
        grouped.setdefault((mod, entity.cls, entity.storey_id or "_"), []).append(entity)

    for (mod, cls, storey), chunk in sorted(grouped.items()):
        part_path = f"model/{_safe_segment(mod)}/{_safe_segment(cls)}/{_safe_segment(storey)}.arrow"
        raw = _serialize_entities(chunk, order_of)
        parts.append(({"path": part_path, "sha256": _sha256_hex(raw), "size": len(raw), "module": mod}, raw))

    rel_bytes = _serialize_relations(builder.relations, order_of)
    parts.append(({"path": "model/relations.arrow", "sha256": _sha256_hex(rel_bytes), "size": len(rel_bytes)}, rel_bytes))

    for text_path, text_bytes, mod in sorted(script_texts, key=lambda x: x[0]):
        parts.append(({"path": text_path, "sha256": _sha256_hex(text_bytes), "size": len(text_bytes), "module": mod}, text_bytes))

    foreign_entries: List[Tuple[Dict[str, Any], bytes]] = []
    if params_meta:
        ir_bytes = (json.dumps({"version": 1, "model": model_name, "params": params_meta}, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        foreign_entries.append(({"path": "source/parametric.json", "sha256": _sha256_hex(ir_bytes), "size": len(ir_bytes)}, ir_bytes))

    manifest: Dict[str, Any] = {
        "format": _FORMAT,
        "version": _VERSION,
        "app": _APP,
        "product": _PRODUCT,
        "modules": [{"id": m, "schemaVersion": 1} for m in sorted({*classes.values(), "core"})],
        "parts": [p for p, _ in parts],
        "classes": dict(sorted(classes.items())),
    }
    if foreign_entries:
        manifest["foreign"] = [f for f, _ in foreign_entries]

    _write_lapin_zip(output_path, manifest, parts, foreign_entries)

    return {
        "path": str(output_path),
        "elements": element_count,
        "sharedBodies": len(builder.shared_by_hash),
        "storeys": len(storey_ids),
        "params": len(params_meta),
    }
