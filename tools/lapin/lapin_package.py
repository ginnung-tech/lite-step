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
import os
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zipfile

from arrow_ipc import ArrowColumn, write_arrow_file

_FORMAT = "lapin-project"
_VERSION = 3
_APP = "lite-step 0.1.0"
_PRODUCT = "lapin"

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
) -> Tuple[Dict[str, Any], str]:
    """Encode triangles into Lapin's `StoredBodies` base64 layout (`v64`/`t64`)."""
    rounded = [round(float(v), 4) for v in verts]
    v_bytes = struct.pack(f"<{len(rounded)}d", *rounded) if rounded else b""
    t_ints = [int(i) for i in faces]
    t_bytes = struct.pack(f"<{len(t_ints)}i", *t_ints) if t_ints else b""
    stored = {
        "parts": [
            {
                "name": part_name,
                "color": color.lower(),
                "v64": base64.b64encode(v_bytes).decode("ascii"),
                "t64": base64.b64encode(t_bytes).decode("ascii"),
            }
        ]
    }
    canonical = json.dumps(stored, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return stored, _sha256_hex(canonical)


def _extract_color(shape_geometry) -> str:
    try:
        mats = getattr(shape_geometry, "materials", None) or ()
        if mats:
            m = mats[0]
            diffuse = getattr(m, "diffuse", None)
            if diffuse is not None:
                r = max(0, min(255, int(round(diffuse.r() * 255))))
                g = max(0, min(255, int(round(diffuse.g() * 255))))
                b = max(0, min(255, int(round(diffuse.b() * 255))))
                return f"#{r:02x}{g:02x}{b:02x}"
    except Exception:
        pass
    return "#c9ced6"


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


def _serialize_entities(entities: Sequence[_Entity], order_of: Dict[str, int]) -> bytes:
    keys = sorted({k for e in entities for k in e.attrs.keys()})
    columns: List[ArrowColumn] = [
        ArrowColumn("id", "utf8", [e.id for e in entities], nullable=False),
        ArrowColumn("class", "utf8", [e.cls for e in entities], nullable=False),
        ArrowColumn("name", "utf8", [e.name for e in entities], nullable=False),
        ArrowColumn("predefinedType", "utf8", [e.predefined_type for e in entities], nullable=True),
        ArrowColumn("order", "int32", [order_of[e.id] for e in entities], nullable=False),
    ]
    for key in keys:
        vals = [e.attrs.get(key) for e in entities]
        present_null = any(key in e.attrs and e.attrs[key] is None for e in entities)
        kind = "Json" if present_null else _arrow_kind(vals)
        meta = (("lapin.kind", kind),)
        if kind == "Number":
            columns.append(ArrowColumn(
                f"attr:{key}", "float64",
                [None if v is None else float(v) for v in vals],
                nullable=True, metadata=meta,
            ))
        elif kind == "Bool":
            columns.append(ArrowColumn(
                f"attr:{key}", "bool",
                [None if v is None else bool(v) for v in vals],
                nullable=True, metadata=meta,
            ))
        elif kind == "Numbers":
            columns.append(ArrowColumn(
                f"attr:{key}", "list_float64",
                [None if v is None else [float(x) for x in v] for v in vals],
                nullable=True, metadata=meta,
            ))
        elif kind == "Text":
            columns.append(ArrowColumn(
                f"attr:{key}", "utf8",
                [None if v is None else str(v) for v in vals],
                nullable=True, metadata=meta,
            ))
        else:
            serialized: List[Optional[str]] = []
            for e in entities:
                if key not in e.attrs:
                    serialized.append(None)
                elif e.attrs[key] is None:
                    serialized.append("null")
                else:
                    serialized.append(json.dumps(e.attrs[key], separators=(",", ":"), ensure_ascii=False))
            columns.append(ArrowColumn(f"attr:{key}", "utf8", serialized, nullable=True, metadata=meta))
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
    payload = json.dumps(params_meta, indent=2, ensure_ascii=False)
    return (
        f'"""Lite-STEP parametric metadata for `{model_name}`.\n\n'
        "Generated by `lite-step lapin push` (Step 1). Pure-Python cascading geometry\n"
        "evaluation and `script.watch` hooks are added in the next step.\n"
        '"""\n\n'
        f"MODEL = {model_name!r}\n"
        f"PARAMS = {payload}\n"
    )


def build_lapin_package(
    ifc_path: Path,
    output_path: Path,
    source_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Convert a compiled IFC4X3 file (and optional Lite-STEP `.py` source) to a Format 3 `.lapin` zip."""
    import ifcopenshell
    import ifcopenshell.geom
    import ifcopenshell.util.element
    import ifcopenshell.util.unit

    ifc = ifcopenshell.open(str(ifc_path))
    model_name = source_path.stem if source_path is not None else ifc_path.stem
    unit_scale = float(ifcopenshell.util.unit.calculate_unit_scale(ifc) or 1.0)

    entities: List[_Entity] = []
    relations: List[_Relation] = []
    used_ids: set[str] = set()

    def unique_id(candidate: Optional[str], fallback_key: str) -> str:
        if candidate and len(candidate) == 22 and candidate not in used_ids:
            used_ids.add(candidate)
            return candidate
        gid = _global_id(f"{model_name}:{fallback_key}")
        while gid in used_ids:
            fallback_key += "#"
            gid = _global_id(f"{model_name}:{fallback_key}")
        used_ids.add(gid)
        return gid

    def add_rel(cls: str, relating: str, related: List[str], candidate_id: Optional[str] = None) -> None:
        rid = unique_id(candidate_id, f"rel:{cls}:{relating}:{','.join(related)}")
        relations.append(_Relation(id=rid, cls=cls, relating=relating, related=related))

    # 1. Spatial hierarchy: IfcProject -> IfcSite -> IfcBuilding -> IfcBuildingStorey
    projects = ifc.by_type("IfcProject")
    proj_ifc = projects[0] if projects else None
    proj_id = unique_id(proj_ifc.GlobalId if proj_ifc else None, "project")
    proj_name = (proj_ifc.Name if proj_ifc and proj_ifc.Name else model_name)
    entities.append(_Entity(
        id=proj_id,
        cls="IfcProject",
        name=proj_name,
        predefined_type=None,
        attrs={"lengthUnit": "METRE"},
    ))

    sites = ifc.by_type("IfcSite")
    site_ifc = sites[0] if sites else None
    site_id = unique_id(site_ifc.GlobalId if site_ifc else None, "site")
    entities.append(_Entity(
        id=site_id,
        cls="IfcSite",
        name=(site_ifc.Name if site_ifc and site_ifc.Name else "Site"),
        predefined_type=None,
        attrs={},
    ))
    add_rel("IfcRelAggregates", proj_id, [site_id])

    buildings = ifc.by_type("IfcBuilding")
    bld_ifc = buildings[0] if buildings else None
    bld_id = unique_id(bld_ifc.GlobalId if bld_ifc else None, "building")
    entities.append(_Entity(
        id=bld_id,
        cls="IfcBuilding",
        name=(bld_ifc.Name if bld_ifc and bld_ifc.Name else "Building"),
        predefined_type=None,
        attrs={},
    ))
    add_rel("IfcRelAggregates", site_id, [bld_id])

    storey_map: Dict[int, str] = {}
    ifc_storeys = sorted(
        ifc.by_type("IfcBuildingStorey"),
        key=lambda s: float(getattr(s, "Elevation", 0.0) or 0.0),
    )
    storey_ids: List[str] = []
    if not ifc_storeys:
        default_storey_id = unique_id(None, "storey:Ground floor")
        entities.append(_Entity(
            id=default_storey_id,
            cls="IfcBuildingStorey",
            name="Ground floor",
            predefined_type=None,
            attrs={"elevation": 0.0, "height": 3.0},
        ))
        storey_ids.append(default_storey_id)
    else:
        elevations = [
            round(float(getattr(s, "Elevation", 0.0) or 0.0) * unit_scale, 6)
            for s in ifc_storeys
        ]
        for i, s in enumerate(ifc_storeys):
            sid = unique_id(s.GlobalId, f"storey:{s.Name or i}")
            elev = elevations[i]
            height = round(elevations[i + 1] - elev, 6) if i + 1 < len(elevations) else 3.0
            if height <= 0:
                height = 3.0
            entities.append(_Entity(
                id=sid,
                cls="IfcBuildingStorey",
                name=s.Name or f"Level {i}",
                predefined_type=None,
                attrs={"elevation": elev, "height": height},
            ))
            storey_map[s.id()] = sid
            storey_ids.append(sid)
    add_rel("IfcRelAggregates", bld_id, storey_ids)
    fallback_storey_id = storey_ids[0]

    # 2. Products -> LapinSharedBody + Element entities
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, True)
    iterator = ifcopenshell.geom.iterator(settings, ifc, max(1, (os.cpu_count() or 2) - 1))

    shared_by_hash: Dict[str, str] = {}
    contained_by_storey: Dict[str, List[str]] = {sid: [] for sid in storey_ids}
    seen_keys: Dict[str, int] = {}
    element_count = 0

    if iterator.initialize():
        while True:
            shape = iterator.get()
            product = ifc.by_id(shape.id)
            verts = list(shape.geometry.verts)
            faces = list(shape.geometry.faces)
            if verts and faces and product.is_a("IfcElement"):
                cls = product.is_a()
                name = product.Name or cls[3:]
                predefined = getattr(product, "PredefinedType", None)
                container = ifcopenshell.util.element.get_container(product)
                if container is not None and container.id() in storey_map:
                    storey_id = storey_map[container.id()]
                else:
                    storey_id = fallback_storey_id

                key = f"{storey_id}/{cls}/{name}"
                seen_keys[key] = seen_keys.get(key, 0) + 1
                if seen_keys[key] > 1:
                    key = f"{key}#{seen_keys[key]}"

                color = _extract_color(shape.geometry)
                stored, content_hash = _encode_stored_body(verts, faces, color=color)
                shared_id = shared_by_hash.get(content_hash)
                if shared_id is None:
                    shared_id = unique_id(None, f"shared:{content_hash}")
                    shared_by_hash[content_hash] = shared_id
                    entities.append(_Entity(
                        id=shared_id,
                        cls="LapinSharedBody",
                        name="Shared body",
                        predefined_type=None,
                        attrs={"body": stored, "contentHash": content_hash},
                    ))

                elem_id = unique_id(getattr(product, "GlobalId", None), f"elem:{key}")
                entities.append(_Entity(
                    id=elem_id,
                    cls=cls,
                    name=name,
                    predefined_type=predefined if isinstance(predefined, str) else None,
                    attrs={
                        "bodyOf": shared_id,
                        "xf": list(_IDENTITY_XF),
                        "owner": f"lite-step:{model_name}",
                        "externalAttrs": {"liteStep": key},
                    },
                    storey_id=storey_id,
                ))
                contained_by_storey[storey_id].append(elem_id)
                element_count += 1
            if not iterator.next():
                break

    for sid, elem_ids in contained_by_storey.items():
        if elem_ids:
            add_rel("IfcRelContainedInSpatialStructure", sid, elem_ids)

    # 3. Extract param() declarations -> source/parametric.json + starter LapinScript
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
        script_name = f"{_safe_segment(model_name)}_params.py"
        script_code = _starter_script_code(model_name, params_meta)
        script_bytes = script_code.encode("utf-8")
        script_id = unique_id(None, f"script:{script_name}")
        entities.append(_Entity(
            id=script_id,
            cls="LapinScript",
            name=script_name,
            predefined_type=None,
            attrs={
                "code": script_code,
                "language": "python",
                "sha256": _sha256_hex(script_bytes),
                "description": f"Lite-STEP parameters for {model_name}",
            },
        ))
        script_texts.append((f"scripts/{script_name}", script_bytes, "scripting"))

    # 4. Build Arrow IPC parts & manifest
    order_of: Dict[str, int] = {}
    for idx, e in enumerate(entities):
        order_of[e.id] = idx
    for idx, r in enumerate(relations):
        order_of[r.id] = idx

    parts: List[Tuple[Dict[str, Any], bytes]] = []
    classes: Dict[str, str] = {}

    doc_bytes = b"{}"
    parts.append(({"path": "project/document.json", "sha256": _sha256_hex(doc_bytes), "size": len(doc_bytes)}, doc_bytes))

    grouped: Dict[Tuple[str, str, str], List[_Entity]] = {}
    for e in entities:
        mod = _module_of(e.cls)
        classes[e.cls] = mod
        grouped.setdefault((mod, e.cls, e.storey_id or "_"), []).append(e)

    for (mod, cls, storey), chunk in sorted(grouped.items()):
        part_path = f"model/{_safe_segment(mod)}/{_safe_segment(cls)}/{_safe_segment(storey)}.arrow"
        raw = _serialize_entities(chunk, order_of)
        parts.append((
            {"path": part_path, "sha256": _sha256_hex(raw), "size": len(raw), "module": mod},
            raw,
        ))

    rel_bytes = _serialize_relations(relations, order_of)
    parts.append(({"path": "model/relations.arrow", "sha256": _sha256_hex(rel_bytes), "size": len(rel_bytes)}, rel_bytes))

    for text_path, text_bytes, mod in sorted(script_texts, key=lambda x: x[0]):
        parts.append((
            {"path": text_path, "sha256": _sha256_hex(text_bytes), "size": len(text_bytes), "module": mod},
            text_bytes,
        ))

    foreign_entries: List[Tuple[Dict[str, Any], bytes]] = []
    if params_meta:
        ir_doc = {
            "version": 1,
            "model": model_name,
            "params": params_meta,
        }
        ir_bytes = (json.dumps(ir_doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        foreign_entries.append((
            {"path": "source/parametric.json", "sha256": _sha256_hex(ir_bytes), "size": len(ir_bytes)},
            ir_bytes,
        ))

    modules_list = [{"id": m, "schemaVersion": 1} for m in sorted({*classes.values(), "core"})]
    manifest: Dict[str, Any] = {
        "format": _FORMAT,
        "version": _VERSION,
        "app": _APP,
        "product": _PRODUCT,
        "modules": modules_list,
        "parts": [p for p, _ in parts],
        "classes": dict(sorted(classes.items())),
    }
    if foreign_entries:
        manifest["foreign"] = [f for f, _ in foreign_entries]

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

    return {
        "path": str(output_path),
        "elements": element_count,
        "sharedBodies": len(shared_by_hash),
        "storeys": len(storey_ids),
        "params": len(params_meta),
    }
