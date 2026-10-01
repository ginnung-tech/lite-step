"""
Embed and extract round-trip metadata (LITESTEP_META) in IFC files.

Embeds Lite-STEP source code, element manifest, version, and hash
as an IfcPropertySet attached to IfcBuilding. This enables round-trip
editing: generate IFC → edit in external BIM tool → re-import with
full source recovery.
"""

import base64
import gzip
import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Optional

try:
    import ifcopenshell
    IFC_AVAILABLE = True
except ImportError:
    IFC_AVAILABLE = False
    ifcopenshell = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

LITESTEP_META_PSET_NAME = "LITESTEP_META"

#: The ONE version this compiler writes, and the ONE version it will
#: continue. Bump it in the same PR as any change to what a stored source
#: MEANS — a removed field, a renamed container, a re-based coordinate
#: frame, a verb that refuses what an earlier spelling accepted.
#:
#: There is no ladder of previous versions, deliberately. See
#: :func:`base_version_refusal`.
#: v11 — ``.void()`` emits on every host it is legal on. The same case as
#: v10, in its purest form: a stored source that voids a ``Pipe`` or a nested
#: aggregate still EXECUTES, unchanged, and now compiles to a DIFFERENT file
#: — the hole it always asked for is finally in it. The file moves without
#: the source moving, which is exactly what this gate exists to catch.
#: v12 — an element's FRAME is derived from its own geometry plus its
#: ``.add()``ed children only; a child placed by ``.anchor()`` does not
#: feed back into it. Same shape of hazard again: every v11 source still
#: executes, and one that anchors a child extending its host's AABB now
#: derives a different thickness, run axis or facing — so it builds in a
#: different PLACE while raising nothing.
#: v13 — an anchored child INHERITS its host's ``facing`` instead of
#: re-deriving it from its placed AABB. Same shape a third time, and the
#: reason to gate it is one level deeper than v12: the child's own solids do
#: not move, so a stored source recompiles to the same body — but everything
#: anchored INSIDE that child now measures from the other face, and a v12
#: source with an anchored container silently builds its contents there.
#: v14 — that inheritance travels down the whole anchored SUBTREE rather than
#: stopping at the anchored node. Same shape a fourth time, and the reason to
#: gate it is that the two consumers of ``facing`` read different objects: a
#: v13 source whose anchored container carries BOTH an ``.anchor()``ed solid
#: and an opening put them on opposite ends of the same element, because the
#: bake reads the container's frame and the opening reads its body's. Measured
#: at 3.7 m on a 6 m leaf, silently. Recompiling such a source now moves the
#: opening.
#: v15 — a ``Space`` joins its storey / site / facility part through
#: ``IfcRelAggregates`` instead of ``IfcRelContainedInSpatialStructure``
#: (#714: the latter's WR31 forbids a spatial element in it). A DIFFERENT
#: shape from v11-v14, and it is worth naming: nothing about the source
#: MOVED, and no body changed — every v14 source recompiles to the same
#: geometry in the same place. What changed is where the space hangs in the
#: spatial tree, which is what a consumer reading the file back walks. A
#: stored v14 source describes a file whose ``IfcSpace`` is one relation away
#: from where this compiler now puts it, and the round-trip manifest is keyed
#: on that tree.
#: v16 — a layered element's BODY reads the ``LayerSet.outward`` authored on
#: the container whose buildup it is, instead of falling through to
#: the scope-centre rule and disagreeing with the wall it is part of. The
#: v11-v14 shape a fifth time and for the same reason: the two consumers of
#: ``facing`` read different objects, so a v15 source with a TOP-LEVEL layered
#: wall carrying both an ``.anchor()``ed solid and an opening was building them
#: at opposite ends of it. v14 closed that for an ANCHORED wall only — nothing
#: places a top-level ``.add()``, so no inherited reference exists to reach one
#: and only the authored claim can. Recompiling such a source now moves the
#: opening (measured: 3.4 m on a 6 m wall). It executes unchanged either way,
#: which is precisely what the gate is for.
#: v17 — EVERY mesh is a displacement carve host, not only a TERRAIN-wrapped
#: one. A stored source with a solid sunk into a gravel bed, a scan mesh or any
#: other un-wrapped mesh executes unchanged and now pits it. The corpus reaches
#: none of this (measured: 20/20 byte-identical, 0 newly-hosting meshes), which
#: is a statement about the corpus, not about a user's model.
#: v18 — ``.no_carve()`` is DELETED; the exemption is ``carve="none"`` on the
#: ``.add()`` / ``.anchor()`` that attached the element. This is the
#: removed-API shape, the one the ladder started with, and it is the case where
#: a missing bump is worst: a stored v17 source calling ``.no_carve()`` does not
#: build the wrong thing, it dies with a bare ``AttributeError`` deep inside the
#: sandboxed exec, with nothing naming the cutover. Every model built from the
#: skills corpus before that calls it — 22 sites across 8 systems — so the
#: reach here is the opposite of v17's: broad among stored user models, and zero
#: in the corpus, which was migrated in the same chapter.
#: v19 — ``Box(rotations=)`` is DELETED; the one rotation path is
#: ``placement=Transform(...)``. Removed-API shape, like v18: a stored v18
#: source passing ``rotations=`` to a Box used to compile (the solid rotated, but
#: every bounds query ignored it) and now stops at construction with a message
#: naming the replacement. Zero uses in the skills corpus; the reach is stored
#: user models.
LITESTEP_META_VERSION = "19"


@dataclass
class LitestepMeta:
    """Extracted round-trip metadata from an IFC file."""
    source: str
    manifest: Dict[str, str]
    version: str
    hash: str
    hash_valid: bool
    # True when this IFC was produced by cold-importing a third-party
    # IFC (no original LITESTEP_META) and then recompiling the
    # reconstructed source. Sticky across edits: when the patch flow
    # recompiles a cold-imported file, the output is marked
    # cold_imported=True too. UI uses this to keep the lossiness
    # banner visible until the file is regenerated from scratch.
    cold_imported: bool = False


def base_version_refusal(
    meta: Optional[LitestepMeta], *, base_name: str = "the base IFC",
) -> Optional[str]:
    """Why this stored base cannot be continued — or ``None`` when it can.

    One rule: **a base may only be continued by the compiler that wrote
    it.** Any ``LITESTEP_META`` version other than
    :data:`LITESTEP_META_VERSION` is refused.

    That is an ALLOWLIST, and the allowlist is the point. This guard used
    to enumerate the known-bad versions, which is a denylist: it passes
    everything it has not been taught about, so a base written by a
    *newer* compiler than this one sailed straight through unguarded —
    exactly the case where the stored source is most likely to mean
    something this compiler does not implement. An allowlist cannot have
    that hole and needs no maintenance when the version moves.

    ``meta is None`` is NOT a stale model. It is a third-party IFC that
    carries no ``LITESTEP_META`` at all — a legitimate cold-import — so it
    returns ``None`` and takes the normal cold path.

    The refusal is loud and actionable on purpose: the user's next move is
    to start a fresh build, and nothing downstream can repair the file.
    """
    if meta is None or meta.version == LITESTEP_META_VERSION:
        return None
    return (
        f"this model was built by a different version of Lite-STEP — start a "
        f"fresh build. {base_name} carries LITESTEP_META version "
        f"{meta.version}; this compiler reads and writes version "
        f"{LITESTEP_META_VERSION}. A stored source is only meaningful to the "
        f"compiler that wrote it: re-executing it here either raises on DSL "
        f"that no longer exists or — worse — still runs and means something "
        f"else, which is wrong geometry in silence."
    )


def _encode_source(source_code: str) -> str:
    """Gzip-compress and base64-encode source code.

    ``mtime=0`` is load-bearing, not tidiness: gzip stamps the current time into
    its header, so without it the SAME source compressed twice produces DIFFERENT
    bytes. That made the embedded blob — and therefore the whole IFC file — differ
    run-to-run on an unchanged model, which defeats any content-hash comparison
    over the output and makes "did this change?" unanswerable by diff.
    """
    compressed = gzip.compress(source_code.encode("utf-8"), mtime=0)
    return base64.b64encode(compressed).decode("ascii")


def _decode_source(encoded: str) -> str:
    """Base64-decode and gzip-decompress source code."""
    compressed = base64.b64decode(encoded)
    return gzip.decompress(compressed).decode("utf-8")


def _hash_source(source_code: str) -> str:
    """SHA-256 hash of source code."""
    return hashlib.sha256(source_code.encode("utf-8")).hexdigest()


def embed_litestep_meta(
    model,
    host,
    source_code: str,
    manifest: Dict[str, str],
) -> None:
    """
    Embed round-trip metadata into an IFC model as a PropertySet on the given
    spatial host, the IfcSite.

    ``LITESTEP_META`` is re-homed to the IfcSite (not the IfcBuilding): the
    IfcSite always exists, whereas a site-only model (terrain, no building
    elements) has no IfcBuilding. ``extract_litestep_meta`` finds the pset by
    NAME globally, so the host object does not affect round-trip recovery.

    Creates LITESTEP_META IfcPropertySet with:
    - SOURCE: gzip+base64 encoded Lite-STEP source
    - MANIFEST: JSON {element_id: ifc_global_id}
    - VERSION: schema version ("1")
    - HASH: SHA-256 of source

    Args:
        model: ifcopenshell model object
        host: spatial-structure entity (the IfcSite) to attach the PropertySet to
        source_code: Raw Lite-STEP Python source string
        manifest: Mapping of element IDs to IFC GlobalIds
    """
    encoded_source = _encode_source(source_code)
    manifest_json = json.dumps(manifest, separators=(",", ":"))
    source_hash = _hash_source(source_code)

    # Create IfcPropertySingleValue entries
    # Use IfcText for SOURCE (can be very long), IfcLabel for shorter values
    props = []

    source_prop = model.create_entity(
        "IfcPropertySingleValue",
        Name="SOURCE",
        NominalValue=model.create_entity("IfcText", wrappedValue=encoded_source),
    )
    props.append(source_prop)

    manifest_prop = model.create_entity(
        "IfcPropertySingleValue",
        Name="MANIFEST",
        NominalValue=model.create_entity("IfcText", wrappedValue=manifest_json),
    )
    props.append(manifest_prop)

    version_prop = model.create_entity(
        "IfcPropertySingleValue",
        Name="VERSION",
        NominalValue=model.create_entity("IfcLabel", wrappedValue=LITESTEP_META_VERSION),
    )
    props.append(version_prop)

    hash_prop = model.create_entity(
        "IfcPropertySingleValue",
        Name="HASH",
        NominalValue=model.create_entity("IfcLabel", wrappedValue=source_hash),
    )
    props.append(hash_prop)

    # Create PropertySet
    pset = model.create_entity(
        "IfcPropertySet",
        GlobalId=ifcopenshell.guid.new(),
        Name=LITESTEP_META_PSET_NAME,
        HasProperties=props,
    )

    # Attach to the spatial host (the IfcSite) via IfcRelDefinesByProperties
    model.create_entity(
        "IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(),
        RelatingPropertyDefinition=pset,
        RelatedObjects=[host],
    )

    logger.info(
        "Embedded LITESTEP_META: source=%d bytes, manifest=%d elements, hash=%s",
        len(source_code), len(manifest), source_hash[:12],
    )


def extract_litestep_meta(ifc_content: str) -> Optional[LitestepMeta]:
    """
    Extract LITESTEP_META from an IFC STEP string.

    Searches for IfcPropertySet named "LITESTEP_META" and extracts
    SOURCE, MANIFEST, VERSION, HASH properties.

    Args:
        ifc_content: IFC STEP file content as string

    Returns:
        LitestepMeta if found, None if no embedded metadata exists
    """
    if not IFC_AVAILABLE:
        # Plan §3.D E.5: user-uploaded IFC — parse failure is user input, not infra.
        logger.warning("ifcopenshell not available")
        return None

    # ifcopenshell requires a file path
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
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # Search by PropertySet Name (FR-2.7: not by GlobalId)
    pset = None
    for ps in model.by_type("IfcPropertySet"):
        if ps.Name == LITESTEP_META_PSET_NAME:
            pset = ps
            break

    if pset is None:
        logger.debug("No LITESTEP_META PropertySet found — cold import path")
        return None

    # Extract properties by name
    prop_values = {}
    for prop in pset.HasProperties:
        if hasattr(prop, "Name") and hasattr(prop, "NominalValue"):
            value = prop.NominalValue
            if value is not None:
                prop_values[prop.Name] = value.wrappedValue if hasattr(value, "wrappedValue") else str(value)

    # Validate required properties
    if "SOURCE" not in prop_values:
        logger.warning("LITESTEP_META found but SOURCE property missing")
        return None

    try:
        source = _decode_source(prop_values["SOURCE"])
    except Exception as e:
        logger.warning("Failed to decode SOURCE: %s", e)
        return None

    try:
        manifest = json.loads(prop_values.get("MANIFEST", "{}"))
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse MANIFEST: %s", e)
        manifest = {}

    version = prop_values.get("VERSION", "1")
    stored_hash = prop_values.get("HASH", "")

    actual_hash = _hash_source(source)
    hash_valid = stored_hash == actual_hash
    if not hash_valid:
        logger.warning("Source hash mismatch: stored=%s actual=%s", stored_hash[:12], actual_hash[:12])

    cold_imported_raw = prop_values.get("COLD_IMPORTED", "")
    cold_imported = str(cold_imported_raw).strip().lower() in {"true", "1", "yes"}

    return LitestepMeta(
        source=source,
        manifest=manifest,
        version=version,
        hash=stored_hash,
        hash_valid=hash_valid,
        cold_imported=cold_imported,
    )


def set_cold_imported_flag(ifc_content: str) -> str:
    """Append a COLD_IMPORTED=true property to the LITESTEP_META PSet
    in ``ifc_content`` and return the rewritten STEP string.

    If no LITESTEP_META PSet is present (the source had embedding
    disabled), returns ``ifc_content`` unchanged. Idempotent: if the
    flag is already present and truthy, returns unchanged.
    """
    if not IFC_AVAILABLE:
        logger.warning("ifcopenshell not available; can't set cold_imported flag")
        return ifc_content

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".ifc")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            f.write(ifc_content)
        try:
            model = ifcopenshell.open(tmp_path)
        except Exception as e:
            logger.warning("Failed to parse IFC for cold_imported tag: %s", e)
            return ifc_content

        pset = None
        for ps in model.by_type("IfcPropertySet"):
            if ps.Name == LITESTEP_META_PSET_NAME:
                pset = ps
                break
        if pset is None:
            return ifc_content  # nothing to mark

        # Idempotency: if COLD_IMPORTED is already truthy, skip rewrite
        for prop in pset.HasProperties:
            if hasattr(prop, "Name") and prop.Name == "COLD_IMPORTED":
                val = prop.NominalValue
                existing = val.wrappedValue if hasattr(val, "wrappedValue") else str(val)
                if str(existing).strip().lower() in {"true", "1", "yes"}:
                    return ifc_content
                # Else: an explicit "false" — overwrite with "true" below
                pset.HasProperties = [p for p in pset.HasProperties if p is not prop]
                break

        prop = model.create_entity(
            "IfcPropertySingleValue",
            Name="COLD_IMPORTED",
            NominalValue=model.create_entity("IfcLabel", wrappedValue="true"),
        )
        pset.HasProperties = list(pset.HasProperties) + [prop]
        return model.to_string()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
