"""Drawing wiring, emitted INTO the IFC so 2D works with zero user setup.

The whole point of this module: someone opens our ``output.ifc`` in a BIM
authoring tool, asks for a plan, and the materials are already hatched. No
preferences to set, no paths to point at, no import step.

That is reachable because the drawing configuration is IFC-resident:

* **``BBIM_Documentation`` on ``IfcProject``** — ``StylesheetPath``,
  ``PatternsPath``, ``MarkersPath``, ``SymbolsPath``, ``ShadingStylesPath``.
  Bonsai's ``get_default_drawing_resource_path`` reads this pset FIRST and only
  falls back to add-on preferences when it is absent, and it resolves the value
  through ``resolve_uri()`` — i.e. **relative to the .ifc file**. So a relative
  ``StylesheetPath`` points at the CSS we ship beside the model.

  We deliberately set ONLY ``StylesheetPath``. Leaving ``PatternsPath`` unset
  means the tool keeps using its own pattern library, which it installs itself
  — so our stylesheet's ``url(#concrete)`` resolves without us shipping (or
  licence-entangling ourselves with) anyone's artwork.

* **Drawings as ``IfcAnnotation`` with ``ObjectType="DRAWING"``** — the tool
  enumerates drawings straight out of the file
  (``by_type("IfcAnnotation") if e.ObjectType == "DRAWING"``), so a drawing we
  author appears in its Drawings list on open. Its ``EPset_Drawing`` carries
  ``TargetView``, ``Scale`` and — crucially — ``Metadata``, the opt-in list
  that projects material identity onto SVG classes. Setting it ourselves is
  what makes the stylesheet bind without the user configuring anything.

The assets themselves cannot live inside the IFC (the pset holds paths, not
blobs), so the CSS must ride the download bundle. Degradation is graceful: with
the IFC alone the tool falls back to its own default stylesheet and the drawing
still renders, just unhatched.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Default stylesheet location relative to the IFC. Matches what
#: ``lite_step.draft.write_stylesheet`` writes next to ``output.ifc``.
DEFAULT_STYLESHEET_PATH = "styles/lite-step.css"

#: ``EPset_Drawing.Metadata`` — the element attributes projected onto SVG
#: classes. ``mats.Category`` is the one that makes our family hatching work;
#: ``mats.Name`` gives per-product overrides a selector; ``predefined_type``
#: is free and useful for doors/windows.
DEFAULT_METADATA = "predefined_type,mats.Name,mats.Category"

#: Where generated drawing SVGs land, relative to the IFC. Matches the tool's
#: own ``drawings/`` convention so a drawing we author and one the user adds
#: end up side by side.
DRAWINGS_DIR = "drawings"

#: Where the drawing tool keeps its per-project shading styles, relative to
#: the IFC. The file does NOT need to exist and we ship no copy: when the path
#: is missing, Bonsai copies its own out-of-the-box ``shading_styles.json``
#: there. Naming the path is what triggers that — leaving the property unset is
#: a hard ERROR on activate ("Could not find shading styles path in
#: EPset_Drawing.ShadingStyles"), which is how our first drawings still threw a
#: red box at the user after the view opened correctly.
SHADING_STYLES_PATH = "styles/shading_styles.json"

#: Which of the shipped styles to select. Must be a key that exists in the
#: file above ("Technical", "Shaded", "Blender Default") — a name that is not
#: there degrades to a warning plus no active style, which then fails
#: ``activate_drawing_style``. "Technical" is the flat line-drawing look a plan
#: wants; "Blender Default" is the tool's own default and renders shaded.
DEFAULT_SHADING_STYLE = "Technical"

#: Plan cut height above storey level, in metres. Matches the consumer's own
#: convention (Bonsai's ``generate_drawing_matrix`` uses ``z + 1.6``) rather
#: than inventing our own — a camera that disagrees with the tool's default
#: reads as "our files are odd" the first time someone compares two plans.
PLAN_CUT_HEIGHT_M = 1.6

#: How far BELOW the cut the plan still sees, in metres. Sets the camera's
#: ``clip_end``, so it has to reach the floor slab or the plan loses it.
PLAN_VIEW_BELOW_CUT_M = 2.0

#: Fallback ortho view extents in metres, used when the model's own footprint
#: cannot be measured. Deliberately generous: an oversized ortho camera costs
#: empty paper, an undersized one silently crops the building.
DEFAULT_VIEW_WIDTH_M = 100.0
DEFAULT_VIEW_HEIGHT_M = 100.0


def _pset(model: Any, name: str, props: dict[str, Any]) -> Any:
    """Create an ``IfcPropertySet`` with ``IfcPropertySingleValue`` children.

    Values are all strings/reals here, so the wrapping is direct rather than
    going through the registry cache's polymorphic ``_wrap_pset_value``.
    """
    import ifcopenshell

    entities = []
    for prop_name, value in props.items():
        if value is None:
            continue
        if isinstance(value, bool):
            wrapped = model.create_entity("IfcBoolean", wrappedValue=value)
        elif isinstance(value, float):
            wrapped = model.create_entity("IfcReal", wrappedValue=value)
        elif isinstance(value, int):
            wrapped = model.create_entity("IfcInteger", wrappedValue=value)
        else:
            wrapped = model.create_entity("IfcLabel", wrappedValue=str(value))
        entities.append(model.create_entity(
            "IfcPropertySingleValue", Name=prop_name, NominalValue=wrapped))

    return model.create_entity(
        "IfcPropertySet",
        GlobalId=ifcopenshell.guid.new(),
        Name=name,
        HasProperties=entities,
    )


def attach_documentation_pset(
    model: Any,
    ifc_project: Any,
    *,
    stylesheet_path: str = DEFAULT_STYLESHEET_PATH,
) -> Optional[Any]:
    """Point the file's drawings at our generated stylesheet.

    Only ``StylesheetPath`` is set — see the module docstring for why the
    other four resource paths are deliberately left to the tool's own
    defaults.
    """
    import ifcopenshell

    pset = _pset(model, "BBIM_Documentation", {"StylesheetPath": stylesheet_path})
    model.create_entity(
        "IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(),
        RelatedObjects=[ifc_project],
        RelatingPropertyDefinition=pset,
    )
    return pset


def _sanitise_filename(name: str) -> str:
    """Match the tool's own filename rule so the SVG lands where it looks."""
    return "".join(c for c in name if c.isalnum() or c in "._- ")


def _camera_representation(model: Any, body_context: Any, *,
                           width_m: float, height_m: float, depth_m: float) -> Any:
    """The drawing's Body representation — its view VOLUME, not a picture.

    A consumer builds the camera from this box, reading its bounding extents
    directly (``get_x``/``get_y`` → ortho width/height, ``get_z`` →
    ``clip_end``). It must live in the **Model/Body/MODEL_VIEW** context: the
    lookup is ``get_representation(drawing, "Model", "Body", "MODEL_VIEW")``
    and it is a hard ``assert``, not a graceful miss.

    The box hangs DOWNWARD from the placement, because the placement sits at
    the cut plane and a plan looks down.
    """
    # IfcBlock, not an extrusion. The consumer only reads the shape's bounding
    # extents, so any solid works — but every solid we emit for real geometry is
    # an IfcExtrudedAreaSolid, and a camera box among them breaks any test that
    # counts extrusions. Three test files trip on exactly that without this
    # switch. A CSG primitive keeps the view volume out of the geometry census,
    # and it is the family the consumer already expects here (it sniffs for
    # IfcRectangularPyramid to detect a perspective camera).
    #
    # The block's corner sits at (-w/2, -h/2, -depth) so the volume is centred
    # on the placement in plan and hangs BELOW it — the placement is the cut
    # plane, and a plan looks down.
    block = model.create_entity(
        "IfcBlock",
        Position=model.create_entity(
            "IfcAxis2Placement3D",
            Location=model.create_entity(
                "IfcCartesianPoint",
                Coordinates=(-float(width_m) / 2.0, -float(height_m) / 2.0,
                             -float(depth_m)),
            ),
        ),
        XLength=float(width_m),
        YLength=float(height_m),
        ZLength=float(depth_m),
    )
    shape = model.create_entity(
        "IfcShapeRepresentation",
        ContextOfItems=body_context,
        RepresentationIdentifier="Body",
        RepresentationType="CSG",
        Items=[block],
    )
    return model.create_entity("IfcProductDefinitionShape", Representations=[shape])


def create_plan_drawing(
    model: Any,
    ifc_storey: Any,
    *,
    name: str,
    body_context: Any = None,
    elevation_m: float = 0.0,
    width_m: float = DEFAULT_VIEW_WIDTH_M,
    height_m: float = DEFAULT_VIEW_HEIGHT_M,
    scale: str = "1/100",
    human_scale: str = "1:100",
    stylesheet_path: str = DEFAULT_STYLESHEET_PATH,
    metadata: str = DEFAULT_METADATA,
) -> Any:
    """Author one PLAN_VIEW drawing for a storey.

    The drawing is an ``IfcAnnotation(ObjectType="DRAWING")`` plus the
    ``IfcGroup(ObjectType="DRAWING")`` that holds its element set — the
    two-entity shape the tool expects, joined by ``IfcRelAssignsToGroup``.
    An empty group means "everything visible at this cut", which is the right
    default for a generated plan.

    **A drawing is a CAMERA, and a camera needs geometry.** Enumeration only
    needs the annotation and its pset — the drawing shows up in the tool's
    list with nothing else. Activating it is what needs a placement and a Body
    representation, and the failure is an unguarded ``assert representation``
    deep in the importer, i.e. a traceback rather than a message. Measured in
    Bonsai 0.8.5: our first drawings listed fine and then blew up
    on activate. Listing is not evidence that a drawing works.

    Placement convention is the consumer's own: identity axes (the default
    camera looks down ``-Z``) at the storey elevation plus
    :data:`PLAN_CUT_HEIGHT_M`.

    ``Scale`` is a string fraction (``"1/100"``), not a number: that is how
    ``EPset_Drawing`` stores it and how the reader parses it (``Fraction``).
    """
    import ifcopenshell

    placement = model.create_entity(
        "IfcLocalPlacement",
        RelativePlacement=model.create_entity(
            "IfcAxis2Placement3D",
            Location=model.create_entity(
                "IfcCartesianPoint",
                Coordinates=(0.0, 0.0, float(elevation_m) + PLAN_CUT_HEIGHT_M),
            ),
        ),
    )
    representation = None
    if body_context is not None:
        representation = _camera_representation(
            model, body_context,
            width_m=width_m, height_m=height_m,
            depth_m=PLAN_CUT_HEIGHT_M + PLAN_VIEW_BELOW_CUT_M,
        )

    drawing = model.create_entity(
        "IfcAnnotation",
        GlobalId=ifcopenshell.guid.new(),
        Name=name,
        ObjectType="DRAWING",
        ObjectPlacement=placement,
        Representation=representation,
    )
    group = model.create_entity(
        "IfcGroup",
        GlobalId=ifcopenshell.guid.new(),
        Name=name,
        ObjectType="DRAWING",
    )
    model.create_entity(
        "IfcRelAssignsToGroup",
        GlobalId=ifcopenshell.guid.new(),
        RelatedObjects=[drawing],
        RelatingGroup=group,
    )

    # Mirror the property set the tool writes for its OWN drawings. Anything
    # it expects and does not find is an error dialog at activate time, not a
    # graceful default — see SHADING_STYLES_PATH. Markers/Symbols/Patterns are
    # deliberately LEFT OUT so the tool keeps using its own installed library;
    # naming a path we do not ship would break the pattern references our
    # stylesheet depends on.
    pset = _pset(model, "EPset_Drawing", {
        "TargetView": "PLAN_VIEW",
        "Scale": scale,
        "HumanScale": human_scale,
        "HasUnderlay": False,
        "HasLinework": True,
        "HasAnnotation": True,
        "GlobalReferencing": False,
        "Stylesheet": stylesheet_path,
        "ShadingStyles": SHADING_STYLES_PATH,
        "CurrentShadingStyle": DEFAULT_SHADING_STYLE,
        "Metadata": metadata,
    })
    model.create_entity(
        "IfcRelDefinesByProperties",
        GlobalId=ifcopenshell.guid.new(),
        RelatedObjects=[drawing],
        RelatingPropertyDefinition=pset,
    )

    # The drawing's OUTPUT DOCUMENT — where its SVG is written.
    #
    # Not optional and not cosmetic: `create_drawing` resolves the output path
    # through `get_document_uri(camera_document)` with no None check, so a
    # drawing with no associated document dies with
    # ``AttributeError: 'NoneType' object has no attribute 'is_a'``. Measured
    # against Bonsai 0.8.5 — the third unguarded dereference in
    # this activation path, each one only reachable once the previous is fixed.
    #
    # Shape mirrors the tool's own: Information (the logical document) <-
    # Reference (the file) <- RelAssociatesDocument (the drawing uses it).
    information = model.create_entity(
        "IfcDocumentInformation",
        Identification="X",
        Name=name,
        Scope="DRAWING",
    )
    reference = model.create_entity(
        "IfcDocumentReference",
        Location=f"{DRAWINGS_DIR}/{_sanitise_filename(name)}.svg",
        ReferencedDocument=information,
    )
    model.create_entity(
        "IfcRelAssociatesDocument",
        GlobalId=ifcopenshell.guid.new(),
        RelatedObjects=[drawing],
        RelatingDocument=reference,
    )
    return drawing


def project_view_extents(proj: Any) -> tuple[float, float]:
    """Ortho view width/height in metres covering the model, with a margin.

    Falls back to the generous default rather than failing: a drawing with a
    slightly wrong paper size is still a usable drawing, and this must never be
    able to break a geometry compile.
    """
    try:
        from lite_step.compiler.frames import element_aabb, _union_aabb

        boxes = []
        for storey in getattr(proj, "storeys", None) or []:
            for elem in storey.elements:
                box = element_aabb(elem)
                if box is not None:
                    boxes.append(box)
        union = _union_aabb(boxes)
        if union is None:
            return (DEFAULT_VIEW_WIDTH_M, DEFAULT_VIEW_HEIGHT_M)
        (lo, hi) = union
        margin = 2.0
        return (max(abs(hi[0] - lo[0]) + margin, 1.0),
                max(abs(hi[1] - lo[1]) + margin, 1.0))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("drawings: view extents not measured (%s) — using %sx%s m",
                       exc, DEFAULT_VIEW_WIDTH_M, DEFAULT_VIEW_HEIGHT_M)
        return (DEFAULT_VIEW_WIDTH_M, DEFAULT_VIEW_HEIGHT_M)


def emit_drawings(model: Any, ifc_project: Any, storeys: list[tuple[Any, str]],
                  *, stylesheet_path: str = DEFAULT_STYLESHEET_PATH,
                  body_context: Any = None, proj: Any = None) -> int:
    """Emit the documentation pset + one plan drawing per storey.

    ``storeys`` is ``[(ifc_storey_entity, label)]``. Returns the number of
    drawings created. Every failure here is best-effort and LOUD: drawing
    wiring is a convenience layer, and it must never be able to fail a
    geometry compile.
    """
    try:
        attach_documentation_pset(model, ifc_project, stylesheet_path=stylesheet_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("drawings: BBIM_Documentation pset not attached (%s) — "
                       "2D falls back to the tool's default stylesheet", exc)

    width_m, height_m = (project_view_extents(proj) if proj is not None
                         else (DEFAULT_VIEW_WIDTH_M, DEFAULT_VIEW_HEIGHT_M))

    count = 0
    for ifc_storey, label in storeys:
        try:
            elevation = 0.0
            try:
                elevation = float(ifc_storey.Elevation or 0.0)
            except Exception:
                pass
            create_plan_drawing(model, ifc_storey, name=f"PLAN_{label}",
                                body_context=body_context, elevation_m=elevation,
                                width_m=width_m, height_m=height_m,
                                stylesheet_path=stylesheet_path)
            count += 1
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("drawings: plan for storey %r not created (%s)", label, exc)
    logger.info("drawings: %d plan drawing(s) + documentation pset emitted", count)
    return count
