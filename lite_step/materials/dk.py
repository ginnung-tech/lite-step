"""Lite-STEP material registry — typical modern Danish single-family house.

Ships as `lite_step.materials.dk`. Strict Pydantic throughout: the registry
validates at import, the `Material` selection validates at construction.

Resolution of `material=` on any geometry primitive
(`Optional[str | Material]`, default `None`; a bare string coerces to
`Material(key=...)`):

  * None        -> sketch stage: no material committed; renders in SKETCH_VIEW
                   with the palette color (color= / container auto-color,
                   including color="glass" for ideation transparency).
  * key found   -> material Psets attached to the IfcMaterial, `render` emitted
                   as IfcSurfaceStyle, `geometry` consulted for defaults and
                   validation. Element-level Psets (ExposureClass, U-values,
                   FireRating) stay on the element via its own props=.
  * key unknown -> plain named IfcMaterial, stderr warning (open vocabulary).
                   Unknown key + dimension fields = construction error (nothing
                   to validate against).

Dimension-carrying Material — the material IS the product spec:

    Sweep(material=Material(key="Timber_C24", profile_mm=(45, 195)), path=[...])
    Extrude(contour=[...], material=Material(key="Chipboard_P6", thickness_mm=22))

One-source rule: a dimension lives in exactly one place. `profile_mm` on the
Material replaces `profile=` on the Sweep; `thickness_mm` replaces
`thickness=` on the Solid. Providing both fails at construction.

`geometry.form` (discriminated union) — the modeling form; the primitive follows:

  mass      poured/laid volume            -> Solid (box or contour)
  sheet     flat stock                    -> Solid (contour; thickness from stock)
  member    linear stock                  -> Sweep (section from stock)
  bar       reinforcement                 -> Bar
  membrane  roll goods wrapping surfaces  -> Solid (thin sheet; overlap_mm applies)
  fill      loose material by layer       -> Solid (mass; layer_thickness_min_mm)

Conventions:
  * Pset values: SI, floats legal (non-geometric). Density kg/m3, lambda W/(m.K),
    strength N/mm2 (MPa), heat capacity J/(kg.K).
  * geometry values: int mm unless the key says otherwise; angles centidegrees.
  * GWP: A1-A3, EN 15804+A2, generic benchmarks for early-stage BR18 LCA.
    Replace with product EPDs at detail stage. Timber GWP is fossil-only;
    biogenic carbon is reported separately, never netted out here.
"""
from __future__ import annotations

import warnings
from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lite_step.materials.treatments import (  # noqa: F401  (re-exported)
    MassTreatment, SectionTreatment, Treatment, UnitTreatment, resolve_module_mm,
)

PsetValue = Union[bool, int, float, str]


class _Strict(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


# ---------------------------------------------------------------- appearance

class Render(_Strict):
    rgb: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    alpha: float = Field(ge=0.0, le=1.0)
    finish: Literal["matt", "satin", "gloss", "metallic"]


#: Material FAMILY — the coarse bucket, emitted as ``IfcMaterial.Category``.
#:
#: This is the key 2D drawing tools hatch on. Bonsai derives the SVG class
#: ``layer-material-category-<Category>`` straight from this attribute, so the
#: vocabulary is a published interface: lowercase and alphanumeric, because
#: Bonsai's ``canonicalise_class_name`` is ``re.sub("[^0-9a-zA-Z]+", "", name)``
#: — it strips separators and does NOT lowercase, so ``Concrete`` and
#: ``concrete`` are different selectors.
#:
#: Distinct from ``MaterialDef.function``, which is the layer ROLE
#: (LoadBearing / Insulation / Finish) and rides ``IfcMaterialLayer.Category``.
#: Two different IFC attributes, two different vocabularies — do not merge them:
#: a brick is family ``masonry`` whether its role is LoadBearing or Finish.
MaterialCategory = Literal[
    "concrete", "masonry", "timber", "metal", "insulation",
    "glass", "board", "membrane", "aggregate", "ceramic", "cavity",
]


class Draft(_Strict):
    """2D presentation — sibling of ``render``, because how a material reads in
    a plan or section is part of its IDENTITY, not of any one element's
    geometry.

    Emitted to a generated stylesheet, NOT to IFC entities:
    ``IfcFillAreaStyleHatching`` / ``IfcFillAreaStyleTiles`` are marked "no
    support planned" upstream in IfcOpenShell and every drawing tool
    we target styles via CSS keyed on material identity instead. See
    ``lite_step/draft/stylesheet.py``.

    All three fields are optional; ``hatch`` defaults from ``category``.
    """
    #: SVG ``<pattern>`` id. Resolves against the drawing tool's own pattern
    #: library — we reference ids, we do not ship patterns.
    hatch: Optional[str] = None
    #: Cut-line weight in mm (CSS ``stroke-width``).
    line_weight_mm: Optional[float] = Field(default=None, gt=0.0)
    #: Fill of the CUT face where it differs from the projected surface colour.
    cut_rgb: Optional[str] = Field(default=None, pattern=r"^#[0-9A-Fa-f]{6}$")


# ------------------------------------------------- geometry (one model per form)

class MasonryModule(_Strict):
    """Exact integer ratio for masonry coordination (murmaal) — e.g. 3 courses
    = 200 mm; the single-course height (66.7) is never stored as a float.
    Snap vertical masonry dimensions to multiples of module_height_mm."""
    courses_per_module: int
    module_height_mm: int


class MassGeometry(_Strict):
    form: Literal["mass"]
    cast_in_place: bool = False
    default_thickness_mm: dict[str, int] = Field(default_factory=dict)
    joint_thickness_mm: Optional[int] = None            # mortar
    format_mm: Optional[tuple[int, int, int]] = None    # brick l x w x h
    masonry_module: Optional[MasonryModule] = None
    module_length_mm: Optional[int] = None
    leaf_thickness_mm: Optional[int] = None


class SheetGeometry(_Strict):
    form: Literal["sheet"]
    format_mm: Optional[tuple[int, int]] = None
    thicknesses_mm: Optional[list[int]] = None
    joist_spacing_max_mm: Optional[int] = None
    min_bend_radius_mm: Optional[int] = None
    min_bend_radius_factor_t: Optional[float] = None
    # roofing-layer sheets (tiles)
    batten_spacing_mm: Optional[tuple[int, int]] = None
    cover_width_mm: Optional[int] = None
    tiles_per_m2: Optional[int] = None
    min_roof_pitch_centideg: Optional[int] = None


class MemberGeometry(_Strict):
    form: Literal["member"]
    stock_profiles_mm: Optional[list[tuple[int, int]]] = None
    stock_lengths_mm: Optional[list[int]] = None
    stock_widths_mm: Optional[list[int]] = None          # glulam: w x d matrix
    stock_depths_mm: Optional[list[int]] = None
    max_length_mm: Optional[int] = None

    @model_validator(mode="after")
    def _has_catalog(self) -> "MemberGeometry":
        """A member must SAY what its stock is — including saying it has none.

        ``stock_profiles_mm=[]`` is the explicit made-to-order declaration
        (ironmongery: turned, cast or formed to the product, so there is no
        stock list to check against). Leaving every field ``None`` stays an
        error, so a registry author who simply forgot the catalog still fails
        loudly rather than shipping an unconstrained member.
        """
        if self.stock_profiles_mm is None and not (self.stock_widths_mm and self.stock_depths_mm):
            raise ValueError("member needs stock_profiles_mm (use [] for "
                             "made-to-order) or stock_widths_mm+stock_depths_mm")
        return self

    def has_profile(self, w: int, h: int) -> bool:
        """Catalog membership, orientation-agnostic. An explicitly EMPTY
        ``stock_profiles_mm`` is made-to-order — any profile is in stock."""
        if self.stock_profiles_mm is not None:
            if not self.stock_profiles_mm:
                return True
            return (w, h) in self.stock_profiles_mm or (h, w) in self.stock_profiles_mm
        ws, ds = self.stock_widths_mm or [], self.stock_depths_mm or []
        return (w in ws and h in ds) or (h in ws and w in ds)


class BarGeometry(_Strict):
    form: Literal["bar"]
    stock_diameters_mm: list[int]
    stock_length_mm: int
    bend_rule: Literal["EC2_8.1N"]      # applied by Bar(bend_radius=None)


class MembraneGeometry(_Strict):
    form: Literal["membrane"]
    roll_width_mm: int
    overlap_mm: int
    thicknesses_mm: Optional[list[float]] = None   # sub-mm products
    min_bend_radius_mm: int = 0


class FillGeometry(_Strict):
    form: Literal["fill"]
    layer_thickness_min_mm: int


class CavityGeometry(_Strict):
    """An air/cavity layer in a planar buildup — absence of material, not a
    product. Occupies its thickness in the stack but the compiler emits NO
    solid slice; ``ventilated`` drives ``IfcMaterialLayer.IsVentilated``. The
    home for what would otherwise be an authored ``ventilated=`` flag: it is a
    registry FACT (Cavity_Ventilated vs Cavity_Unventilated), so a buildup
    stays uniform (every entry a ``Material``) and one-source."""
    form: Literal["cavity"]
    ventilated: bool


Geometry = Annotated[
    Union[MassGeometry, SheetGeometry, MemberGeometry,
          BarGeometry, MembraneGeometry, FillGeometry, CavityGeometry],
    Field(discriminator="form"),
]


# ------------------------------------------------------------- registry entry

class MaterialDef(_Strict):
    psets: dict[str, dict[str, PsetValue]]
    render: Render
    geometry: Geometry
    #: Functional role in a planar buildup → ``IfcMaterialLayer.Category``
    #: (buildingSMART values: LoadBearing / Insulation / Finish / Membrane /
    #: …). SINGLE SOURCE — the compiler derives Category from this, never from
    #: an authored per-layer string (form is GEOMETRIC and can't give it: a
    #: `mass` brick is Finish-as-veneer or LoadBearing-as-murværk). ``None``
    #: → Category omitted (e.g. a cavity, where IsVentilated carries meaning).
    function: Optional[str] = None
    #: Material FAMILY → ``IfcMaterial.Category``. The 2D hatch key. See
    #: :data:`MaterialCategory`. NOT the same axis as ``function``.
    category: Optional[MaterialCategory] = None
    #: 2D presentation. ``None`` → derived entirely from ``category``.
    draft: Optional[Draft] = None
    #: Geometric treatments, keyed by role (e.g. ``"tile"``, ``"moulding"``).
    #: Empty = implicitly ``mass``, i.e. exactly today's behaviour. Selection
    #: between them lives in the skill tree, never in the compiler.
    treatments: dict[str, Treatment] = Field(default_factory=dict)
    texture: Optional[str] = None


# ------------------------------------------------------------------ selection
# `Material` (the selection: key + optional product dimensions) ships from
# lite_step.models — it must see all registered registries, not just this
# module's. Re-imported here so registry users can grab both from one place.

from lite_step.models.material import Material  # noqa: E402


# ------------------------------------------------------------------- registry

MATERIALS: dict[str, MaterialDef] = {

    # Concrete — DS/EN 206 + DK NA
    # Element-level, not here: Pset_ConcreteElementGeneral.ExposureClass
    "Concrete_C30-37": MaterialDef(
        category="concrete",
        function="LoadBearing",
        psets={
            "Pset_MaterialCommon":   {"MassDensity": 2400},
            "Pset_MaterialConcrete": {"CompressiveStrength": "C30/37", "MaxAggregateSize": 32},
            "Pset_MaterialThermal":  {"ThermalConductivity": 2.0, "SpecificHeatCapacity": 880},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_m3": 150,
                       "Note": "low-carbon mix (FUTURECEM-class)", "DataSource": "generic"},
        },
        render=Render(rgb="#B4B4B0", alpha=1.0, finish="matt"),
        geometry=MassGeometry(form="mass", cast_in_place=True,
                              default_thickness_mm={"foundation_slab": 300,
                                                    "strip_footing": 400, "wall": 150}),
        texture="gravel_concrete_diff_1k.jpg",
    ),
    "Concrete_C25-30": MaterialDef(
        category="concrete",
        function="LoadBearing",
        psets={
            "Pset_MaterialCommon":   {"MassDensity": 2400},
            "Pset_MaterialConcrete": {"CompressiveStrength": "C25/30", "MaxAggregateSize": 32},
            "Pset_MaterialThermal":  {"ThermalConductivity": 2.0, "SpecificHeatCapacity": 880},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_m3": 140, "DataSource": "generic"},
        },
        render=Render(rgb="#BBBBB6", alpha=1.0, finish="matt"),
        geometry=MassGeometry(form="mass", cast_in_place=True,
                              default_thickness_mm={"inner_wall": 150, "deck": 180}),
        texture="gravel_concrete_diff_1k.jpg",
    ),

    # Rebar — DS/EN 10080 B500B. Bar(grade="B500B") resolves here.
    "Steel_B500B": MaterialDef(
        category="metal",
        psets={
            "Pset_MaterialCommon":     {"MassDensity": 7850},
            "Pset_MaterialSteel":      {"YieldStress": 500, "UltimateStress": 540},
            "Pset_MaterialMechanical": {"YoungModulus": 200000, "PoissonRatio": 0.3},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 0.55,
                       "Note": "scrap-based EAF", "DataSource": "generic"},
        },
        render=Render(rgb="#55585E", alpha=1.0, finish="matt"),
        geometry=BarGeometry(form="bar", stock_diameters_mm=[6, 8, 10, 12, 16, 20, 25, 32],
                             stock_length_mm=12000, bend_rule="EC2_8.1N"),
        texture="rusty_metal_04_diff_1k.jpg",
    ),

    # Red clay brick — DS/EN 771-1 / EC6, bloedstroegen
    # Danish module: 228x108x54 + 12 mm joint; 3 courses = 200 mm
    "Brick_Red_DK": MaterialDef(
        category="masonry",
        function="LoadBearing",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 1800},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.62, "SpecificHeatCapacity": 840},
            "DK_Masonry": {"NormalizedCompressiveStrength_fb": 20},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 0.22, "DataSource": "generic"},
        },
        render=Render(rgb="#9E3B2B", alpha=1.0, finish="matt"),
        geometry=MassGeometry(form="mass", format_mm=(228, 108, 54),
                              masonry_module=MasonryModule(courses_per_module=3,
                                                           module_height_mm=200),
                              module_length_mm=240, leaf_thickness_mm=108),
        texture="red_brick_diff_1k.jpg",
    ),

    # Mortar — KC 50/50/700, receptmoertel M5
    "Mortar_KC50-50-700": MaterialDef(
        category="masonry",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 1900},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.80, "SpecificHeatCapacity": 900},
            "DK_Masonry": {"MortarClass": "M5", "Recipe": "KC 50/50/700"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 0.15, "DataSource": "generic"},
        },
        render=Render(rgb="#C8C4B7", alpha=1.0, finish="matt"),
        geometry=MassGeometry(form="mass", joint_thickness_mm=12),
    ),

    # Structural softwood — DS/EN 338 C24, gran/fyr
    "Timber_C24": MaterialDef(
        category="timber",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 420},   # rho_mean; rho_k=350 for statik
            "Pset_MaterialWood":    {"Species": "Spruce/Pine", "StrengthGrade": "C24",
                                     "MoistureContent": 12},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.12, "SpecificHeatCapacity": 1600},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_m3": 60,
                       "Note": "fossil only, excl. biogenic", "DataSource": "generic"},
        },
        render=Render(rgb="#C79A5B", alpha=1.0, finish="matt"),
        geometry=MemberGeometry(form="member",
                                stock_profiles_mm=[(45, 45), (45, 70), (45, 95), (45, 120),
                                                   (45, 145), (45, 170), (45, 195), (45, 220)],
                                stock_lengths_mm=list(range(2400, 5401, 300))),
        texture="wooden_floor_02_diff_1k.jpg",
    ),

    # Glulam — DS/EN 14080 GL24h (long spans, ridge beams)
    "Glulam_GL24h": MaterialDef(
        category="timber",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 420},
            "Pset_MaterialWood":    {"Species": "Spruce", "StrengthGrade": "GL24h",
                                     "MoistureContent": 12},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.13, "SpecificHeatCapacity": 1600},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_m3": 100,
                       "Note": "fossil only, excl. biogenic", "DataSource": "generic"},
        },
        render=Render(rgb="#D2A868", alpha=1.0, finish="satin"),
        geometry=MemberGeometry(form="member",
                                stock_widths_mm=[90, 115, 140, 160],
                                stock_depths_mm=list(range(180, 631, 45)),   # 45 mm lamella
                                max_length_mm=12000),
        texture="coated_pine_02_diff_1k.jpg",
    ),

    # Glulam — DS/EN 14080 GL28h. The HEAVY long-span grade: homogeneous
    # lay-up, fm,k = 28 MPa (vs 24), E0,mean = 12 600 MPa, rho_k = 425.
    # Use when a single beam carries a roof over a clear span with NO
    # intermediate support (simply supported, bearing at both ends only) —
    # ridge/purlin beams, carport and hall spans, open-plan roof beams.
    # GL24h stays the default for ordinary rafters and short spans.
    #
    # Sizing a simply-supported roof beam: bending capacity grows with the
    # SQUARE of depth and deflection falls with its CUBE, while width scales
    # both only linearly — so buy depth first. Rule of thumb d ~ span/17 to
    # span/20, with width >= d/6 for lateral stability:
    #     6 m clear span  -> ~ 140 x 360
    #     8 m clear span  -> ~ 165 x 450   (140 x 495 equally valid)
    #    10 m clear span  -> ~ 190 x 585
    # Deflection (not strength) usually governs a roof beam this long —
    # verify to EC5 / DK NA for the actual snow + permanent load; these
    # sections are the starting point, not a substitute for the check.
    "Glulam_GL28h": MaterialDef(
        category="timber",
        function="LoadBearing",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 460},   # rho_mean; rho_k=425
            "Pset_MaterialWood":    {"Species": "Spruce", "StrengthGrade": "GL28h",
                                     "MoistureContent": 12},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.13, "SpecificHeatCapacity": 1600},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_m3": 100,
                       "Note": "fossil only, excl. biogenic", "DataSource": "generic"},
        },
        render=Render(rgb="#C08E4E", alpha=1.0, finish="satin"),
        geometry=MemberGeometry(form="member",
                                stock_widths_mm=[115, 140, 165, 190, 215],
                                stock_depths_mm=list(range(360, 1081, 45)),  # 45 mm lamella
                                max_length_mm=18000),
        texture="coated_pine_02_diff_1k.jpg",
    ),

    # Floor panels — DS/EN 312 P6 gulvspaanplade, tongue & groove
    "Chipboard_P6": MaterialDef(
        category="timber",
        function="Substrate",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 680},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.13, "SpecificHeatCapacity": 1700},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_m3": 230, "DataSource": "generic"},
        },
        render=Render(rgb="#C9A876", alpha=1.0, finish="matt"),
        geometry=SheetGeometry(form="sheet", format_mm=(2400, 600),
                               thicknesses_mm=[22], joist_spacing_max_mm=600),
    ),

    # Monolithic float glass — DS/EN 572, single panes: partitions, sketch->real
    # first step. Multi-cavity IGUs are assemblies -> element-level
    # (Pset_WindowCommon.ThermalTransmittance on the Window), not materials.
    "Glass_Monolithic": MaterialDef(
        category="glass",
        function="Finish",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 2500},
            "Pset_MaterialThermal": {"ThermalConductivity": 1.0, "SpecificHeatCapacity": 840},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 1.4, "DataSource": "generic"},
        },
        render=Render(rgb="#B8D4DA", alpha=0.30, finish="gloss"),
        geometry=SheetGeometry(form="sheet", thicknesses_mm=[4, 6]),
    ),

    # Vacuum insulated glazing (VIG) — two panes fused around an evacuated
    # ~0.2 mm gap with micro-pillars (FINEO/LandVac class). The one glazing
    # UNIT modeled as a material: it is a monolithic rigid sheet product.
    # No ThermalConductivity — vacuum has no lambda; the honest number is the
    # unit U-value in DK_Product.
    "Glass_VIG": MaterialDef(
        category="glass",
        function="Finish",
        psets={
            "Pset_MaterialCommon": {"MassDensity": 2400},   # effective, incl. gap
            "Pset_MaterialThermal": {"SpecificHeatCapacity": 840},
            "DK_Acoustic": {"Rw_dB": 36},
            "DK_Product": {"Ug_W_m2K": 0.7, "Structure": "4-0.2V-4, micro-pillar",
                           "LightTransmittance": 0.80,
                           "Note": "retrofit-capable in existing frames"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 1.8, "DataSource": "generic"},
        },
        render=Render(rgb="#B0CFD8", alpha=0.30, finish="gloss"),
        geometry=SheetGeometry(form="sheet", format_mm=(1500, 2500),   # max fabrication
                               thicknesses_mm=[8, 10]),
    ),

    # Acoustic mineral wool — inner walls/ceilings
    "MineralWool_Acoustic37": MaterialDef(
        category="insulation",
        function="Insulation",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 28},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.037, "SpecificHeatCapacity": 1030},
            "DK_Fire": {"ReactionToFire": "A1"},
            "DK_Acoustic": {"AirflowResistivity_kPa_s_m2": 5},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 1.2, "DataSource": "generic"},
        },
        render=Render(rgb="#C9B36B", alpha=1.0, finish="matt"),
        geometry=SheetGeometry(form="sheet", format_mm=(960, 560),
                               thicknesses_mm=[45, 70, 95, 120]),
    ),

    # Steel brackets — CE-marked connectors (vinkelbeslag), S250GD+Z275
    "Steel_S250GD_Z275": MaterialDef(
        category="metal",
        psets={
            "Pset_MaterialCommon":     {"MassDensity": 7850},
            "Pset_MaterialSteel":      {"YieldStress": 250, "UltimateStress": 330},
            "Pset_MaterialMechanical": {"YoungModulus": 210000, "PoissonRatio": 0.3},
            "DK_Product": {"CorrosionProtection": "Z275", "ServiceClass": "1-2"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 2.5, "DataSource": "generic"},
        },
        render=Render(rgb="#A8ADB5", alpha=1.0, finish="metallic"),
        geometry=SheetGeometry(form="sheet", thicknesses_mm=[2, 3],
                               min_bend_radius_factor_t=1.0),
        texture="blue_metal_plate_diff_1k.jpg",
    ),

    # Sålbænk / inddækning — titanium zinc to EN 988, the classic DK flashing
    # metal. Pre-weathered blue-grey. Cold-forming below +7 C is prohibited
    # (the metal cracks), hence the generous bend factor.
    #
    # thicknesses_mm=[1] is DELIBERATE, not an oversight. Real zinc gauges are
    # 0.65 / 0.7 / 0.8 mm, but `SheetGeometry.thicknesses_mm` is `list[int]`
    # and going sub-mm would ripple through every stock check in the registry
    # for the sake of one product. 1 mm is the modelled gauge.
    "Zinc_Titanium_EN988": MaterialDef(
        category="metal",
        function="Finish",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 7200},
            "Pset_MaterialThermal": {"ThermalConductivity": 110, "SpecificHeatCapacity": 385},
            "DK_Product": {"Designation": "EN 988 titanium zinc",
                           "SurfaceFinish": "pre-weathered blue-grey",
                           "MinFormingTemperature_C": 7,
                           "ServiceLife_years": 80},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 3.9, "DataSource": "generic"},
        },
        render=Render(rgb="#7D8A90", alpha=1.0, finish="satin"),
        geometry=SheetGeometry(form="sheet", thicknesses_mm=[1],
                               min_bend_radius_factor_t=1.75),
        texture="blue_metal_plate_diff_1k.jpg",
    ),

    # Ironmongery — stainless steel EN 1.4301 / AISI 304. Door and window
    # furniture. NOT a structural grade: it is here so a handle stops
    # borrowing the bracket-steel key.
    "Steel_Stainless_304": MaterialDef(
        category="metal",
        function="Finish",
        psets={
            "Pset_MaterialCommon":     {"MassDensity": 7900},
            "Pset_MaterialSteel":      {"YieldStress": 210, "UltimateStress": 520},
            "Pset_MaterialMechanical": {"YoungModulus": 200000, "PoissonRatio": 0.3},
            "DK_Product": {"Designation": "EN 1.4301 (AISI 304)",
                           "SurfaceFinish": "brushed, grit 240",
                           "CorrosionClass": "C3 inland, C4 coastal"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 4.5, "DataSource": "generic"},
        },
        render=Render(rgb="#C7CCD1", alpha=1.0, finish="satin"),
        # Ironmongery is not stock-sectioned — turned, cast or formed to the
        # product. The empty stock list is the made-to-order declaration, so
        # `Material(profile_mm=)` is unconstrained here.
        geometry=MemberGeometry(form="member", stock_profiles_mm=[]),
        texture="blue_metal_plate_diff_1k.jpg",
    ),

    # Gypsum boards — loft/sloped-wall cladding
    "Gypsum_Standard": MaterialDef(
        category="board",
        function="Finish",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 720},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.25, "SpecificHeatCapacity": 1000},
            "DK_Fire": {"ReactionToFire": "A2-s1,d0", "SurfaceClass": "K1 10 / B-s1,d0"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 0.22, "DataSource": "generic"},
        },
        render=Render(rgb="#E8E4DC", alpha=1.0, finish="matt"),
        geometry=SheetGeometry(form="sheet", format_mm=(900, 2400),
                               thicknesses_mm=[13], min_bend_radius_mm=2750),
    ),
    "Gypsum_Fiber": MaterialDef(
        category="board",
        function="Finish",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 1150},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.32, "SpecificHeatCapacity": 1100},
            "DK_Fire": {"ReactionToFire": "A2-s1,d0", "SurfaceClass": "K1 10 / B-s1,d0"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 0.30, "DataSource": "generic"},
        },
        render=Render(rgb="#D9D2C4", alpha=1.0, finish="matt"),
        geometry=SheetGeometry(form="sheet", format_mm=(1200, 2400),
                               thicknesses_mm=[13, 15]),
    ),

    # Acoustic ceiling panels — Troldtekt-class wood wool
    "WoodWool_Acoustic": MaterialDef(
        category="insulation",
        function="Insulation",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 450},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.08, "SpecificHeatCapacity": 1500},
            "DK_Fire": {"ReactionToFire": "B-s1,d0"},
            "DK_Acoustic": {"SoundAbsorption_alpha_w": 0.85, "AbsorptionClass": "A"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 0.50, "DataSource": "generic"},
        },
        render=Render(rgb="#B7AE92", alpha=1.0, finish="matt"),
        geometry=SheetGeometry(form="sheet", format_mm=(600, 1200),
                               thicknesses_mm=[25, 35]),
    ),

    # Roof insulation — high-density stone wool boards (tagbatts)
    "MineralWool_Roof38": MaterialDef(
        category="insulation",
        function="Insulation",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 160},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.038, "SpecificHeatCapacity": 1030},
            "DK_Fire": {"ReactionToFire": "A1"},
            "DK_Product": {"CompressiveStress_CS10_kPa": 40},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 1.3, "DataSource": "generic"},
        },
        render=Render(rgb="#8A7B55", alpha=1.0, finish="matt"),
        geometry=SheetGeometry(form="sheet", format_mm=(2000, 1200),
                               thicknesses_mm=[80, 100, 120, 150, 180, 220]),
    ),

    # Facade insulation — rigid mineral wool (facadebatts)
    "MineralWool_Facade34": MaterialDef(
        category="insulation",
        function="Insulation",
        psets={
            "Pset_MaterialCommon":      {"MassDensity": 90},
            "Pset_MaterialThermal":     {"ThermalConductivity": 0.034, "SpecificHeatCapacity": 1030},
            "Pset_MaterialHygroscopic": {"UpperVaporResistanceFactor": 1,
                                         "LowerVaporResistanceFactor": 1},
            "DK_Fire": {"ReactionToFire": "A1"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 1.2, "DataSource": "generic"},
        },
        render=Render(rgb="#97865E", alpha=1.0, finish="matt"),
        geometry=SheetGeometry(form="sheet", format_mm=(1000, 600),
                               thicknesses_mm=[125, 150, 190, 220, 250, 300]),
    ),

    # Vapour barrier — reinforced PE foil (dampspaerre), SBi 224
    "PE_VapourBarrier": MaterialDef(
        category="membrane",
        function="Membrane",
        psets={
            "Pset_MaterialCommon": {"MassDensity": 920},
            "DK_Product": {"ZValue_GPa_s_m2_per_kg": 400,
                           "Note": "warm-side rule: Z_warm >= 10 x Z_cold"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 2.0, "DataSource": "generic"},
        },
        render=Render(rgb="#7FA6B8", alpha=0.45, finish="gloss"),
        geometry=MembraneGeometry(form="membrane", roll_width_mm=4000,
                                  thicknesses_mm=[0.2], overlap_mm=150),
    ),

    # Drainage gravel — noeddesten 16/32, washed
    "Gravel_16-32": MaterialDef(
        category="aggregate",
        psets={
            "Pset_MaterialCommon": {"MassDensity": 1700},
            "DK_Product": {"GrainSize_mm": "16-32", "Fines": "washed, zero fines"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 0.005, "DataSource": "generic"},
        },
        render=Render(rgb="#9B9284", alpha=1.0, finish="matt"),
        geometry=FillGeometry(form="fill", layer_thickness_min_mm=150),
        texture="clean_pebbles_diff_1k.jpg",
    ),

    # Roof tiles — engobed black clay (tegltag), DS/EN 1304
    "ClayTile_Black_Engobed": MaterialDef(
        category="ceramic",
        function="Finish",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 2100},
            "Pset_MaterialThermal": {"ThermalConductivity": 1.0, "SpecificHeatCapacity": 800},
            "DK_Product": {"DeadLoad_kg_per_m2": 43,
                           "FrostResistance": "EN 1304 tested, Nordic",
                           "SurfaceFinish": "Engoberet, black satin"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 0.45, "DataSource": "generic"},
        },
        render=Render(rgb="#1F1F23", alpha=1.0, finish="gloss"),
        geometry=SheetGeometry(form="sheet",
                               batten_spacing_mm=(320, 345), cover_width_mm=300,
                               tiles_per_m2=10, min_roof_pitch_centideg=2500),
        texture="worn_tile_floor_diff_1k.jpg",
    ),

    # --- Gaps vs. source list — needed for a complete typical DK house ---

    # Radon barrier — BR18 §332 mandatory; distinct from dampspaerre
    "RadonBarrier": MaterialDef(
        category="membrane",
        function="Membrane",
        psets={
            "Pset_MaterialCommon": {"MassDensity": 950},
            "DK_Product": {"RadonTight": True, "Standard": "SBi 233"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 2.2, "DataSource": "generic"},
        },
        render=Render(rgb="#3E5C6E", alpha=0.6, finish="gloss"),
        geometry=MembraneGeometry(form="membrane", roll_width_mm=4000,
                                  thicknesses_mm=[0.4], overlap_mm=300),
    ),

    # Terraendaek insulation — EPS under ground slab
    "EPS_S250": MaterialDef(
        category="insulation",
        function="Insulation",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 25},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.038, "SpecificHeatCapacity": 1450},
            "DK_Product": {"CompressiveStress_CS10_kPa": 250},
            "DK_Fire": {"ReactionToFire": "F"},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 3.5, "DataSource": "generic"},
        },
        render=Render(rgb="#F2F2F2", alpha=1.0, finish="matt"),
        geometry=SheetGeometry(form="sheet", format_mm=(1200, 600),
                               thicknesses_mm=[100, 150, 200, 300, 400]),
    ),

    # Roofing underlay (undertag) — required beneath tile roofs
    "Underlay_Membrane": MaterialDef(
        category="membrane",
        function="Membrane",
        psets={
            "Pset_MaterialCommon": {"MassDensity": 400},
            "DK_Product": {"Type": "diffusion-open banevare", "Sd_m": 0.05},
            "DK_EPD": {"GWP_A1A3_kgCO2e_per_kg": 2.5, "DataSource": "generic"},
        },
        render=Render(rgb="#4E5A52", alpha=1.0, finish="matt"),
        geometry=MembraneGeometry(form="membrane", roll_width_mm=1500, overlap_mm=150),
    ),

    # Cavities — the air/ventilation gap in a planar buildup. A cavity is the
    # ABSENCE of material: it holds its thickness in the stack (spacing the
    # neighbouring leaves) but the compiler emits NO solid slice; IsVentilated
    # rides the form. Two products so ventilation is a registry FACT, not an
    # authored flag (a ventilated brick-veneer cavity vs a sealed cavity use
    # the same air but different IsVentilated). Thermal Psets are the still/
    # ventilated air-gap resistances used in BR18 U-value calc (DS 418).
    "Cavity_Ventilated": MaterialDef(
        category="cavity",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 0},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.0, "Description": "ventilated air gap — R by convention (DS 418)"},
        },
        render=Render(rgb="#B8C4CC", alpha=0.0, finish="matt"),
        geometry=CavityGeometry(form="cavity", ventilated=True),
    ),
    "Cavity_Unventilated": MaterialDef(
        category="cavity",
        psets={
            "Pset_MaterialCommon":  {"MassDensity": 0},
            "Pset_MaterialThermal": {"ThermalConductivity": 0.025, "Description": "still air gap (DS 418)"},
        },
        render=Render(rgb="#B8C4CC", alpha=0.0, finish="matt"),
        geometry=CavityGeometry(form="cavity", ventilated=False),
    ),
}


# Element-level assembly values — attach to the Wall element, not to a material:
#   Wall(name="wall_south", props={
#       "Pset_WallCommon": {"ThermalTransmittance": 0.15},   # BR18 target U-value
#       "DK_Masonry": {"CharacteristicStrength_fk": 5.5,     # EC6 system strength
#                      "BlendedDensity_kg_m3": 1820,
#                      "BlendedLambda_W_mK": 0.68},          # per DS 418
#   })


if __name__ == "__main__":
    # Registry validated by construction above; exercise the selection class.
    Material(key="Timber_C24", profile_mm=(45, 195))       # ok
    Material(key="Timber_C24", profile_mm=(195, 45))       # ok — orientation-agnostic
    Material(key="Glulam_GL24h", profile_mm=(115, 405))    # ok — w x d matrix
    Material(key="Glass_VIG", thickness_mm=8)             # ok — real glazing, stock hit

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        Material(key="Chipboard_P6", thickness_mm=18)      # off-stock -> warns
        assert len(w) == 1 and "off-stock" in str(w[0].message)

    for bad in (dict(key="Timber_C24", profile_mm=(50, 200)),    # off-catalog
                dict(key="Chipboard_P6", profile_mm=(45, 195)),  # wrong form
                dict(key="NoSuchMaterial", thickness_mm=22)):    # dims on unknown key
        try:
            Material(**bad)
        except Exception:
            pass
        else:
            raise AssertionError(f"should have failed: {bad}")

    forms = sorted({m.geometry.form for m in MATERIALS.values()})
    print(f"OK — {len(MATERIALS)} materials validated at import; forms: {forms}")
