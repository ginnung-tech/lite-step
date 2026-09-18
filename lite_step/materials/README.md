# Adding a material to the registry

The registry lives in [`dk.py`](dk.py) as a dict of `key -> MaterialDef`. This
file is the checklist for adding one.

> **Scope.** This is about extending the *shipped registry* — product data that
> every model can select by key. Defining a one-off material inline in a
> model's `output.py` is a different thing; see "Before you add one at all"
> below, and the agent-facing skill for inline materials.

A material answers **four independent questions**, and the commonest mistake is
answering one of them in another's slot.

| field | question | vocabulary |
| --- | --- | --- |
| `geometry.form` | how is it MODELLED? | `mass · sheet · member · bar · membrane · fill · cavity` |
| `category` | what FAMILY of stuff is it? | `concrete · masonry · timber · metal · insulation · glass · board · membrane · aggregate · ceramic · cavity` |
| `function` | what ROLE does it play in a buildup? | `LoadBearing · Insulation · Finish · Membrane · …` |
| `treatments` | what does ONE OF IT look like? | `mass · section · unit` |

They are genuinely orthogonal. A brick is `mass` / `masonry` / `LoadBearing` as
murværk and `mass` / `masonry` / `Finish` as a veneer — same family, same form,
different role. A clay tile is a `sheet` product whose `section` treatment is a
curved wave.

`category` and `function` land on **different IFC attributes** —
`IfcMaterial.Category` and `IfcMaterialLayer.Category` respectively. Merging
them looks harmless and breaks both consumers at once.

## The five rules

1. **One source per dimension.** A number lives in exactly one place. Stock
   lists, formats and thicknesses go in `geometry`; a `unit` treatment's
   `module_mm` is left `None` so it derives from `cover_width_mm` /
   `format_mm`. Restating a dimension is how the two copies drift.
2. **`category` is a published interface.** It becomes `IfcMaterial.Category`
   and every 2D drawing tool hatches on it. Tools canonicalise a class name
   with `re.sub("[^0-9a-zA-Z]+", "", name)` — which strips separators and does
   **not** lowercase — so the vocabulary is lowercase and separator-free.
   Adding a family means adding it to `MaterialCategory` (dk.py) **and** to
   `CATEGORY_HATCH` in [`../draft/stylesheet.py`](../draft/stylesheet.py); a
   family with no hatch mapping fails its test.
3. **`draft` only when the product differs from its family.** Hatching follows
   `category` automatically. Add `Draft(hatch=…, line_weight_mm=…, cut_rgb=…)`
   only for a product that must read differently from its siblings in section.
4. **Absence of material is still a material.** A ventilation gap is
   `Cavity_Ventilated`, not an authored flag — that keeps a buildup uniform
   (every entry a `Material`) and one-source. `form="cavity"` emits no solid.
5. **Psets are SI and honest.** Density kg/m³, λ W/(m·K), strength N/mm² (MPa),
   capacity J/(kg·K). Floats are legal here — the strict int-mm gate is a
   geometry gate. GWP is A1–A3 per EN 15804+A2; mark generic benchmarks
   `DataSource: "generic"` and replace with product EPDs at detail stage.
   Timber GWP is **fossil-only** — biogenic carbon is reported separately and
   never netted out.

## Minimum viable entry

```python
"Zinc_Titanium_EN988": MaterialDef(
    category="metal",            # family  → IfcMaterial.Category (2D hatch key)
    function="Finish",           # role    → IfcMaterialLayer.Category
    psets={
        "Pset_MaterialCommon": {"MassDensity": 7200},
        "DK_EPD": {"GWP_A1A3_kgCO2e_per_m3": 24000, "DataSource": "generic"},
    },
    render=Render(rgb="#8C8F91", alpha=1.0, finish="metallic"),
    geometry=SheetGeometry(form="sheet", thicknesses_mm=[1],
                           min_bend_radius_mm=2),
)
```

A `member` must **say** what its stock is, including saying it has none:
`stock_profiles_mm=[]` is the explicit made-to-order declaration (ironmongery is
turned or cast to the product, so there is no stock list to check against).
Leaving every stock field `None` stays an error, so a forgotten catalog fails
loudly instead of shipping an unconstrained member.

## Treatments — optional, and additive

`geometry` records product FACTS; `treatments` records what one of the thing
looks like. An entry with no `treatments` behaves exactly as it always has.

```python
treatments={
  "tile": SectionTreatment(kind="section", profile=[(0,0), (300,0), (150,40)]),
}
```

Profile points are int mm and the outline is implicitly closed — do **not**
repeat the first point (a repeated closing point is a zero-length edge, and it
is rejected at construction rather than surviving into a sweep).

Which treatment a model uses is a **skill** decision, never a compiler one. The
compiler emits what it was asked for and never inspects LOD.

## Before you add one at all

`material="SomeUnknownKey"` already works — it emits a plain named
`IfcMaterial` and a compile warning. That is the open-vocabulary escape hatch,
and it is the right move for a genuine one-off.

What it costs you: no Psets, no LCA, no render colour, no dimension validation,
no 2D hatch. Add a registry entry when the material will recur, or when any of
those five matter.

## Verify

```bash
pytest lite_step/tests/test_treatments_and_draft.py lite_step/tests/test_materials.py
```

The family/hatch coverage and canonicalisation checks fail on a malformed
vocabulary before anything reaches a drawing. Then compile any model and check
the emitted attribute directly:

```bash
grep -o "IFCMATERIAL('<YourKey>'[^)]*)" output.ifc
# → IFCMATERIAL('Zinc_Titanium_EN988',$,'metal')
```

A `$` in the third slot means `category` is missing and the material will not
hatch in any drawing.
