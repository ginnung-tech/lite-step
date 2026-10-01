# Lite-STEP — Agent Context & Rules

## 1. One tree, four verbs

Lite-STEP is semantic building geometry: you declare ELEMENTS and how they relate.
The model is one containment tree — grown with `.add()` and `.anchor()`, cut with `.void()` and `.opening()`.
The compiler derives the rest: joinery, holes, rooms, schedules, identity.

```text
Project
├─ .add(Site "site")                                          Spatial
│    └─ .add(Mesh "terrain")                                  Geometry
└─ .add_storey(Storey "ground")                               Spatial
     └─ .add(Wall "south")                                    Physical
          ├─ .add(Box "body")                                 Geometry, in WORLD coords
          ├─ .add(Wall "cladding")
          ├─ .add(Element "studs")                            ifc_class="IfcMember"
          │    └─ .add(Element "stud_00" …)
          ├─ .anchor(Box "ledge", along=2000, up=1200)        Placed in the HOST frame
          └─ .opening(Window "kitchen_window", along=1500, up=900)   Hole + the unit filling it
```

**Identity flows down the tree.**
`name=` is the lowercase leaf (`[a-z0-9_]+`, unique per scope); the canonical path is derived from containment, leaf-first: `window:kitchen_window:wall:south:storey:ground`.
To point AT an element — `Anchor(host=…)`, `bounds=[…]`, `members=[…]`, `assert_carved(by=…)`, `--q` — use that string.
Fragments match whole `type:leaf` pairs FROM THE TAIL. On a NAMED storey `wall:south` never matches; write `wall:south:storey:ground`. The single ANONYMOUS storey emits no storey segment at all, so there `wall:south` is the whole path.
Identity IS the canonical name: it is the IFC `Name`, the manifest key and the patch-mode differ key. Moving an element to another container is a NEW identity.
A `Product` is the one OBJECT handle, and it has no path.

**Any element or Point is callable → a derived copy.**
`s2 = s1(start=s1.start(y=4200))`. Same container → a new `name` is required.
`Product` is the exception: `.occurrence(name=)` is its only copy verb.

**The CSG budget is two numbers, printed on every compile.**
`csg depth: max=N (budget 10)  width: max=M (budget 99)`.

- **Depth** breaks HARD and INVISIBLY: past ~13 nested `IfcBooleanResult` levels the web viewer stops evaluating and renders the UN-subtracted base solid. Valid IFC, no error, wrong picture. The compiler warns above 10 and never fails the build.
- **Width** is the boolean COUNT on one representation. It breaks SOFT — the geometry stays correct and evaluation time grows about `N^1.33`, with no cliff anywhere. 99 is a reporting threshold, not a limit.
- Operands of one class are emitted as a BALANCED tree (`((A−c1)−c2)−c3 ≡ A−(c1∪c2∪c3)`), so depth is `ceil(log2(V))+1`, not `V`. The class order cuts → adds → intersects is preserved; `.clip()` half-spaces stay a chain.
- Deep sculpting belongs in `Mesh`.

## 2. Hard rules (violations raise)

1. **Keyword-only** — every constructor: `Point(x=0, y=0, z=0)`. One exception: `Product(source, name=)` takes its source positionally.
   A positional call raises `TypeError: BaseModel.__init__() takes 1 positional argument but N were given`. It names no parameter, so read it as "spell the keyword" and nothing else. The single-argument constructors are where this bites: `Element(ifc_class="IfcStair")`, `Material(key="Timber_C24")`, `Space(name="kitchen")`.
2. **Int millimetres** — every mm dimension; floats are rejected at construction, always, with no switch to relax it. Wrap division in `I()` (`I = lambda x: int(round(x))`, part of the skeleton). Direction tuples (`outward=`, `normal=`, `edge=`) take floats; otherwise floats live only in `props={}` and in `round=False` query returns.
3. **Centidegrees** — `90° = 9000`, full turn `36000`. Rotation axes are lowercase `"x"` / `"y"` / `"z"`.
4. **World frame** — Z-up, right-handed, building centred at the origin. North = Y+, South = Y−, East = X+, West = X−. "South wall" = the wall at negative y.
5. **The local frame is seen from −Y.** Every element is authored as if it were the south wall:
   - origin = its OUTER-face lower-left corner,
   - `along` (+x) = right as seen from outside,
   - `inset` (+y) = INTO the element from its OUTER face,
   - `up` (+z) = up.
6. **One placement per element** — `.anchor()` twice, `.add()` plus `.anchor()`, or `placement=` on an anchored child: each raises. Derive a copy per placement (`win(name="w1")`).
7. **`placement=` is consumed on TOP-LEVEL storey elements only.** On a nested element (container child, opening child), on a boolean operand, or on a `Site` container it is a loud compile error, never a silent skip. Place the container and author its children in container-local coordinates.
8. **A `Window`/`Door` can never be `.add()`ed** — not to a container, not to a `Storey`, not to a `Project`. An opening carries a SIZE and no position of its own, and `.add()` claims the child's coordinates are already world coordinates, so there is nothing for it to keep. The HOST places it: `wall.anchor(win, along=…, up=…)` or `body.opening(win, along=…, up=…)`.
9. **Meshes that are building elements must be WATERTIGHT solids with thickness.** A custom roof, slab, plate or wall authored as `Mesh` must be a closed volume that holds water. Never author a zero-thickness open sheet: an open mesh cannot receive boolean cuts, culls from underneath in viewers, and has zero BIM volume. Triangle faces are wound counter-clockwise seen from OUTSIDE, so normals point out (positive signed volume).
10. **Fixed skeleton** — this header, this tail, never edit below it:

    ```python
    import math
    import numpy as np
    from lite_step import compile_main
    from lite_step.models import (Project, Storey, Point, Point2D,
        Wall, Window, Door, Slab, Roof, Column, Beam, Space, Site, Element,
        Box, Extrude, Sweep, Sheet, Pipe, Revolve, Mesh, Bar, Material, LayerSet,
        ReferencePoint, GuideLine, Transform, Anchor)
    from lite_step.helpers import arc_path, stirrup_path
    from lite_step.materials.dk import MATERIALS   # stock introspection

    I = lambda x: int(round(x))

    def generate_project():
        proj = Project(name="New Project")
        # … your model …
        return proj

    result = generate_project()

    if __name__ == "__main__":
        compile_main()
    ```

    `Product`, `SpatialElement`, `miter` and `HalfSpace` are not in the seed import block — add them when you use them. Unused names in that block are deliberate; never prune them.
11. **Patch mode** — `base_ifc = "./input.ifc"` at module scope; `generate_project()` still declares the FULL end state.
    Continuity is carried ONLY by the canonical name — a rename or a move reads downstream as delete + create.
    First build: no `base_ifc` line.

## 3. API quick reference

If a constructor is not here, it does not exist.

```python
# ── API QUICK REFERENCE ─────────────────────────────────────────────
# Every constructor is keyword-only. Every geometry value is int mm.
# Every element also accepts: name=, material=, placement=, props={}.
# layers= additionally on Wall / Slab / Roof / planar Element.

# --- primitives ---
Point(x=0, y=0, z=0)                          Point2D(x=0, y=0)
Box(start: Point, end: Point, material=, color=, type=)   # always axis-aligned; rotate via placement=Transform(...). Box(rotations=) is a hard error
Extrude(contour: list[Point], thickness: int, material=, color=, type=)
# Sweep/Revolve profiles are LOCAL cross-sections about the path — never world coords:
Sweep(profile: list[Point2D], path: list[Point], fillet_radius=0, profile_rotation=0, material=, color=)
Revolve(profile: list[Point2D], path: list[Point], angle=36000, material=, color=)   # path = the 2-point AXIS
Pipe(path: list[Point], radius: int, fillet_radius=0, material=, color=)
Bar(path: list[Point], diameter: int, bend_radius=None, grade='B500B',
    bar_type='main'|'shear'|'stirrup'|'edge', mark=None, material=, color=)
Sheet(profile: list[Point2D], thickness: int, path: list[Point])   # a FUNCTION returning a Sweep

# Mesh takes ONE of two spellings, never both:
Mesh(vertices: list[Point], faces: list[(i, j, k)], color=, source_type=, is_watertight=False)
Mesh(heightmap: list[list[int]], corner_min: Point2D, corner_max: Point2D, depth=|bottom_z=, color=)
# heightmap mode names its base plane with exactly ONE of depth= / bottom_z=:
# passing both raises, passing neither raises.
# Mesh operations (immediate and chainable):
mesh.subdivide(*, height_map_only=False) -> mesh              # 1 -> 4 triangle midpoint subdivision
mesh.smooth(strength=50, *, height_map_only=False) -> mesh    # Laplacian, strength 1..100
mesh.add_random(seed=42, strength=500, walk=0, *, height_map_only=False) -> mesh
mesh.simplify(strength=50, *, height_map_only=False) -> mesh  # decimation, strength 1..100

# --- materials ---
Material(key: str, thickness_mm: int = None, profile_mm: (w, h) = None)
LayerSet(layers: list[Material], outward: (dx, dy, dz), name=None)

# --- elements ---
Wall()  Slab()  Roof()  Column()  Beam()  Site()
Project(name='Project', site_longitude=, site_latitude=, site_elevation=, site_true_north=)
Storey(elevation=0)
Space(space_type=None, bounds: list[str] = None)
Element(ifc_class: str, predefined_type=None)              # PHYSICAL product — see the whitelist below
SpatialElement(ifc_class: str, usage=None, predefined_type=None, long_name=None)
Window(width=None, height=None, style='fixed', panes=1)
Door(width=None, height=None, swing='left', style='hinged')
ReferencePoint(location: Point, point_type='survey', marker_color=, accuracy_mm=, survey_method=, crs=)
GuideLine(points: list[Point], line_type='boundary', label=, buffer_width=, source=, constraint=)

# --- placement (as placement=, TOP-LEVEL storey elements only) ---
Transform(origin: Point, rotations: list[(axis, centideg)] = [])   # rotates about the LOCAL ORIGIN
Anchor(host: str, attach_to='start'|'end'|'center'|'face', offset_along=0,
       offset_inset=0, offset_up=0, rotations=[])

# --- assembly ---
parent.add(*children, carve=None)        -> parent   # WORLD coords
parent.anchor(child, *, along=0|along_center, inset=0, up=0, rotations=None, carve=None) -> parent  # HOST frame
e.opening(child, *, along=0|along_center, inset=0, up=0, rotations=None)  -> e
e.void(*operands, name=None)             -> e
proj.add(*e)   proj.add_storey(s)   storey.add(*e, carve=None)
proj.assert_carved(target, *, by)
proj.aggregate(name, *, members=[...])   proj.zone(name, *, members=[...])

# --- booleans (any solid receiver; operands are CONSUMED) ---
e.difference(*operands)   e.union(*operands)   e.intersection(*operands)
e.clip(*, origin: Point, normal: (dx, dy, dz), overrun=0)  # removes the half-space the normal points at
miter(a, b, *, at: Point, edge: (dx, dy, dz), overrun=0) -> (a, b)

# --- products ---
P = Product(source, *, name)                      # promote a finished UNPLACED element
P.occurrence(*, name)                    -> e     # identity only; geometric kwargs are an error

# --- measuring: THREE FRAMES, and §7 is the rule. Never mix them ---
e.height_at(*, x, y, round=False)                        -> float   # SELF
e.raycast(*, origin: Point, direction: (dx,dy,dz), round=True) -> Point | None  # SELF
e.heightmap_at(*, corner_min, corner_max, rows, cols, round=False) -> list[list]  # SELF
e.authored_aabb()                                        -> Bounds  # AUTHORED
e.world_aabb()  e.obb()                                  -> Bounds  # WORLD
# Bounds: .min .max .center .size .point_at(i, j, k) .is_axis_aligned .rule .exact

# --- path helpers (from lite_step.helpers) ---
arc_path(center: Point, radius: int, start_angle: int, end_angle: int, segments=16, plane='xy'|'xz'|'yz')
stirrup_path(width: int, height: int, origin: Point, hook_angle=13500, hook_length=None, plane='xz')

# --- optimization (from lite_step.optimization) ---
@jit_compile                                  # JIT numerical / raycasting loops (fastmath=True, nogil=True)
solve_sequence(n_items, transition_costs, base_costs=None)  # CP-SAT permutation solver
solve_milp(c, integrality=, bounds=, constraints=)          # HiGHS mixed-integer linear program
solve_path(cost_matrix, start_costs=None)     # vectorized shortest Hamiltonian path / layer matching

# ── CLOSED VOCABULARIES ─────────────────────────────────────────────
# Anything outside these is refused. Full definitions live in files you
# can read:
#   materials   -> lite_step/materials/dk.py     (the MATERIALS dict)
#   ifc classes -> lite_step/models/elements.py  (ELEMENT_IFC_CLASS_WHITELIST)
#   colours     -> lite_step/ifc/colors.py       (ELEMENT_TYPE_COLORS)

# material keys (28) — for Material(key=…) and every LayerSet layer:
#   Brick_Red_DK, Cavity_Unventilated, Cavity_Ventilated, Chipboard_P6,
#   ClayTile_Black_Engobed, Concrete_C25-30, Concrete_C30-37, EPS_S250,
#   Glass_Monolithic, Glass_VIG, Glulam_GL24h, Glulam_GL28h,
#   Gravel_16-32, Gypsum_Fiber, Gypsum_Standard, MineralWool_Acoustic37,
#   MineralWool_Facade34, MineralWool_Roof38, Mortar_KC50-50-700,
#   PE_VapourBarrier, RadonBarrier, Steel_B500B, Steel_S250GD_Z275,
#   Steel_Stainless_304, Timber_C24, Underlay_Membrane,
#   WoodWool_Acoustic, Zinc_Titanium_EN988

# palette colour names (13) — anywhere color= is accepted:
#   beam, ceiling, column, floor, floor_panels, foundation, glass, roof,
#   site, slab, wall, wall-opening, wood_panels

# Element(ifc_class=…) (40) — PHYSICAL products: IFC4 + IFC4.3
# groundworks / landscape / marine + rail. The output schema is IFC4X3,
# so every one emits NATIVELY. predefined_type refines the class; a value
# outside that class's own enum emits as USERDEFINED with the truth in
# ObjectType.
#   IfcBeam, IfcBearing, IfcBorehole, IfcBuildingElementProxy,
#   IfcCaissonFoundation, IfcChimney, IfcColumn, IfcCourse, IfcCovering,
#   IfcCurtainWall, IfcDoor, IfcEarthworksCut, IfcEarthworksFill,
#   IfcFooting, IfcGeographicElement, IfcGeomodel, IfcGeoslice,
#   IfcGeotechnicalStratum, IfcKerb, IfcMember, IfcMooringDevice,
#   IfcNavigationElement, IfcPavement, IfcPile, IfcPlate, IfcRail,
#   IfcRailing, IfcRamp, IfcRampFlight, IfcReinforcedSoil, IfcRoof,
#   IfcShadingDevice, IfcSign, IfcSignal, IfcSlab, IfcStair,
#   IfcStairFlight, IfcTrackElement, IfcWall, IfcWindow
#
# NOT on that list. Each is a construction error, so reach for the
# remedy instead:
#   IfcElementAssembly   -> Element(ifc_class="IfcBuildingElementProxy")
#                           is the wrapper for a multi-part detail cluster
#   IfcWallStandardCase  -> IfcWall
#   IfcBridge / IfcRoad / IfcRailway and their *Part
#                        -> SpatialElement(ifc_class=…). They are SPATIAL,
#                           joined by aggregation, not containment
#   IfcAlignment and its segment / placement family
#                        -> there is no stationing; author road and rail
#                           geometry as ordinary 3D solids
```

## 4. Placement & frames

**`.add(child)` — the child is already in world coordinates.**
It is not moved. It helps DEFINE its container's frame: run axis, thickness, facing.

**`.anchor(child, along=, inset=, up=)` — the child is placed in the host's frame.**
It never defines that frame. It re-derives its own run and across axes from its own body.
It INHERITS the host's facing, down its whole subtree — container and body measure `along` from the same end.
An authored `LayerSet.outward` beats every derivation.
The ONLY difference between the two verbs is the coordinate claim — both make the child a part.

**The two anchor spellings measure differently — never transplant numbers between them.**
`parent.anchor(child, along=…)` zeros at the host's outer-face lower-left corner.
`placement=Anchor(host="…", attach_to=, offset_*=)` zeros at its `attach_to` point and walks the host's raw axis.
`attach_to` is `"center"` by default (the host's AABB centre); `"face"` is the outer side on a `Wall` and the top-face centre on a `Beam`/`Slab`; `"start"` and `"end"` are the two ends of the anchor line.
`inset` / `offset_inset` are positive-INWARD from the outer face.

**`along=` places the child's local origin; `along_center=` centres its host-axis footprint.**
Unrotated, `along=` is the child's near edge. Rotation swings the geometry around the local origin — position rotated children with `along_center=`.
For a `Window`/`Door` the footprint is the hole's width.

**The south-wall frame on every other host is ROTATED about Z — never mirrored.**
Both in-plane axes flip together: a detail authored once mounts on every facade, with no per-facade sign.
`along=0` is the corner on your LEFT facing the outer face — name ends `_left`/`_right`, never `_south`.
Anchor whole assemblies: a sub-assembly authored around its own origin resolves recursively in the host's frame, outside the carve pass.

**One rotation path: `placement=Transform(origin=, rotations=)`.** It rotates about the element's LOCAL ORIGIN `(0, 0, 0)`, so to spin a box about its own centre author it centred on the origin and put the centre in `origin=`. `Box(rotations=)` no longer exists (a hard error pointing here): it was accepted but `world_aabb()` ignored it.
A `.difference()` operand or a `.clip()` plane on a rotated `Box` is read in the box's UNROTATED frame: author the cutter where the box was drawn, not where it ends up.
The list composes left-to-right as an INTRINSIC sequence: each rotation is about the already-rotated axes.

**Default massing — a CONVENTION, not a rule: nothing raises if you turn the building.**
Run the long axis along X, face the entrance facade south (`−Y`), and the gable ends land at `±X`; `examples/08_murermestervilla.py` is authored this way. It is worth keeping because a brief's words are read against it: "the south facade", "the gable window", "the east elevation" each resolve to one wall only while the massing sits where the vocabulary expects. Turn it and nothing fails — the model just stops answering the brief that describes it.

`Site` and `SpatialElement` have no `.anchor()` at all — they are places, not frames; `.add()` is the verb.

**A container may emit no solid of its own.**
A `Wall` aggregating `.add()`ed leaves derives its frame from them — thickness = their sum — and hosts `.anchor()` / `.opening()` against it.

## 5. Geometry & profiles

**Extents vs paths.**
`Box(start=, end=)` names two opposite corners; they auto-normalise, so any opposite pair is accepted.
`Sweep(path=)` is the line the profile's `(0, 0)` travels: a `Material(profile_mm=)` section is CENTRED on it — the solid reaches half a section past the path — while an explicit `profile=` sits wherever it was drawn.
Convert via a real face: `girder.obb().point_at(0, 0, 100)` is the top face centre.
Never guess an extrusion or section DIRECTION — ask `lite_step.compiler.extent.extrude_offset(elem)`.

**An `Extrude`'s winding decides the normal ONLY on a VERTICAL contour.**
The compiler canonicalises the surface normal Newell's method computes:

- **Sloped or horizontal contours** (`|nz| > 0.001`): the normal is forced to point UP. Reversing the contour changes nothing; author layered sloped buildups by offsetting each layer's contour DOWN.
- **Vertical contours** (`|nz| <= 0.001`): the compiler does NOT canonicalise; it trusts the author's winding. Newell's normal follows the right-hand rule — looking from outside, counter-clockwise points outward, clockwise points away. `Extrude` sweeps `thickness` along that normal, so a reversed winding sweeps the solid a full thickness the OTHER way. It compiles, validates and renders; it is simply somewhere else. The compiler warns only where a solid touches nothing as authored AND would touch something reversed, which is exactly that mistake and nothing else.
- **Thickness is strictly a positive magnitude.** `thickness < 0` is an invalid measure, not a backwards sweep. To reverse the extrusion direction on a vertical face, reverse the contour vertex sequence.

**A `profile=` is a LOCAL cross-section, and `(0, 0)` is on the path.**
Nothing validates it, because every offset is a legal one: a world coordinate in a profile reads as a metres-scale offset FROM the path, so the solid compiles clean and stands somewhere the model does not. Author every profile around `(0, 0)` and let `path=` carry the position — which puts the path up the MIDDLE of what it sweeps, never along an edge.
That is also what makes the across-axis SIGN stop mattering: the axis is `unit(Z × d)`, so it REVERSES with the path. Two roof slopes run from opposite eaves to one ridge take the same profile and throw it opposite ways — centred, both land on the building; off-centre, one lands on the building and the other beside it.
`Sweep`: profile X is horizontal ACROSS the path, Y is up — except on a near-vertical segment (`|d.z| > 0.99`), where X turns to world +X so a frame's `profile_mm=(95, 45)` reads "95 through the wall" on stiles and rails alike. A ridge is `path=[…ridge line…]` plus `profile=[Point2D(x=-half_w, y=-drop), Point2D(x=0, y=0), Point2D(x=half_w, y=-drop)]`.
`Revolve`: `path=` is the 2-point AXIS, not a travel line. Profile X is the RADIAL distance from it and must be `>= 0` — crossing the axis raises; mirror the profile instead. Profile Y is distance along the axis from `path[0]`.

**`Sheet` closes an open profile into a section and returns a `Sweep`.**
The point order decides which side gets the metal — material lies 90° LEFT of travel, and this one has no vertical exception.
Mirroring a profile per facade therefore moves the metal to the far side. The host frame already rotates: author the section ONCE.

**`Mesh` takes EITHER `vertices` + `faces` (0-based) OR the heightmap fields.**
Heightmap values are literal top heights; `[0][0]` sits at `corner_min` (SW), rows run +y, cols run +x. `corner_min` / `corner_max` are `Point2D`.
The bottom of the prism is `depth=` OR `bottom_z=`, exactly one — passing both raises, and so does passing neither. `depth=` is RELATIVE: that many mm below the lowest height, so the skirt follows the terrain. `bottom_z=` is an ABSOLUTE level plane and must lie strictly BELOW the lowest height — the prism needs positive thickness everywhere, so it cannot trim a mesh from underneath.

## 6. Overlaps, carving & holes

### Tree-based auto-carving

A solid carves its overlap out of solids EARLIER in the containment tree, walked depth-first — a container's whole subtree precedes its next sibling. WHERE you add decides order, not when. Never author these carves; tree order is the joint. Flush contact does not carve.

**`.add()` is the carver.** Two solids that overlap inside an `.add()` tree are resolved by tree order — one loses material.
**`placement=`, `.anchor()` and `carve="none"` opt out, subtree and all.** Solids that meet under those keep overlapping: an anchored child's position is a RELATIONSHIP to its host, so abutting or entering that host is what anchoring MEANS.
**A `Mesh` never carves — it is only ever carved**, by every overlapping un-exempt solid, regardless of tree order. Exemption is a mesh's only defence.
Boolean operands and mitred pairs are out too: already resolved.

Each verb answers three questions, and they are independent:

| | **tree** — does it bind? | **carve** | **placement** — whose frame? |
| --- | --- | --- | --- |
| `.add(x)` | x becomes a child | in the overlap pass; `carve=` names the direction. A hole in the host cuts x | world — nothing moves x |
| `.add(x, carve="none")` | x becomes a child | out of the overlap pass **and** out of the hole: "this matter is meant to be there" | world |
| `.anchor(x)` | x becomes a child | out of the overlap pass already — its position is a relationship, not an accident. A hole still cuts x; `carve="none"` opts out of that too | the host's frame places x |
| `.opening(win)` | win becomes a child | MAKES the hole. The fill is never carved and never carves — zero booleans | the host's frame; win's own children stand in opening-local coords |
| `.void(tool)` | tool is CONSUMED — never a child, no product | the tool cuts the host and the host's details; it is not itself carved | world — a `.void()` tool is absolute geometry and may overshoot |
| `placement=Transform` | binds NOTHING — a field, and **top-level only** | out of the overlap pass, whole subtree. No host, so no hole reaches it | its own frame |

Read the carve column down and it is one rule three times: `placement=`, `.anchor()` and `carve="none"` each say "I placed this", and `placement=` is that statement without binding to a container at all. The one thing only `carve="none"` reaches is the hole — a void removes matter however the matter got there.

**Who loses material is `carve=` on the `.add()`.**
`"other"`: the element already there yields. `"self"`: the element being added yields instead — for something that POKES OUT of what it is trimmed against; on an element wholly inside its incumbent it yields everything, which compiles and warns. `"none"`: neither, overlap preserved.
On `.anchor()`, `"other"` and `"self"` are accepted and INERT — an anchored child is out of the pairing by construction, so only `"none"` does anything there, and what it does is opt out of the HOLE.
The direction covers the whole SUBTREE and is read INSIDE-OUT: a nested `carve=` outranks the one enclosing it, and a part added later is inside the scope too.
Omitting `carve=` behaves as `"other"` but is not the same statement: an undeclared pair is decided by tree order and flips if two `.add()` lines are swapped, so the compile's `carves: N pairs` line counts those separately and the number can be driven to zero.

**A mesh never carves another element — it can only be carved.**
EVERY mesh is carved by the solids sunk into it: terrain, a fill bed, a bare context mesh alike. Tree order does not apply here; everything overlapping carves it unless exempted.
The carve is EXACT, so clearance is enough — a deck spanning a valley takes nothing out of the ground beneath it.
A solid that swallows the whole mesh carves it to nothing, and says so at compile, naming the volume lost.

### Explicit booleans

**Booleans consume their operands.** `.difference()` / `.union()` / `.intersection()` / `.clip()`: the receiver keeps its identity; the sculpt is invisible to schedules and quantities.
An operand's own CUTS are refused at compile — never nest differences to fake `intersection`; use `.intersection()`. An operand's own `.union()` IS honoured: an arched doorway is `wall.difference(door.union(arch))`.

**`.clip(origin=, normal=, overrun=)` removes the half-space the normal points at.**
The plane is infinite — use it for an angled trim, a level cutoff, or lopping a corner, anywhere a finite cutter box would be noise. Chainable; multiple clips compose. Geometry queries read the AUTHORED shape, so a clip is not reflected in `height_at` / `raycast`.
**On a CONTAINER the cut reaches every part under it**, resolved at the end of authoring — so parts added after the call are cut too, and the line's position in the file does not change what it does. A `Bar` / `Pipe` / path-`Sweep` is TRIMMED rather than subtracted: it stays a bar for schedules and costs no CSG depth; a bar trimmed to nothing leaves the model. A clip on a container holding no geometry is a compile ERROR, not a no-op.
`overrun=` (mm) lets a trimmed PATH reach past the plane — reinforcement laps into the neighbour while the concrete is cut flush on the joint. It applies to path trims ONLY: it never lengthens a bar that already stops short, and the boolean on a solid ignores it. Lap LENGTH is not the compiler's to know.

**`miter(a, b, at=, edge=)` joins a pair flush — an authored joint, so their auto-carve is skipped.**
Both elements are clipped by the SHARED bisector plane through the joint, so each keeps exactly its half and the two faces meet.
`at` is the joint corner. `edge` is the direction of the shared joint EDGE — the line the two elements meet along: `(0, 0, 1)` for two upright leaves at a wall or cladding corner, a horizontal vector for a raked corner or two roof planes meeting at a hip. It has no default, and a zero vector raises immediately.
The cut ANGLE is DERIVED at compile, from the directions `at` → each centroid projected onto the plane perpendicular to `edge`: a 90° corner yields two 45° faces, a 120° corner two 60°. Deriving late is what makes the result independent of where the line sits in the file.
`overrun=` is passed to both clips and means what it means on `.clip()`. Returns `(a, b)`, so `a, b = miter(a, b, at=c, edge=(0, 0, 1))` reads naturally.

**Pin intent: `proj.assert_carved(target, by=…)`.** Fragments resolve like `Anchor(host=)`; the compile fails if the joint broke.

### Holes

**A hole cuts the host's parts too.**
Framing, ties, rebar — `.add()`ed and `.anchor()`ed alike; matter inside a hole is not matter.
The void spans the host BODY's thickness — a part beyond the body (cladding proud of it) is not cut; a body-less host distributes the hole to its leaves instead.
Only `.difference()` leaves parts standing.
`carve="none"` on the `.add()` / `.anchor()` that attached it opts a detail meant to stand in the reveal — a sill, a lintel — back out.
Reinforce with `.add(*bars)`: the one-body rule counts PRISMS (`Box` / `Extrude`), so bars are never a second body.

**`.void(tool, name=)` — the same hole from a world-coordinate solid.**
The tool is CONSUMED; nothing fills the hole — counted, no fill.
Omit `name=` and the opening emits anonymously; give it and the void gets a canonical name derived like any other element's. A NAMED void takes exactly ONE operand — one name cannot identify two holes, so chain the calls instead.
Never author a MEANINGFUL hole with `.difference()`: it is a sculpt, so nothing reaches the schedule.

**`.void()` works on every solid and every container.**
Curved solids (`Sweep` / `Revolve` / `Pipe` / `Bar`) included; on a container the hole lands on every part it overlaps, at any depth, and warns if it overlaps none.
On a `Site` it is a different verb: it clears the TERRAIN mesh — an uncounted subtraction — and never touches what stands on the site.
On a bare `Mesh` and on a `Space` it is refused at compile; sculpt a mesh with `.difference()`.

**`.opening(unit, along=, up=)` — `.anchor()` for a `Window` / `Door`.**
Cuts the hole through the host and every part standing in it, then places the unit as the fill. Counted AND filled, measured in the host frame.
`wall.anchor(win, along=…, up=…)` is the same relation spelled for a `Wall`: the void comes from what the CHILD is, not from which verb placed it, so both stamp the same anchor and cut the same void.

**`.opening()` hosts — everything else is a compile refusal naming the remedy:**

- a `Wall`, with a body or without;
- a standalone `Box`, with `start` / `end`;
- a standalone `Extrude` with a VERTICAL planar contour;
- a `Column` / `Beam` / `Element` assembly, which carries no representation of its own and therefore DISTRIBUTES the hole to the geometry-bearing leaves it overlaps.

A distributed hole is N openings and ONE `IfcWindow` / `IfcDoor` — a fill per leaf would double-count every window in every schedule. The fill sits on the OUTERMOST leaf, the face `inset=` measures from, which is what a window in a buildup physically is.
The usual remedies: a `Slab` / `Roof` is horizontal, so it has no outer face and no run axis for `along` / `inset` / `up` — put the opening on its `.add()`ed body, or use `.void()`; a curved solid → `.void()`; a `Wall`'s body child → the wall itself, same scalars.
A skew host still emits its hole and fill; only the detail-carve warns and skips, so details keep their material.

**The unit carries a SIZE; the host says WHERE.**
Omitted `width` / `height` are inferred from the unit's children, measured from the joinery's own min corner — author joinery at the local origin. State the size when the joinery overflows the hole.
Unit children live in the opening's local frame, identical on every facade.
The void cuts the full thickness; `inset=` sets the unit back.
Glass is an explicit CHILD — `color="glass"` at sketch stage, `Material(key="Glass_VIG", thickness_mm=8)` for real. A pane without it renders OPAQUE; no children at all is a bare hole.

## 7. Measuring frames

Read the frame before you use the number. Mixing frames is the bug.

1. **SELF** — `height_at`, `raycast`, `heightmap_at`. Run on the AUTHORED shape in the element's OWN `placement=`, and nothing above it. Carves are never reflected. Available on the geometric primitives (`Box` / `Extrude` / `Sweep` / `Revolve` / `Pipe` / `Bar` / `Mesh`); containers and annotations reject loudly, and so does an element whose `placement=` is an `Anchor`, which cannot be resolved standalone.
2. **AUTHORED** — `authored_aabb()`. The coordinates as written, before ANY placement.
3. **WORLD** — `world_aabb()`, `obb()`. The full ancestor chain resolved. Raises until the element is reachable from the Project.

A `Box` rotated by a wrapping element's `placement=` is RAYCAST in its unrotated authoring position, so a ray aimed where it actually stands MISSES. To find a surface point on anything an ancestor placed, bound it (`world_aabb()` / `obb()`); do not raycast at its apparent position.
A `Mesh` with `heightmap=` and no `placement=` answers `height_at` / `heightmap_at` from the retained grid — that is how terrain is measured before it is added.

**`max − min` on an OBB is not a width.**
On an oriented box those are two DIAGONAL corners, so `min.z` can exceed `max.z`. Use `.size`, whose components run along / across / up, and `.point_at(i, j, k)`, where `i` / `j` / `k` are int percentages `−100..100` of the distance FROM the centre TO the face: `(0, 0, 0)` is the centre, `±100` is a face, `(100, 100, 100)` a corner. Outside that range is an error, not an extrapolation — reach past the box with arithmetic (`b.point_at(100, 0, 0) + Point(x=300)`) so that on-box and off-box read differently at the call site.
`.exact=False` means the value is a BOUND, not an extent; `.rule` names how it was derived; `.is_axis_aligned` says whether `.min` / `.max` are componentwise extremes at all.

**WORLD reads are recorded.**
Placing A from `B.world_aabb()` notes A→B, and a loop (A→B→A, any length) is REFUSED where it closes. Not because it cannot be solved — it can — but because a ring describes more than one design and statement order would silently pick one. Anchor ONE element to a fixed coordinate and measure outward from it. SELF and AUTHORED reads record nothing and never form a cycle.

## 8. Spaces - derived or authored

A Space is the AIR in a room, never a container for products. Name its bounding elements and the compiler derives the volume; anything else authors its own geometry. Either way the Space must be **named**, it is added to a `Storey` like any other primary element (§9), and `.union()` / `.difference()` compose onto the volume.

```python
kitchen = Space(name="kitchen", bounds=["wall:north:storey:ground", "wall:k1:storey:ground",
                                        "slab:floor:storey:ground", "slab:deck:storey:ground"])  # derived
stairwell = Space(name="stairwell").add(Box(start=Point(x=0, y=0, z=0),
                                            end=Point(x=2400, y=3000, z=2500)))                  # authored
storey.add(kitchen, stairwell)
```

**Derived mode — cite each room's OWN wall segments.**
Bound fragments resolve like `Anchor(host=)`: whole `type:leaf` pairs from the tail, so on a named storey use the full path.
A wall shared full-length by two rooms derives the wrong side for the second.
Adding a bound never GROWS the room: a detached one does not touch, and one lying wholly inside bounds nothing — both warn, and the second is dropped while its boundary still emits.
One bound alone therefore fails, leaving the axes half-open.

**Authored mode — draw it to the INNER faces of its walls.**
It infers its own bounds: they must touch with AREA — flush counts, a 1 mm gap or a shared edge does not — and reach past a face.
A room drawn to the OUTER faces swallows its walls and they bound nothing (a warning, not an error). Bounds-mode rooms derive the inner faces and cannot make this mistake.

**Derived mode reports its volume; authored mode does not.**
A bounds-mode space prints `space <name>: derived W x D x H mm`, and names any bound that only pokes into the volume or lies wholly inside it. An authored space prints nothing.
`bounds=` and `.add()` on one Space are mutually exclusive and raise at the authoring line; `.void()` on a Space is a compile error.
An anonymous Space is refused at compile in derived mode; in authored mode it compiles and silently skips its boundaries.

**Boundaries are derived — never author them.**
The compiler expands `IfcRelSpaceBoundary1stLevel`, one per (space, element) pair including `Window` / `Door` fillings, every one `PHYSICAL`.
Exception: a body-less buildup's filling boundary is skipped, with a warning.
Internal / external stays COMPUTED, never authored: `INTERNAL` a room beyond · `EXTERNAL` nothing · `EXTERNAL_EARTH` terrain · `NOTDEFINED` mixed.
`IfcRelConnectsElements` is never written — overlap produces the carve, and `assert_carved` asserts it.

---

## 9. Site, storeys & grouping

**A `Site` is the ground.**
It holds the terrain, the products standing on it, and the facilities that subdivide it. A facility (`SpatialElement`) composes further and belongs to its Site.
`proj.add(site)` on its own leaves `proj.storeys` empty and compiles to a site-only IFC — no phantom building, no phantom storey.

**A `Storey` is a level, and every building element lives on one.**
`proj.add(*e)` lands on the first storey, auto-creating an anonymous one if none exists; `proj.add_storey(s)` and `proj.storeys[n].add(*e)` are explicit.
One anonymous storey is fine; two or more must ALL be named.
Naming a previously anonymous storey renames everything on it — if a second floor is at all likely, name storeys from day one.

**Aggregate sub-elements inside an intermediate parent.**
Nothing raises here — a storey accepts any element — which is exactly why this has to be a discipline.

- A `Storey` should directly hold only primary building elements: `Wall`, `Slab`, `Roof`, `Column`, `Beam`, `Space`, `Element(ifc_class="IfcStair")`.
- Rebars (`Bar`) go on the host `Wall` / `Slab` — `wall.add(*bars)` — never on the `Storey`.
- Studs, framing (`Element(ifc_class="IfcMember")`) and fixings go on the `Wall`: `wall.add(studwork, fixings)`.
- Multi-part detail clusters — a corner continuity rebar cage, a bracket assembly — go inside an `Element(ifc_class="IfcBuildingElementProxy", name="…")` wrapper, added to the host element or to the storey.
- A `Site` holds terrain geometry (`Mesh` / `Box`) directly, or an `Element(ifc_class="IfcGeographicElement", …)` wrapper.

Aggregation emits `IfcRelAggregates` and collapsible branches in IFC viewers; loose parts on a storey pollute `IfcRelContainedInSpatialStructure` and flood the storey root.

**Grouping states something about elements ALREADY in the tree.**
Geometry-less and never `.add()`ed; the members are canonical names.
`proj.aggregate("north_facade", members=["wall:north:storey:ground", …])` · `proj.zone("thermal", members=["space:kitchen:storey:ground", …])`, which takes Spaces only.
Walls are authored per storey; the facade is the aggregate. Mixing classes in one group is an error.

## 10. Materials & products

**One dimension, one source.**
When `Material` carries the dimension (`thickness_mm` / `profile_mm`), omit the primitive's own — stating both is a compile error.
Its `geometry.form` dictates the primitive: mass → `Box` / `Extrude` · sheet → `Extrude` · member → `Sweep` · bar → `Bar` · membrane → thin `Extrude` · fill → `Box` / `Extrude` · cavity → no solid at all, a `LayerSet` layer only.
`material=None` is sketch stage: `color=` carries the whole appearance.
A registry key attaches material Psets itself — `props=` is for element-level Psets only (exposure class, U-values, fire rating), and floats are legal there because the int-mm gate covers geometry only.
`MATERIALS["Timber_C24"].geometry.stock_profiles_mm` is readable from a model file when you need the stock list.

**`color` takes a hex string or a palette name, on EVERY element and every primitive.**
`#RRGGBB`, `#RGB` shorthand, or `#RRGGBBAA` where the last byte is OPACITY (`"#9E9E9E80"` = 50 % ghost massing, `FF` opaque).
Or one of the 13 palette names listed in §3 — `glass` carries its own transparency.
A value that is neither is a compile ERROR naming the remedy; it does not fall back to grey.

**Layers are a buildup ON one body, not child nodes.**
`layers=LayerSet(layers=[Material…], outward=(dx, dy, dz))` over ONE body sized to the layer sum. Legal only on the planar archetypes: `Wall`, `Slab`, `Roof`, and an `Element` with a planar `ifc_class`.
First layer = OUTER, never reversed.
`outward` points inner→outer — one list serves every facade (S `(0,-1,0)`, N `(0,1,0)`, slab `(0,0,1)`) — and fixes the element's outer face outright, body included.
A band is EITHER `layers=` OR authored geometry, never both.

**Products are finished catalog items.**
Build UNPLACED → promote `P = Product(src, name="nordic_1400x1600")` → place `occ = P.occurrence(name="kitchen_window")`.
No variables, ever: geometric kwargs on `.occurrence()` are an error, and two windows differing by 200 mm ARE two Products.

## 11. Inspecting a model

Compile with `python output.py`, which writes `output.ifc` beside it. For any other filename use `python -m lite_step <file>.py`.

Two flags work on the direct `python <model>.py` form only. They read `sys.argv`, and are deliberately inert under `python -m lite_step` and under a test runner:

- `--check` — validate and report, write NOTHING. Use it while iterating, so a half-finished model never clobbers an IFC someone is looking at.
- `--q <selector>` (or `--q=<selector>`) — dump every bounding accessor for each match: `authored_aabb`, `world_aabb` and `obb`, each with `min`, `max`, `size` and `rule`, plus the reason when a bound is not exact and a note when a box is oriented. The selector matches exactly like `Anchor(host=)` — whole `type:leaf` pairs from the TAIL — so on a named storey `--q wall:north` matches nothing and `--q wall:north:storey:ground` returns the wall AND `box:body:wall:north:storey:ground` under it. No match prints the known names, which is the fastest way to read the canonical names a model actually produced.

Two environment variables widen what a compile says. Each takes `all` for full detail, `0` for silence, anything else or unset for the default summary:

- `LITESTEP_TREE_REPORT` — the container tree the compiler actually built: one row per element, indentation is containment, and the character before the label is the VERB (`@` for `.anchor()`, blank for `.add()`). This is how you catch a leaf that landed on the wrong container. The default collapses runs of identical siblings to `xN` and caps depth, always stating the `+N more` it dropped.
- `LITESTEP_CARVE_REPORT` — default `carves: N pairs`; `all` prints one `carve: <loser> <- <winner>` per pair, sorted. A reordering that changes the joinery changes the count.

`LITESTEP_EMIT_DRAWINGS` is a separate boolean, ON unless set to `0`: it emits the 2D drawing annotations alongside the IFC.

Read the stderr warnings. They are LOUD and non-fatal by design: a degenerate-but-well-formed model warns and still compiles, and only a question the compiler cannot answer correctly raises.

## 12. Execution environment

Lite-STEP executes ordinary Python in a restricted namespace. What is missing is missing at runtime, not merely discouraged.

**Not available:** `setattr`, `delattr`, `globals`, `locals`, `eval`, `exec`, `compile`, `open`, `input`, `breakpoint`, `memoryview`, `help`, `exit`, `quit`, dunder attribute access (including through `getattr`), `os.system` / `os.exec*` / `os.spawn*`, and any import outside `lite_step`, `math`, `numpy`, `random`.

**Available:** read-only introspection (`getattr`, `hasattr`, `type`, `isinstance`, `vars`, `dir`, `callable`), the ordinary iteration, number, string and container builtins, `print`, `math` and `numpy` (as `np`). `arc_path`, `stirrup_path` and `I()` are pre-bound when the DSL is hosted, and the skeleton in §2 imports or defines all three so the same file also runs as a plain script.

The guard runs again whenever a later edit recompiles the stored source — write it clean the first time.
