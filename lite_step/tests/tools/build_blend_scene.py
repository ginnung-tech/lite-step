"""
Build the model.blend presentation shell for the Lite-STEP IFC model.
Executed via:
    blender -b --factory-startup --python lite_step/tests/tools/build_blend_scene.py
"""

import importlib.util
import math
import sys
from pathlib import Path

# Resolve repo root relative to this script, ensuring paths resolve regardless of CWD
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Ensure pydantic and project dependencies are importable in Blender's bundled Python
if not importlib.util.find_spec("pydantic"):
    import site
    for candidate in [
        Path(sys.executable).parent / "Lib" / "site-packages",
        Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python313" / "Lib" / "site-packages",
        Path.home() / "AppData" / "Local" / "Programs" / "Python" / "Python314" / "Lib" / "site-packages",
        Path.home() / ".local" / "lib" / "python3.13" / "site-packages",
        Path.home() / ".local" / "lib" / "python3.14" / "site-packages",
    ]:
        if candidate.exists():
            site.addsitedir(str(candidate))
            if importlib.util.find_spec("pydantic"):
                break

try:
    import bpy  # noqa: E402
except ImportError:
    bpy = None
from lite_step.materials.dk import MATERIALS  # noqa: E402


def get_material_pbr_config(mat_name: str, mdef) -> dict:
    """Derive PBR shader parameters from MaterialDef properties."""
    # Finish -> Roughness
    finish = mdef.render.finish if mdef.render else "matt"
    if finish == "gloss":
        roughness = 0.25
    elif finish == "satin":
        roughness = 0.45
    elif finish == "metallic":
        roughness = 0.30
    else:
        roughness = 0.80

    # Category -> Metallic
    is_metallic = (mdef.category == "metal" or finish == "metallic")
    metallic = 1.0 if is_metallic else 0.0

    # Geometry form and category -> UV mapping metrics & bump
    form = mdef.geometry.form if hasattr(mdef.geometry, "form") else "mass"

    if form == "member" or mdef.category == "timber":
        scale = (1.0 / 0.30, 1.0 / 0.30, 1.0 / 2.00)
        rotation = (0.0, math.radians(90.0), 0.0)  # longitudinal grain
        bump = 0.15
    elif mdef.category == "masonry":
        scale = (1.0 / 1.20, 1.0 / 1.20, 1.0 / 0.60)
        rotation = (0.0, 0.0, 0.0)
        bump = 0.30
    elif mdef.category == "concrete":
        scale = (1.0 / 2.00, 1.0 / 2.00, 1.0 / 2.00)
        rotation = (0.0, 0.0, 0.0)
        bump = 0.25
    elif mdef.category == "ceramic":
        scale = (1.0 / 1.00, 1.0 / 1.00, 1.0 / 1.00)
        rotation = (0.0, 0.0, 0.0)
        bump = 0.20
        roughness = 0.30
    elif mdef.category == "aggregate":
        scale = (1.0 / 1.00, 1.0 / 1.00, 1.0 / 1.00)
        rotation = (0.0, 0.0, 0.0)
        bump = 0.40
        roughness = 0.90
    elif mdef.category == "metal":
        if form == "bar":
            scale = (1.0 / 0.50, 1.0 / 0.50, 1.0 / 0.50)
            bump = 0.30
            roughness = 0.70
            metallic = 0.80
        else:
            scale = (1.0 / 1.00, 1.0 / 1.00, 1.0 / 1.00)
            bump = 0.05
            roughness = 0.30
            metallic = 0.90
        rotation = (0.0, 0.0, 0.0)
    else:
        scale = (1.0, 1.0, 1.0)
        rotation = (0.0, 0.0, 0.0)
        bump = 0.10

    return {
        "texture": mdef.texture,
        "scale": scale,
        "rotation": rotation,
        "roughness": roughness,
        "bump": bump,
        "metallic": metallic,
    }


def setup_scene():
    if bpy is None:
        raise RuntimeError("setup_scene must be executed within Blender (bpy not found)")
    bpy.ops.wm.read_factory_settings(use_empty=True)

    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'

    # 1. Setup World with Golden Hour Atmospheric Sky
    world = bpy.data.worlds.new("GoldenHour_World")
    scene.world = world
    world_tree = world.node_tree
    world_tree.nodes.clear()

    node_out = world_tree.nodes.new(type='ShaderNodeOutputWorld')
    node_out.location = (500, 0)

    node_bg = world_tree.nodes.new(type='ShaderNodeBackground')
    node_bg.location = (250, 0)
    node_bg.inputs['Strength'].default_value = 1.0

    try:
        node_sky = world_tree.nodes.new(type='ShaderNodeTexSky')
        node_sky.location = (0, 0)
        # In Blender 5.2, multiple scattering atmospheric model
        node_sky.sky_type = 'MULTIPLE_SCATTERING'
        node_sky.sun_elevation = math.radians(11.0)
        node_sky.sun_rotation = math.radians(130.0)
        node_sky.altitude = 0.0
        node_sky.air_density = 1.0
        node_sky.aerosol_density = 1.5
        node_sky.ozone_density = 1.0
        world_tree.links.new(node_sky.outputs['Color'], node_bg.inputs['Color'])
        print("[Lite-STEP] Configured MULTIPLE_SCATTERING golden hour sky.")
    except Exception as e:  # noqa: BLE001
        print("[Lite-STEP] Fallback to atmospheric background color:", e)
        node_bg.inputs['Color'].default_value = (0.95, 0.75, 0.55, 1.0)

    world_tree.links.new(node_bg.outputs['Background'], node_out.inputs['Surface'])

    # 2. Setup Lighting Rig Collection
    lighting_col = bpy.data.collections.new("Auto_Lighting")
    scene.collection.children.link(lighting_col)

    # Golden Hour Sun
    sun_data = bpy.data.lights.new(name="GoldenHour_Sun", type='SUN')
    sun_data.energy = 5.0
    sun_data.color = (1.0, 0.72, 0.42)  # ~3200K warm golden sunlight
    sun_data.angle = math.radians(2.0)   # Soft penumbra

    sun_obj = bpy.data.objects.new("GoldenHour_Sun", sun_data)
    sun_obj.location = (0, 0, 25)
    rot_x = math.radians(90.0 - 11.0)
    rot_z = math.radians(130.0)
    sun_obj.rotation_euler = (rot_x, 0.0, rot_z)
    lighting_col.objects.link(sun_obj)

    # 3. Build Triplanar Box-Mapped PBR Materials directly driven by dk.py MATERIALS
    assets_dir = (ROOT / "lite_step" / "assets").resolve()
    textures_dir = assets_dir / "textures"
    if not textures_dir.exists():
        raise FileNotFoundError(f"Textures directory missing: {textures_dir}")

    loaded_images = {}
    material_count = 0

    for mat_name, mdef in MATERIALS.items():
        if not mdef.texture:
            continue

        cfg = get_material_pbr_config(mat_name, mdef)
        tex_filename = cfg["texture"]
        tex_abs = textures_dir / tex_filename
        if not tex_abs.exists():
            raise FileNotFoundError(
                f"Texture '{tex_filename}' referenced by material '{mat_name}' not found at: {tex_abs}"
            )

        mat = bpy.data.materials.new(name=mat_name)
        mat.use_fake_user = True
        tree = mat.node_tree
        tree.nodes.clear()

        # Output
        out_node = tree.nodes.new('ShaderNodeOutputMaterial')
        out_node.location = (600, 0)

        # Principled BSDF
        bsdf = tree.nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.location = (300, 0)
        bsdf.inputs['Roughness'].default_value = cfg['roughness']
        bsdf.inputs['Metallic'].default_value = cfg['metallic']
        tree.links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])

        # Texture Coordinate
        tex_coord = tree.nodes.new('ShaderNodeTexCoord')
        tex_coord.location = (-600, 0)

        # Mapping
        mapping = tree.nodes.new('ShaderNodeMapping')
        mapping.location = (-400, 0)
        mapping.inputs['Scale'].default_value = cfg['scale']
        mapping.inputs['Rotation'].default_value = cfg['rotation']
        tree.links.new(tex_coord.outputs['Object'], mapping.inputs['Vector'])

        # Image Texture (Box Mapping)
        tex_image = tree.nodes.new('ShaderNodeTexImage')
        tex_image.location = (-150, 0)
        tex_image.projection = 'BOX'
        tex_image.projection_blend = 0.15

        if tex_filename not in loaded_images:
            img = bpy.data.images.load(filepath=str(tex_abs))
            # Set relative path so it resolves anywhere relative to model.blend
            img.filepath = f"//textures/{tex_filename}"
            loaded_images[tex_filename] = img

        tex_image.image = loaded_images[tex_filename]

        tree.links.new(mapping.outputs['Vector'], tex_image.inputs['Vector'])
        tree.links.new(tex_image.outputs['Color'], bsdf.inputs['Base Color'])

        # Bump node for surface relief
        if cfg.get('bump', 0) > 0:
            bump = tree.nodes.new('ShaderNodeBump')
            bump.location = (100, -200)
            bump.inputs['Strength'].default_value = cfg['bump']
            bump.inputs['Distance'].default_value = 0.005
            tree.links.new(tex_image.outputs['Color'], bump.inputs['Height'])
            tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])

        material_count += 1

    print(f"[Lite-STEP] Built {material_count} PBR materials driven by MATERIALS.")

    # Glass Material
    glass_mat = bpy.data.materials.new(name="Glass_Monolithic")
    glass_mat.use_fake_user = True
    gtree = glass_mat.node_tree
    gtree.nodes.clear()
    g_out = gtree.nodes.new('ShaderNodeOutputMaterial')
    g_out.location = (400, 0)
    g_bsdf = gtree.nodes.new('ShaderNodeBsdfPrincipled')
    g_bsdf.location = (100, 0)
    g_bsdf.inputs['Roughness'].default_value = 0.05
    g_bsdf.inputs['IOR'].default_value = 1.52
    if 'Transmission Weight' in g_bsdf.inputs:
        g_bsdf.inputs['Transmission Weight'].default_value = 1.0
    elif 'Transmission' in g_bsdf.inputs:
        g_bsdf.inputs['Transmission'].default_value = 1.0
    gtree.links.new(g_bsdf.outputs['BSDF'], g_out.inputs['Surface'])

    # 4. Embedded Setup Script init_sidecars.py
    script_text = """\
# init_sidecars.py — Automatic setup script for Lite-STEP Bonsai / Blender presentation shell
import bpy
import math
from pathlib import Path
from mathutils import Vector

# Real-world texture metrics (meters) and grain alignment rules
TEXTURE_METRICS = {
    "Brick_Red_DK": {"w": 1.20, "h": 0.60, "mode": "WALL_RUN"},
    "Timber_C24":   {"w": 0.30, "h": 2.00, "mode": "LONGITUDINAL"},
    "Glulam_GL24h": {"w": 0.30, "h": 2.00, "mode": "LONGITUDINAL"},
    "Glulam_GL28h": {"w": 0.30, "h": 2.00, "mode": "LONGITUDINAL"},
    "Concrete_C30-37": {"w": 2.00, "h": 2.00, "mode": "WORLD_XY"},
    "Concrete_C25-30": {"w": 2.00, "h": 2.00, "mode": "WORLD_XY"},
    "ClayTile_Black_Engobed": {"w": 1.00, "h": 1.00, "mode": "WORLD_XY"},
    "Gravel_16-32": {"w": 1.00, "h": 1.00, "mode": "WORLD_XY"},
}

def sync_sidecars():
    blend_dir = Path(bpy.path.abspath("//"))
    ifc_path = blend_dir / "model.ifc"
    css_path = blend_dir / "styles" / "lite-step.css"
    if not css_path.exists():
        fallback = blend_dir / "lite-step.css"
        if fallback.exists():
            css_path = fallback

    # Wire IFC project path into Bonsai if available
    if hasattr(bpy.context.scene, "BIMProjectProperties"):
        props = bpy.context.scene.BIMProjectProperties
        if not props.ifc_file and ifc_path.exists():
            props.ifc_file = str(ifc_path)
            print(f"[Lite-STEP] Wired IFC path: {ifc_path}")

    # Wire CSS path into Bonsai Drawing Properties
    if hasattr(bpy.context.scene, "BIMDrawingProperties"):
        draw_props = bpy.context.scene.BIMDrawingProperties
        if css_path.exists():
            draw_props.css_file = str(css_path)
            print(f"[Lite-STEP] Wired CSS path: {css_path}")

def auto_align_element_texture(obj):
    if not obj.data or not hasattr(obj.data, "materials"):
        return

    for mat in obj.data.materials:
        if not mat or not mat.use_nodes:
            continue

        meta = TEXTURE_METRICS.get(mat.name)
        if not meta:
            continue

        mapping_node = next((n for n in mat.node_tree.nodes if n.type == 'MAPPING'), None)
        if not mapping_node:
            continue

        scale_x = 1.0 / meta["w"]
        scale_y = 1.0 / meta["w"]
        scale_z = 1.0 / meta["h"]
        mapping_node.inputs['Scale'].default_value = (scale_x, scale_y, scale_z)

        if meta["mode"] == "LONGITUDINAL":
            mapping_node.inputs['Rotation'].default_value[1] = math.radians(90.0)
        elif meta["mode"] == "WALL_RUN":
            mapping_node.inputs['Rotation'].default_value = (0.0, 0.0, 0.0)

def clear_existing_rig(collection_name="Auto_Lighting"):
    col = bpy.data.collections.get(collection_name)
    if col:
        for obj in list(col.objects):
            if obj.type == 'LIGHT' and obj.name != "GoldenHour_Sun":
                bpy.data.objects.remove(obj, do_unlink=True)
    else:
        col = bpy.data.collections.new(collection_name)
        bpy.context.scene.collection.children.link(col)
    return col

def add_space_interior_lights(collection):
    spaces = [
        o for o in bpy.data.objects 
        if "IfcSpace" in o.name or (hasattr(o, "BIMObjectProperties") and o.BIMObjectProperties.ifc_definition_id)
    ]
    if not spaces:
        spaces = [o for o in bpy.data.objects if o.type == 'MESH' and "Storey" in o.name]

    for obj in spaces:
        try:
            bbox_corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
            min_x = min(c.x for c in bbox_corners)
            max_x = max(c.x for c in bbox_corners)
            min_y = min(c.y for c in bbox_corners)
            max_y = max(c.y for c in bbox_corners)
            min_z = min(c.z for c in bbox_corners)
            max_z = max(c.z for c in bbox_corners)

            center_x = (min_x + max_x) / 2.0
            center_y = (min_y + max_y) / 2.0
            mount_z = max_z - 0.20

            light_data = bpy.data.lights.new(name=f"Light_{obj.name}", type='POINT')
            light_data.energy = 60.0
            light_data.color = (1.0, 0.62, 0.35)  # ~2700K warm indoor glow
            light_data.shadow_soft_size = 0.2

            light_obj = bpy.data.objects.new(f"Light_{obj.name}", light_data)
            light_obj.location = (center_x, center_y, mount_z)
            collection.objects.link(light_obj)
        except Exception:
            pass

def run_lighting_rig():
    rig_col = clear_existing_rig()
    add_space_interior_lights(rig_col)

# Execute setup
sync_sidecars()
run_lighting_rig()

# Register handler for future file loads
if sync_sidecars not in bpy.app.handlers.load_post:
    bpy.app.handlers.load_post.append(sync_sidecars)
"""

    text_block = bpy.data.texts.new(name="init_sidecars.py")
    text_block.write(script_text)
    text_block.use_module = True

    out_blend = (assets_dir / "model.blend").resolve()
    bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
    print(f"[Lite-STEP] Successfully created {out_blend} ({out_blend.stat().st_size} bytes)")
    sys.exit(0)


if __name__ == "__main__":
    setup_scene()
