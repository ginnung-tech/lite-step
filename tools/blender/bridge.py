import bpy
import logging
import os
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL_BLEND = "model.blend"
DEFAULT_MODEL_IFC = "model.ifc"
DEFAULT_EXPORT_IFC = "export.ifc"


def _find_template_from_env() -> Optional[str]:
    env_dir = os.environ.get("LITE_STEP_ASSETS_DIR") or os.environ.get("LITE_STEP_PATH")
    if not env_dir:
        return None
    cand = os.path.join(env_dir, DEFAULT_MODEL_BLEND) if not env_dir.endswith(".blend") else env_dir
    if os.path.exists(cand):
        return cand
    cand = os.path.join(env_dir, "lite_step", "assets", DEFAULT_MODEL_BLEND)
    return cand if os.path.exists(cand) else None


def _find_template_relative() -> Optional[str]:
    addon_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(addon_dir, "..", "..", "lite_step", "assets", DEFAULT_MODEL_BLEND),
        os.path.join(addon_dir, "..", "lite_step", "assets", DEFAULT_MODEL_BLEND),
    ]
    for cand in candidates:
        norm = os.path.normpath(cand)
        if os.path.exists(norm):
            return norm
    return None


def _find_template_package_or_home() -> Optional[str]:
    try:
        import lite_step
        if hasattr(lite_step, "__file__") and lite_step.__file__:
            cand = os.path.join(os.path.dirname(lite_step.__file__), "assets", DEFAULT_MODEL_BLEND)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass

    home = os.path.expanduser("~")
    for cand in [
        os.path.join(home, "projects", "lite-step", "lite_step", "assets", DEFAULT_MODEL_BLEND),
        os.path.join(os.getcwd(), "lite_step", "assets", DEFAULT_MODEL_BLEND),
    ]:
        if os.path.exists(cand):
            return cand
    return None


def find_blend_template(preferred_path: Optional[str] = None) -> Optional[str]:
    """
    Locate the presentation shell .blend file (model.blend).
    Checks preferred_path, active Blender file, add-on preferences, and repository assets.
    """
    if (
        isinstance(preferred_path, str)
        and preferred_path.lower().endswith(".blend")
        and os.path.exists(preferred_path)
    ):
        return preferred_path

    if bpy.data.is_saved and bpy.data.filepath and bpy.data.filepath.endswith(".blend"):
        return bpy.data.filepath

    try:
        from .preferences import get_preferences
        prefs = get_preferences()
        if prefs and getattr(prefs, "blend_template", None):
            cand = bpy.path.abspath(prefs.blend_template)
            if os.path.exists(cand):
                return cand
    except Exception:
        pass

    return _find_template_from_env() or _find_template_relative() or _find_template_package_or_home()


def sanitize_filename(filename: Optional[str]) -> str:
    """Validate and sanitize an incoming IFC filename."""
    if not isinstance(filename, str) or not filename.strip():
        return DEFAULT_MODEL_IFC
    cleaned = os.path.basename(filename.replace("\\", "/"))
    if cleaned in ("import.ifc", "output.ifc", ""):
        return DEFAULT_MODEL_IFC
    if not cleaned.lower().endswith(".ifc"):
        raise ValueError("Only .ifc files are accepted.")
    return cleaned


def load_ifc(ifc_bytes: bytes, filename: str = DEFAULT_MODEL_IFC, blend_path: Optional[str] = None) -> None:
    """
    Save the IFC bytes and load it into Blender via the presentation shell .blend file.
    
    1. Locates or opens the model.blend presentation shell (with sky, sun, and PBR materials).
    2. Writes the incoming IFC into the .blend file directory so textures and styles resolve.
    3. Loads the project into the scene via Bonsai.
    4. Executes init_sidecars.py (if enabled) and binds master PBR shaders to imported geometry.
    """
    target_filename = sanitize_filename(filename)
    validated_blend = blend_path if isinstance(blend_path, str) and blend_path.lower().endswith(".blend") else None
    target_blend = find_blend_template(validated_blend)

    if target_blend and os.path.exists(target_blend):
        blend_dir = os.path.dirname(target_blend)
        target_ifc_path = os.path.join(blend_dir, target_filename)
    else:
        temp_dir = tempfile.gettempdir()
        target_ifc_path = os.path.join(temp_dir, target_filename)
        target_blend = None

    try:
        with open(target_ifc_path, "wb") as f:
            f.write(ifc_bytes)
        print(f"Ginnung: Wrote received IFC to {target_ifc_path} ({len(ifc_bytes)} bytes)")
        logger.info(f"Wrote received IFC to {target_ifc_path}")
    except Exception as e:
        print(f"Ginnung Error: Failed to write IFC file: {e}")
        logger.exception("Failed to write IFC file")
        raise

    def do_load() -> None:
        nonlocal target_blend
        # 1. Open presentation shell .blend if not already open
        if target_blend and os.path.exists(target_blend):
            current_fp = os.path.normpath(bpy.data.filepath) if bpy.data.is_saved else ""
            target_fp = os.path.normpath(target_blend)
            if current_fp != target_fp:
                try:
                    print(f"Ginnung: Opening presentation shell: {target_blend}")
                    bpy.ops.wm.open_mainfile(filepath=target_blend)
                except Exception as e:
                    print(f"Ginnung Warning: Could not open {target_blend}: {e}")

        # 2. Load IFC via Bonsai
        if hasattr(bpy.ops.bim, "load_project"):
            try:
                print(f"Ginnung: Loading project into .blend scene via Bonsai: {target_ifc_path}")
                bpy.ops.bim.load_project(filepath=target_ifc_path)
                print(f"Ginnung: Successfully loaded {target_ifc_path} into Blender scene.")
                logger.info(f"Successfully loaded {target_ifc_path} via Bonsai.")
            except Exception as e:
                print(f"Ginnung Error: Bonsai load_project error: {e}")
                logger.exception("Bonsai load_project error")
        elif hasattr(bpy.ops.import_scene, "ifc"):
            try:
                bpy.ops.import_scene.ifc(filepath=target_ifc_path)
            except Exception as e:
                logger.exception("Import IFC error")
        else:
            print("Ginnung Error: Bonsai add-on (bpy.ops.bim.load_project) is not available. Please install and enable Bonsai.")
            logger.error("Bonsai add-on is not available. Cannot load IFC.")

        # 3. Post-load presentation wiring: sidecars, stylesheet, and lighting
        try:
            # Wire BIM project properties
            if hasattr(bpy.context.scene, "BIMProjectProperties"):
                bpy.context.scene.BIMProjectProperties.ifc_file = target_ifc_path

            # Wire CSS stylesheet for 2D floor plans
            css_dir = os.path.dirname(target_ifc_path)
            css_path = os.path.join(css_dir, "styles", "lite-step.css")
            if not os.path.exists(css_path):
                css_path = os.path.join(css_dir, "model.css")
            if os.path.exists(css_path) and hasattr(bpy.context.scene, "BIMDrawingProperties"):
                bpy.context.scene.BIMDrawingProperties.css_file = css_path
                print(f"Ginnung: Wired stylesheet: {css_path}")

            # Execute init_sidecars.py script block from .blend shell if user enabled it
            from .preferences import get_preferences
            prefs = get_preferences()
            can_exec = prefs and getattr(prefs, "execute_embedded_scripts", False)
            if can_exec and "init_sidecars.py" in bpy.data.texts:
                try:
                    script_src = bpy.data.texts["init_sidecars.py"].as_string()
                    exec(script_src, {"bpy": bpy, "__file__": target_ifc_path})
                    print("Ginnung: Executed embedded init_sidecars.py from .blend shell.")
                except Exception as e:
                    print(f"Ginnung: Warning executing init_sidecars.py: {e}")
                    logger.exception("Warning executing embedded init_sidecars.py")

            # 4. Remap imported materials to shell's master PBR materials
            remapped_count = 0
            for obj in bpy.data.objects:
                if not obj.data or not hasattr(obj.data, "materials"):
                    continue
                for slot_idx, mat in enumerate(obj.data.materials):
                    if not mat:
                        continue
                    base_name = mat.name.split(".")[0]
                    if base_name in bpy.data.materials and mat.name != base_name:
                        obj.data.materials[slot_idx] = bpy.data.materials[base_name]
                        remapped_count += 1
            if remapped_count > 0:
                print(f"Ginnung: Remapped {remapped_count} material slots to master PBR materials.")

            # 5. Switch viewport to Material Preview to immediately reveal textures & sun
            for area in bpy.context.screen.areas:
                if area.type == 'VIEW_3D':
                    for space in area.spaces:
                        if space.type == 'VIEW_3D':
                            space.shading.type = 'MATERIAL'

        except Exception as e:
            print(f"Ginnung Warning: Post-load setup error: {e}")
            logger.exception("Post-load setup error")

        return None

    # Register the callback to run on the main thread
    bpy.app.timers.register(do_load)


def export_ifc(filename: str = DEFAULT_EXPORT_IFC) -> bytes:
    """
    Export the current Blender scene to an IFC file via Bonsai.
    This must be called from the main thread (e.g. within an operator).
    Returns the raw bytes of the exported IFC.
    """
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, filename)

    if hasattr(bpy.ops.export_ifc, "ifc"):
        try:
            bpy.ops.export_ifc.ifc(filepath=temp_path)
            logger.info(f"Successfully exported scene to {temp_path} via Bonsai.")
            if os.path.exists(temp_path):
                with open(temp_path, "rb") as f:
                    return f.read()
            logger.error("Exported IFC file was not found.")
        except Exception as e:
            logger.exception("Bonsai export error")
    else:
        logger.error("Bonsai add-on is not available. Cannot export IFC.")

    return b""
