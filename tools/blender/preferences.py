import bpy
from bpy.types import AddonPreferences
from bpy.props import IntProperty, BoolProperty, StringProperty

class GinnungPreferences(AddonPreferences):
    bl_idname = __package__

    port: IntProperty(
        name="WebSocket Port",
        description="Port for the local WebSocket server",
        default=9876,
        min=1024,
        max=65535,
    ) # type: ignore

    auto_start: BoolProperty(
        name="Auto-Start Server",
        description="Automatically start the server when Blender opens",
        default=False,
    ) # type: ignore

    blend_template: StringProperty(
        name="Presentation Shell (.blend)",
        description="Path to default presentation shell model.blend file (auto-detected if blank)",
        subtype='FILE_PATH',
        default="",
    ) # type: ignore

    execute_embedded_scripts: BoolProperty(
        name="Execute Embedded Sidecar Scripts",
        description="Allow running embedded init_sidecars.py scripts from opened presentation shells",
        default=False,
    ) # type: ignore

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "port")
        layout.prop(self, "auto_start")
        layout.prop(self, "blend_template")
        layout.prop(self, "execute_embedded_scripts")

def get_preferences(context=None):
    if context is None:
        context = bpy.context
    addons = context.preferences.addons
    if __package__ in addons:
        return addons[__package__].preferences
    for addon_name, addon in addons.items():
        if addon_name.endswith("ginnung") or addon_name == "ginnung":
            return addon.preferences
    return None

classes = (
    GinnungPreferences,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)
