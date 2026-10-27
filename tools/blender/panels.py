import bpy
from .server import server_instance

class View3dLiteStepPanel(bpy.types.Panel):
    """Lite-STEP Integration Panel"""
    bl_label = "Lite-STEP"
    bl_idname = "VIEW3D_PT_litestep_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Lite-STEP"

    def draw(self, context):
        layout = self.layout

        # Server Status
        box = layout.box()
        box.label(text="Server Status:")
        
        if server_instance.is_running():
            box.label(text="Running", icon='PLAY')
            box.label(text=f"Port: {server_instance.port}")
            box.label(text=f"Clients: {len(server_instance.active_connections)}")
            box.operator("litestep.stop_server", icon='CANCEL')
        else:
            box.label(text="Stopped", icon='PAUSE')
            box.operator("litestep.start_server", icon='PLAY')

        # Receiver Status
        layout.separator()
        box = layout.box()
        box.label(text="Model Synchronization:")
        if server_instance.is_running():
            box.label(text="Ready to receive IFC pushes", icon='IMPORT')
            box.label(text="Push model from VS Code / CLI")
        else:
            box.label(text="Start server to receive models", icon='INFO')

        # Presentation Shell (.blend) Status
        layout.separator()
        box_shell = layout.box()
        box_shell.label(text="Presentation Shell (.blend):", icon='SCENE_DATA')
        if bpy.data.is_saved and bpy.data.filepath.endswith(".blend"):
            import os
            bname = os.path.basename(bpy.data.filepath)
            box_shell.label(text=f"Active: {bname}", icon='CHECKMARK')
        else:
            from .bridge import find_blend_template
            found = find_blend_template()
            if found:
                import os
                box_shell.label(text=f"Template: {os.path.basename(found)}", icon='FILE_BLEND')
            else:
                box_shell.label(text="Template: None detected", icon='ERROR')

classes = (
    View3dLiteStepPanel,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
