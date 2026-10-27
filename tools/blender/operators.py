import bpy
import logging
from .server import server_instance
from .bridge import load_ifc, export_ifc
from .preferences import get_preferences

logger = logging.getLogger(__name__)

class LiteStepStartServerOperator(bpy.types.Operator):
    """Start the Lite-STEP WebSocket server"""
    bl_idname = "litestep.start_server"
    bl_label = "Start Server"

    @classmethod
    def poll(cls, context):
        return not server_instance.is_running()

    def execute(self, context):
        prefs = get_preferences(context)
        port = prefs.port if prefs else 9876
        
        server_instance.on_file_received = load_ifc
        server_instance.start(port=port, version="1.1")
        
        self.report({'INFO'}, f"Lite-STEP Server started on port {port}")
        return {'FINISHED'}

class LiteStepStopServerOperator(bpy.types.Operator):
    """Stop the Lite-STEP WebSocket server"""
    bl_idname = "litestep.stop_server"
    bl_label = "Stop Server"

    @classmethod
    def poll(cls, context):
        return server_instance.is_running()

    def execute(self, context):
        server_instance.stop()
        self.report({'INFO'}, "Lite-STEP Server stopped")
        return {'FINISHED'}

classes = (
    LiteStepStartServerOperator,
    LiteStepStopServerOperator,
)

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)
