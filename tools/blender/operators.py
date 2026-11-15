import bpy
import logging
from .server import server_instance
from .bridge import load_ifc, export_ifc
from .preferences import get_preferences
from .origin_policy import split_entries

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
        
        origins = split_entries(prefs.allowed_origins) if prefs else []

        server_instance.on_file_received = load_ifc
        server_instance.start(port=port, version="1.3", allowed_origins=origins)

        for problem in server_instance.rejected_origins:
            self.report({'WARNING'}, f"Lite-STEP: ignored allowed-origin entry {problem}")
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
