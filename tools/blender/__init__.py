bl_info = {
    "name": "Lite-STEP Bridge",
    "description": "Bridge Lite-STEP models to Blender via WebSocket and IFC (requires Bonsai)",
    "author": "Lite-STEP Contributors",
    "version": (1, 1, 0),
    "blender": (5, 0, 0),
    "location": "View3D > N-Panel > Lite-STEP",
    "warning": "Requires Bonsai (formerly BlenderBIM) add-on for IFC import/export",
    "category": "Import-Export",
}

import bpy
from . import preferences
from . import operators
from . import panels
from .server import server_instance

def register():
    preferences.register()
    operators.register()
    panels.register()
    
    # Auto-start logic
    try:
        prefs = preferences.get_preferences()
        if prefs and prefs.auto_start:
            from .bridge import load_ifc
            server_instance.on_file_received = load_ifc
            server_instance.start(port=prefs.port, version="1.1")
    except Exception as e:
        print(f"Lite-STEP: Failed to auto-start server - {e}")

def unregister():
    if server_instance.is_running():
        server_instance.stop()
        
    panels.unregister()
    operators.unregister()
    preferences.unregister()

if __name__ == "__main__":
    register()
