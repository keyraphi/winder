from . import properties, operators, ui, handlers

bl_info = {
    "name": "Winder GPU Winding Number & Field Optimizer",
    "author": "Raphael Braun",
    "version": (1, 0, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Winder Viz",
    "description": "Interactive GPU Winding Number Fields, Gradient Vector Fields, and Geometry Optimization",
    "category": "3D View",
}


def register():
    properties.register()
    operators.register()
    ui.register()
    handlers.register_handlers()


def unregister():
    handlers.unregister_handlers()
    ui.unregister()
    operators.unregister()
    properties.unregister()


if __name__ == "__main__":
    register()
