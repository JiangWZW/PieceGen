# <pep8 compliant>
# -*- coding: utf-8 -*-

# Main Addon File: PieceGen/__init__.py
# Handles registration, UI Panel, depsgraph handler.

bl_info = {
    "name": "PieceGen RMF Curve Deform", # Updated Name
    "author": "AI Assistant (Gemini) & User",
    "version": (1, 24, 0), # Incremented for refactor v4 (cleaner structure)
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N Panel) > PieceGen Tab", # Updated Category
    "description": "Generates cylinder/curve pair, applies Python curve deform with RMF.",
    "warning": "PYTHON-BASED REALTIME DEFORM CAN BE SLOW FOR DENSE MESHES!",
    "doc_url": "",
    "category": "Object", # Changed Category
}

# --- Standard Imports ---
import bpy
import bmesh
from mathutils import Vector, Matrix # Keep if needed by Visualize Op or Panel

# --- Local Addon Imports ---
# Import necessary components from other addon files
from .common_imports import HAS_NUMPY, np # Import HAS_NUMPY flag
from . import common_vars as cvars # Import constants and global state
from .prop_serialization import unpack_verts_from_string # Import specific helper
from . import bezier # Import bezier/RMF functions
from .op_generate import OBJECT_OT_generate_cylinder_with_curve # Import Operators
from .op_deform import (
    OBJECT_OT_enable_realtime_bend,
    OBJECT_OT_disable_realtime_bend,
    deform_mesh_along_curve # Import the core deform function for the handler
)

# --- Scene Properties ---
# Define properties used by the UI panel for object generation
class CCDG_Properties(bpy.types.PropertyGroup):
    """Stores settings for the Object Generation Panel."""
    # --- PASTE PROPERTY DEFINITIONS FROM YOUR PREVIOUS __init__.py HERE ---
    num_cap_verts: bpy.props.IntProperty(name="Cap Vertices", description="Num vertices in cylinder cap ring", default=32, min=3, max=256)
    cap_radius: bpy.props.FloatProperty(name="Radius", description="Cylinder radius", default=1.0, min=0.01, unit='LENGTH', precision=3)
    height: bpy.props.FloatProperty(name="Height", description="Cylinder height (local Z)", default=2.0, min=0.01, unit='LENGTH', precision=3)
    num_height_segs: bpy.props.IntProperty(name="Height Segments", description="Num subdivisions along cylinder height", default=10, min=1, max=200)
    num_cpts_curve: bpy.props.IntProperty(name="Curve Points", description="Num control points for initial Bézier curve", default=4, min=2, max=32)
    cap_fill_type: bpy.props.EnumProperty(name="Cap Fill Type", items=[('NGON', "N-Gon", "N-Gon"), ('TRIANGLE_FAN', "Tri Fan", "Tri Fan"), ('NOTHING', "Nothing", "Nothing")], default='NGON', description="How to fill cylinder caps")


# --- Depsgraph Update Handler ---
# This function runs after the dependency graph has been evaluated
# @bpy.app.handlers.persistent # Uncomment if handler needs to persist across file loads
def ccdg_depsgraph_handler(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph):
    """Checks for updates to monitored curves and triggers deformation on linked meshes."""

    # Optimization: Exit early if no meshes are being monitored
    if not cvars.MONITORED_MESH_OBJECTS:
        return

    # --- Identify Updated Curves Relevant to Monitored Meshes ---
    # Create a set of curve object *names* that were updated in this cycle
    # and are actually linked to one of the monitored meshes.
    relevant_updated_curve_names = set()
    monitored_curve_map = {} # Map: curve_name -> list[mesh_name]

    # Build a quick lookup map of which curves are monitored
    for mesh_name in cvars.MONITORED_MESH_OBJECTS:
        mesh_obj = scene.objects.get(mesh_name)
        if mesh_obj and mesh_obj.get(cvars.PROP_ENABLED):
            curve_name = mesh_obj.get(cvars.PROP_CURVE_NAME)
            if curve_name:
                if curve_name not in monitored_curve_map:
                    monitored_curve_map[curve_name] = []
                monitored_curve_map[curve_name].append(mesh_name)

    if not monitored_curve_map: # No valid monitored curves found
        cvars.MONITORED_MESH_OBJECTS.clear() # Clear the set if all links are broken
        return

    # Check dependency graph updates for changes to monitored curves
    for update in depsgraph.updates:
        if not hasattr(update.id, "original"): continue # Skip updates without original ID

        obj_or_data = update.id.original

        # Check if the update is for a Curve Object that we monitor
        if isinstance(obj_or_data, bpy.types.Object) and obj_or_data.name in monitored_curve_map:
            if update.is_updated_geometry or update.is_updated_transform:
                 relevant_updated_curve_names.add(obj_or_data.name)

        # Check if the update is for Curve Data used by a monitored Curve Object
        elif isinstance(obj_or_data, bpy.types.Curve) and update.is_updated_geometry:
             # Find all curve objects using this data and check if they are monitored
             for curve_name, mesh_names in monitored_curve_map.items():
                  curve_obj_check = scene.objects.get(curve_name)
                  if curve_obj_check and curve_obj_check.data == obj_or_data:
                       relevant_updated_curve_names.add(curve_name)
                       # No need to check other meshes for the same curve data update

    # Exit if no relevant curves were updated
    if not relevant_updated_curve_names:
        return

    # --- Deform Meshes Linked to Updated Curves ---
    # print(f"Handler triggered for curves: {relevant_updated_curve_names}") # Debug
    for curve_name in relevant_updated_curve_names:
        curve_obj = scene.objects.get(curve_name)
        if not curve_obj: continue # Should exist if in map, but check anyway

        # Deform all meshes linked to this specific curve
        if curve_name in monitored_curve_map:
            for mesh_name in monitored_curve_map[curve_name]:
                mesh_obj = scene.objects.get(mesh_name)

                # Final validation before deforming
                if not mesh_obj or mesh_obj.type != 'MESH' or not mesh_obj.get(cvars.PROP_ENABLED):
                    cvars.MONITORED_MESH_OBJECTS.discard(mesh_name) # Clean up broken link
                    cvars.original_coords_cache.pop(mesh_name, None)
                    continue

                # Retrieve data needed for deformation
                packed_verts = mesh_obj.get(cvars.PROP_ORIG_VERTS)
                cyl_height = mesh_obj.get(cvars.PROP_HEIGHT)

                if packed_verts and cyl_height is not None:
                    # Get original coords from cache or unpack if needed
                    original_verts = cvars.original_coords_cache.get(mesh_name)
                    if not original_verts:
                        original_verts = unpack_verts_from_string(packed_verts) # From prop_serialization
                        if original_verts:
                            cvars.original_coords_cache[mesh_name] = original_verts # Re-populate cache
                        else:
                            print(f"Error Handler: Could not unpack original verts for {mesh_name}")
                            continue # Skip deformation if data invalid

                    # Check vertex count consistency
                    if len(original_verts) == len(mesh_obj.data.vertices):
                        # Call the deform function (imported from op_deform)
                        # print(f"  Deforming {mesh_name} using {curve_name}") # Debug
                        deform_mesh_along_curve(mesh_obj, curve_obj, original_verts, cyl_height)
                    else:
                        print(f"Error Handler: Vertex count mismatch for {mesh_name}. Disabling deform.")
                        # Optionally auto-disable here if vertex count changes
                        # mesh_obj[cvars.PROP_ENABLED] = False
                        # cvars.MONITORED_MESH_OBJECTS.discard(mesh_name)
                        # cvars.original_coords_cache.pop(mesh_name, None)

                else:
                    print(f"Error Handler: Missing properties for {mesh_name} ({cvars.PROP_ORIG_VERTS=}, {cvars.PROP_HEIGHT=})")


# --- UI Panel ---
# Defines the panel in the 3D Viewport Sidebar
class VIEW3D_PT_piecegen_panel(bpy.types.Panel): # Renamed Panel ID
    """UI Panel for the PieceGen RMF Curve Deform addon"""
    bl_label = "PieceGen Deform"; bl_idname = "VIEW3D_PT_piecegen_deform" # Renamed ID
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'PieceGen' # New Category

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        active_obj = context.active_object

        # Check if the property group exists on the scene
        if not hasattr(scene, 'ccdg_props'):
            layout.label(text="Error: Addon properties not found!", icon='ERROR')
            return
        props = scene.ccdg_props # Get properties from scene

        # --- Section 1: Object Generation ---
        box_gen = layout.box()
        col_gen = box_gen.column(align=True)
        col_gen.label(text="1. Object Generation Settings:")
        # Draw properties from the property group
        col_gen.prop(props, "num_cap_verts")
        col_gen.prop(props, "cap_radius")
        col_gen.prop(props, "height")
        col_gen.prop(props, "num_height_segs")
        col_gen.prop(props, "cap_fill_type")
        col_gen.prop(props, "num_cpts_curve")
        # Draw the operator button, using bl_idname from the imported operator class
        col_gen.operator(OBJECT_OT_generate_cylinder_with_curve.bl_idname, icon='MESH_CYLINDER')

        layout.separator()

        # --- Section 2: Realtime Python Deformer ---
        box_rt = layout.box()
        col_rt = box_rt.column(align=True)
        col_rt.label(text="2. Realtime Python Deformer:")

        # Show status/controls only if a mesh object is active
        if active_obj and active_obj.type == 'MESH':
            # Check if deformation is enabled for this object using custom property
            is_enabled = active_obj.get(cvars.PROP_ENABLED, False) # Use imported constant
            linked_curve_name = active_obj.get(cvars.PROP_CURVE_NAME, "None") # Use imported constant

            if is_enabled:
                # Show status and disable button if enabled
                row = col_rt.row(align=True)
                row.label(text=f"Active: Linked to '{linked_curve_name}'", icon='CHECKMARK')
                # Draw disable operator button, using bl_idname from imported class
                row.operator(OBJECT_OT_disable_realtime_bend.bl_idname, text="", icon='X')
                col_rt.label(text="Edit linked curve for realtime updates.", icon='INFO')
                col_rt.label(text="WARNING: Can be slow!", icon='ERROR')
            else:
                 # Show enable button if not enabled
                 # Draw enable operator button, using bl_idname from imported class
                col_rt.operator(OBJECT_OT_enable_realtime_bend.bl_idname, icon='MOD_CURVE')
                col_rt.label(text="Select Mesh (Active) and Curve, then Enable.", icon='INFO')
        else:
            # Guide user if no mesh is active
            col_rt.label(text="Select Mesh Object to see status.", icon='INFO')


# --- Visualization Operator (Optional - Kept for debugging) ---
# This operator remains independent and useful for inspecting RMF frames.
class VISUALIZE_OT_curve_frames(bpy.types.Operator):
    """Visualizes the Tangent, Normal, and Binormal RMF frames along the active curve"""
    # --- PASTE VISUALIZE OPERATOR CODE FROM YOUR __init__.py HERE ---
    # --- Ensure calls use bezier. prefix where needed ---
    bl_idname = "object.visualize_curve_frames"; bl_label = "Visualize RMF Frames"
    bl_options = {'REGISTER', 'UNDO'}
    num_steps: bpy.props.IntProperty(name="Steps",default=20,min=2,max=200)
    scale: bpy.props.FloatProperty(name="Vector Scale",default=0.2,min=0.01,max=10.0)
    vector_radius: bpy.props.FloatProperty(name="Vector Radius",default=0.015,min=0.001,max=1.0)

    @classmethod
    def poll(cls, context):
        # Only run on active curve object
        return context.active_object and context.active_object.type == 'CURVE'

    def create_gradient_material(self, name, base_color):
        """Creates a node-based material with a gradient along local Z."""
        # --- PASTE create_gradient_material method HERE ---
        mat = bpy.data.materials.get(name);
        if mat is None: mat = bpy.data.materials.new(name=name)
        else:
            if mat.node_tree: mat.node_tree.nodes.clear()
        mat.use_nodes = True; nodes = mat.node_tree.nodes; links = mat.node_tree.links; nodes.clear()
        tex_coord = nodes.new(type='ShaderNodeTexCoord'); separate_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
        color_ramp = nodes.new(type='ShaderNodeValToRGB'); principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_node = nodes.new(type='ShaderNodeOutputMaterial')
        # Configure Color Ramp (Base -> Base @ 0.7 -> White @ 0.8 for more base color)
        color_ramp.color_ramp.elements[0].position = 0.0
        color_ramp.color_ramp.elements[0].color = (*base_color[:3], 1.0)
        color_ramp.color_ramp.elements[1].position = 0.8 # Adjust white position
        color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        new_stop = color_ramp.color_ramp.elements.new(position=0.7) # Add stop at 70%
        new_stop.color = (*base_color[:3], 1.0) # Make it the base color
        # Node positioning (optional)
        tex_coord.location = (-600, 0); separate_xyz.location = (-400, 0); color_ramp.location = (-200, 0)
        principled_bsdf.location = (0, 0); output_node.location = (200, 0)
        # Linking nodes
        links.new(tex_coord.outputs['Generated'], separate_xyz.inputs['Vector'])
        links.new(separate_xyz.outputs['Z'], color_ramp.inputs['Fac'])
        links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Base Color'])
        links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])
        return mat

    def execute(self, context):
        """Calculates and visualizes RMF frames."""
        # --- PASTE execute method HERE, using bezier. functions ---
        curve_obj = context.active_object; curve_data = curve_obj.data
        if not curve_data.splines or len(curve_data.splines) == 0 or curve_data.splines[0].type != 'BEZIER':
            self.report({'ERROR'}, "Active object's first spline must be BEZIER type."); return {'CANCELLED'}
        spline = curve_data.splines[0]
        if len(spline.bezier_points) < 2:
             self.report({'ERROR'}, "Bezier spline needs at least 2 control points for visualization."); return {'CANCELLED'}

        # Create materials
        mat_tangent  = self.create_gradient_material("Vis_Tangent_Grad", (1.0, 0.0, 0.0))
        mat_normal   = self.create_gradient_material("Vis_Normal_Grad", (0.0, 1.0, 0.0))
        mat_binormal = self.create_gradient_material("Vis_Binormal_Grad", (0.0, 0.0, 1.0))

        # Create parent empty for markers, cleaning up previous if exists
        old_empty_name = f"{curve_obj.name}_FramesViz"; old_empty = bpy.data.objects.get(old_empty_name)
        if old_empty:
            # More robust deletion using bpy.ops with selection context override
            objs_to_delete = list(old_empty.children) + [old_empty]
            bpy.ops.object.select_all(action='DESELECT')
            for obj in objs_to_delete:
                obj.select_set(True)
            bpy.ops.object.delete() # Delete selected

        viz_empty = bpy.data.objects.new(old_empty_name, None); context.collection.objects.link(viz_empty); viz_empty.matrix_world = Matrix.Identity(4)

        # Marker creation helper (using low-level BMesh for potential speed)
        def create_vector_marker(name, origin_world, direction_world, length, radius, material, parent_obj):
             if direction_world.length_squared < 1e-12: return None # Check length squared
             rot_quat = direction_world.normalized().to_track_quat('Z', 'Y')
             # Create mesh data
             mesh = bpy.data.meshes.new(name)
             bm = bmesh.new()
             bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=8, radius1=radius, radius2=radius, depth=length)
             # Translate origin to base of the cylinder/cone
             bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, length / 2.0))
             bm.to_mesh(mesh); bm.free()
             # Create object
             marker = bpy.data.objects.new(name, mesh)
             # Assign material
             if marker.data.materials: marker.data.materials[0] = material
             else: marker.data.materials.append(material)
             # Set transform (rotation first, then translation)
             marker.matrix_world = Matrix.Translation(origin_world) @ rot_quat.to_matrix().to_4x4()
             marker.parent = parent_obj
             context.collection.objects.link(marker) # Link to scene collection
             return marker

        # Calculate RMF frames using the bezier module
        rmf_frames = bezier.calculate_rmf_frames(spline, self.num_steps)
        if not rmf_frames:
            self.report({'ERROR'}, "Failed to calculate RMF frames."); return {'CANCELLED'}

        # Get world matrices and inverse-transpose for normal transform
        curve_world_matrix = curve_obj.matrix_world; curve_rot_scale_matrix = curve_world_matrix.to_3x3()
        try:
            inv_matrix = curve_rot_scale_matrix.inverted_safe(); inv_trans_matrix = inv_matrix.transposed(); transform_normals_correctly = True
        except ValueError:
            inv_trans_matrix = Matrix.Identity(3); transform_normals_correctly = False

        # Visualize each frame
        for i, frame_data in enumerate(rmf_frames):
            pos_local, tan_local, norm_rmf_local, frame_t = frame_data
            # Apply tilt using function from bezier module
            interpolated_tilt = bezier.get_interpolated_tilt(spline, frame_t)
            tilt_rotation_matrix = Matrix.Rotation(interpolated_tilt, 3, tan_local)
            norm_final_local = tilt_rotation_matrix @ norm_rmf_local; norm_final_local.normalize()
            bino_final_local = tan_local.cross(norm_final_local).normalized()

            # Transform to World Space
            pos_world = curve_world_matrix @ pos_local
            tan_world  = (curve_rot_scale_matrix @ tan_local).normalized()
            if transform_normals_correctly:
                norm_world = (inv_trans_matrix @ norm_final_local).normalized()
                bino_world = (inv_trans_matrix @ bino_final_local).normalized()
            else:
                norm_world = (curve_rot_scale_matrix @ norm_final_local).normalized()
                bino_world = (curve_rot_scale_matrix @ bino_final_local).normalized()

            # Re-orthogonalize world frame for clean visualization
            # This ensures the displayed markers are perpendicular, even if transforms caused skew
            bino_world = tan_world.cross(norm_world).normalized()
            # Avoid potential zero vector if T and N become parallel after transform
            if bino_world.length_squared > 1e-9:
                 norm_world = bino_world.cross(tan_world).normalized()
            # else: keep the potentially skewed norm_world

            # Optional: Print dot products for debugging orthogonality
            # dot_tn = tan_world.dot(norm_world); dot_tb = tan_world.dot(bino_world); dot_nb = norm_world.dot(bino_world)
            # if abs(dot_tn) > 1e-4 or abs(dot_tb) > 1e-4 or abs(dot_nb) > 1e-4: print(f"Step {i}: Non-orthogonality! T.N={dot_tn:.5f}, T.B={dot_tb:.5f}, N.B={dot_nb:.5f}")

            # Create markers using the helper function
            create_vector_marker(f"T_{i:03d}", pos_world, tan_world, self.scale, self.vector_radius, mat_tangent, viz_empty)
            create_vector_marker(f"N_{i:03d}", pos_world, norm_world, self.scale, self.vector_radius, mat_normal, viz_empty)
            create_vector_marker(f"B_{i:03d}", pos_world, bino_world, self.scale, self.vector_radius, mat_binormal, viz_empty)

        self.report({'INFO'}, f"Generated {self.num_steps} RMF frame visualizations."); return {'FINISHED'}


# --- Registration ---
# List of all classes that need to be registered with Blender
classes = (
    CCDG_Properties,
    OBJECT_OT_generate_cylinder_with_curve, # From op_generate
    OBJECT_OT_enable_realtime_bend,         # From op_deform
    OBJECT_OT_disable_realtime_bend,        # From op_deform
    VIEW3D_PT_piecegen_panel,               # Renamed UI Panel
    VISUALIZE_OT_curve_frames               # Kept in __init__.py
)
# Keep a reference to the handler function
_handler_ref = ccdg_depsgraph_handler

def register():
    """Registers all addon classes, properties, and the depsgraph handler."""
    print(f"Registering PieceGen addon..")
    if HAS_NUMPY: print(f"- Numpy {np.__version__ if hasattr(np,'__version__') else ''} detected.")
    else: print("- Numpy not detected (using fallback packing/serialization).")

    # Register all classes defined in the 'classes' tuple
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
        except ValueError:
            # May already be registered if reloading script using F8
            # print(f"Class '{cls.__name__}' already registered, skipping.")
            pass # Ignore error if already registered

    # Add PropertyGroup pointer to Scene type, making props accessible via scene.ccdg_props
    bpy.types.Scene.ccdg_props = bpy.props.PointerProperty(type=CCDG_Properties)

    # Clear monitored set and global cache on registration (clean start)
    cvars.MONITORED_MESH_OBJECTS.clear()
    cvars.original_coords_cache.clear()

    # Add the depsgraph handler if it's not already present
    if _handler_ref not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_handler_ref)
        print("- Realtime handler added.")
    print(f"Registered.")


def unregister():
    """Unregisters all addon classes, properties, and removes the depsgraph handler."""
    print(f"Unregistering PieceGen addon..")

    # Remove the depsgraph handler first
    if _handler_ref in bpy.app.handlers.depsgraph_update_post:
        try:
            bpy.app.handlers.depsgraph_update_post.remove(_handler_ref)
            print("- Realtime handler removed.")
        except ValueError:
            # Handler might have already been removed
            print(f"- Warning: Realtime handler was already removed or not found.")

    # Clear global state variables
    cvars.original_coords_cache.clear()
    cvars.MONITORED_MESH_OBJECTS.clear()

    # Delete the PropertyGroup pointer from Scene type safely
    if hasattr(bpy.types.Scene, 'ccdg_props'):
        try:
            # Check if the property actually exists on the instance before deleting type attr
            # This avoids errors if the scene context is unusual during unregister
            # if bpy.context.scene and bpy.context.scene.get('ccdg_props') is not None:
            #    del bpy.types.Scene.ccdg_props
            # Simpler approach: just try deleting the type attribute
            del bpy.types.Scene.ccdg_props
        except (AttributeError, Exception) as e:
            # Catch potential errors during unregistration
            print(f"Warning: Could not delete scene property 'ccdg_props': {e}")

    # Unregister all classes in reverse order
    for cls in reversed(classes):
         # Check if class is actually registered before trying to unregister
         if hasattr(bpy.types, cls.__name__):
            try:
                bpy.utils.unregister_class(cls)
            except (RuntimeError, Exception) as e:
                 # Ignore errors during script reload or Blender shutdown
                 print(f"Warning: Could not unregister class {cls.__name__}: {e}")
    print(f"Unregistered.")


# Standard execution guard: Prevents register/unregister running automatically
# when the script is imported as a module (which Blender does).
# Only runs if the script is executed directly (e.g., from Text Editor).
if __name__ == "__main__":
    # For testing from Text Editor, uncomment these lines:
    # print(f"\n--- Running Script Directly: {__file__} ---")
    # try:
    #     print("\n--- Attempting Unregister from __main__ ---")
    #     unregister()
    # except Exception as e:
    #     print(f"Error during automatic unregister on script run: {e}")
    # print("\n--- Attempting Register from __main__ ---")
    # register()
    pass # Keep pass here for normal addon operation

