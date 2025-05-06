# <pep8 compliant>
# -*- coding: utf-8 -*-

# Main Addon File: PieceGen/__init__.py (Toggle Workflow)

bl_info = {
    "name": "PieceGen RMF Curve Deform",
    "author": "AI Assistant (Gemini) & User",
    "version": (1, 25, 2), # Incremented for Readability refactor
    "blender": (3, 0, 0), # Minimum Blender version
    "location": "View3D > Sidebar (N Panel) > PieceGen Tab",
    "description": "Generates cylinder/curve pair, applies Python curve deform with RMF.",
    "warning": "PYTHON-BASED REALTIME DEFORM CAN BE SLOW FOR DENSE MESHES!",
    "doc_url": "", # TODO: Add documentation link
    "category": "Object", # Changed Category
}

# --- Standard Imports ---
import bpy
import mathutils
from mathutils import Vector, Matrix # Keep if needed by Visualize Op or Panel
import bmesh

# --- Local Addon Imports ---
# Import necessary components from other addon files
# Use try/except for robustness during development/reloading
from .common_imports import HAS_NUMPY, np # Import HAS_NUMPY flag and np
from . import common_vars as cvars # Import constants and global state
from .prop_serialization import unpack_verts_from_string # Import specific helper
from . import bezier # Import bezier/RMF functions (needed for Visualize Op)
from .op_generate import OBJECT_OT_generate_cylinder_with_curve # Import Operators
# Import the *new* toggle operator and the deform function
from .op_deform import OBJECT_OT_toggle_realtime_bend, deform_mesh_along_curve
# Keep visualize operator if it exists in its own file or here
# from .op_visualize import VISUALIZE_OT_curve_frames # Example if moved
from .gizmo import PIECEGEN_GGT_radius_control, PIECEGEN_GIZMO_radius_handle

# --- Scene Properties (Improved Readability) ---
class CCDG_Properties(bpy.types.PropertyGroup):
    """
    Stores settings for the Addon Panel, attached to the Scene object.
    These properties control object generation and deformation parameters.
    Accessible via `context.scene.ccdg_props`.
    """
    # --- Generation Properties ---
    num_cap_verts: bpy.props.IntProperty(
        name="Cap Vertices",
        description="Number of vertices in the cylinder cap ring (controls roundness)",
        default=32, min=3, max=256 # Sensible range for cylinder vertices
    )
    cap_radius: bpy.props.FloatProperty(
        name="Radius",
        description="Generated cylinder radius",
        default=1.0, min=0.01, # Avoid zero radius
        unit='LENGTH', # Use Blender's length units (e.g., meters)
        precision=3    # Display precision in the UI
    )
    height: bpy.props.FloatProperty(
        name="Height",
        description="Generated cylinder height (along local Z axis)",
        default=2.0, min=0.01, # Avoid zero height
        unit='LENGTH',
        precision=3
    )
    num_height_segs: bpy.props.IntProperty(
        name="Height Segments",
        description="Number of subdivisions along the cylinder height (more segments = smoother deformation)",
        default=10, min=1, max=500 # Allow more segments if needed
    )
    num_cpts_curve: bpy.props.IntProperty(
        name="Curve Points",
        description="Number of control points for the initial generated Bézier curve",
        default=4, min=2, max=64 # Range for initial curve complexity
    )
    cap_fill_type: bpy.props.EnumProperty(
        name="Cap Fill Type",
        items=[('NGON', "N-Gon", "Fill caps with single N-Gon face (default)"),
               ('TRIANGLE_FAN', "Tri Fan", "Fill caps with triangle fan"),
               ('NOTHING', "Nothing", "Leave caps open")],
        default='NGON',
        description="How to fill the top and bottom faces of the cylinder"
    )

    # --- Deformation Properties (Used by Panel for Active Object Interaction) ---
    target_curve: bpy.props.PointerProperty(
        name="Target Curve",
        type=bpy.types.Object,
        # Poll function ensures only valid Bezier curves appear in the picker
        poll=lambda self, obj: obj is not None and obj.type == 'CURVE' and obj.data and \
                               obj.data.splines and len(obj.data.splines) > 0 and \
                               obj.data.splines[0].type == 'BEZIER' and \
                               len(obj.data.splines[0].bezier_points) >= 2,
        description="Bezier curve object to deform along (must have >= 2 points)"
        # No update callback needed here; the operator reads this property on execution
    )
    rmf_steps_deform: bpy.props.IntProperty(
        name="RMF Steps",
        default=50,
        min=3, # Minimum required for stable RMF calculation
        max=1000, # Allow high resolution if needed
        description="Resolution for RMF calculation during deformation (higher = smoother but potentially slower)"
        # No update callback needed here; the handler reads this property during update
    )


# --- Depsgraph Update Handler (Improved Readability) ---
# This function runs automatically after Blender updates its dependency graph
# @bpy.app.handlers.persistent # Uncomment if handler needs to persist across Blender file loads
def ccdg_depsgraph_handler(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph):
    """
    Checks for updates to monitored curves and triggers deformation on linked meshes.
    This function is registered with Blender's application handlers and runs frequently.
    Efficiency is important here.
    """

    # Optimization: Exit early if no meshes are being monitored by this addon
    if not cvars.MONITORED_MESH_OBJECTS:
        return

    # --- Identify Updated Curves Relevant to Monitored Meshes ---
    # Build a quick lookup map: curve_name -> list of linked mesh_names
    # This avoids iterating all monitored meshes for every potential curve update.
    monitored_curve_map = {}
    # Need to iterate a copy, as the set might be modified if links are broken
    for mesh_name in list(cvars.MONITORED_MESH_OBJECTS):
        mesh_obj = scene.objects.get(mesh_name)
        # Ensure mesh exists and deformation is currently enabled for it
        if mesh_obj and mesh_obj.get(cvars.PROP_ENABLED):
            curve_name = mesh_obj.get(cvars.PROP_CURVE_NAME) # Get linked curve name from mesh prop
            if curve_name:
                # Add mesh to the list for this curve name
                if curve_name not in monitored_curve_map:
                    monitored_curve_map[curve_name] = []
                monitored_curve_map[curve_name].append(mesh_name)
        elif mesh_obj: # Mesh exists but deform is not enabled
             cvars.MONITORED_MESH_OBJECTS.discard(mesh_name) # Clean up monitoring set
             cvars.original_coords_cache.pop(mesh_name, None)
        # Implicit else: mesh_obj is None (deleted), discard handled below if needed

    # If no valid curves are being monitored (e.g., all links broken), clear the set and exit
    if not monitored_curve_map:
        cvars.MONITORED_MESH_OBJECTS.clear()
        return

    # Create a set to store the names of curve objects that were updated *and* are monitored
    relevant_updated_curve_names = set()

    # Check Blender's dependency graph updates for this cycle
    for update in depsgraph.updates:
        # Ensure the update has an 'original' ID to check against scene objects/data
        if not hasattr(update.id, "original"): continue

        obj_or_data = update.id.original # The original data-block that was updated

        # Check if the update is for a Curve *Object* that we are monitoring
        if isinstance(obj_or_data, bpy.types.Object) and obj_or_data.name in monitored_curve_map:
            # If geometry (points/handles) or transform (location/rotation/scale) changed...
            if update.is_updated_geometry or update.is_updated_transform:
                 # Mark this curve name as needing an update for linked meshes
                 relevant_updated_curve_names.add(obj_or_data.name)

        # Check if the update is for Curve *Data* used by a monitored Curve Object
        elif isinstance(obj_or_data, bpy.types.Curve) and update.is_updated_geometry:
             # Find all curve objects using this specific curve data block
             for curve_name_check in monitored_curve_map: # Iterate keys (monitored curve names)
                  curve_obj_check = scene.objects.get(curve_name_check)
                  # Check if the object exists and uses the updated curve data
                  if curve_obj_check and curve_obj_check.data == obj_or_data:
                       # Mark this curve object name as needing an update
                       relevant_updated_curve_names.add(curve_name_check)
                       # Optimization: No need to check other meshes for the same curve data update,
                       # just mark the curve object name once. We'll update all meshes linked to it later.
                       # Note: This assumes a curve object name won't appear multiple times
                       # as a key in monitored_curve_map if multiple meshes link to it.


    # Exit if no relevant curves (curves linked to monitored meshes) were updated
    if not relevant_updated_curve_names:
        return

    # --- Deform Meshes Linked to Updated Curves ---
    # print(f"Handler triggered for curves: {relevant_updated_curve_names}") # Optional Debug print

    # Iterate through the names of curves that were updated and are linked
    for curve_name in relevant_updated_curve_names:
        curve_obj = scene.objects.get(curve_name)
        # Double check the curve object still exists
        if not curve_obj: continue

        # Deform all meshes that are linked to this specific updated curve
        # Check if the curve name is still in our map (it should be)
        if curve_name in monitored_curve_map:
            # Iterate through the list of mesh names linked to this curve
            for mesh_name in monitored_curve_map[curve_name]:
                mesh_obj = scene.objects.get(mesh_name)

                # Final validation before deforming this specific mesh
                # Check if mesh object still exists and is enabled
                if not mesh_obj or mesh_obj.type != 'MESH' or not mesh_obj.get(cvars.PROP_ENABLED):
                    # If link is broken or disabled, remove from monitoring to avoid future checks
                    cvars.MONITORED_MESH_OBJECTS.discard(mesh_name)
                    cvars.original_coords_cache.pop(mesh_name, None) # Clear cache too
                    continue

                # Retrieve data needed for deformation from the mesh object's custom properties
                packed_verts = mesh_obj.get(cvars.PROP_ORIG_VERTS)
                cyl_height = mesh_obj.get(cvars.PROP_HEIGHT)
                # Get RMF steps from the global scene properties (panel setting)
                # Use getattr for safety in case props aren't fully initialized
                rmf_steps = getattr(scene.ccdg_props, 'rmf_steps_deform', 50) # Default if prop missing

                # Check if essential properties exist
                if packed_verts and cyl_height is not None:
                    # Get original coords from cache or unpack if needed (e.g., after file load)
                    original_verts = cvars.original_coords_cache.get(mesh_name)
                    if not original_verts:
                        # Unpack from property if not found in cache
                        original_verts = unpack_verts_from_string(packed_verts) # From prop_serialization
                        if original_verts:
                            # Re-populate cache for next time this mesh needs update
                            cvars.original_coords_cache[mesh_name] = original_verts
                        else:
                            # Failed to get original verts, cannot deform
                            print(f"Error Handler: Could not unpack original verts for {mesh_name}")
                            continue # Skip deformation for this mesh

                    # Check vertex count consistency before deforming (important!)
                    if len(original_verts) == len(mesh_obj.data.vertices):
                        # Call the deform function (imported from op_deform)
                        # print(f"  Deforming {mesh_name} using {curve_name}") # Optional Debug print
                        deform_mesh_along_curve(mesh_obj, curve_obj, original_verts, cyl_height, rmf_steps) # Pass rmf_steps
                    else:
                        # Vertex count changed - deformation is no longer valid
                        print(f"Error Handler: Vertex count mismatch for {mesh_name}. Disabling deform.")
                        # Optionally auto-disable here if vertex count changes
                        # target_obj.pop(cvars.PROP_ENABLED, None) ... etc ...

                else:
                    # Missing necessary properties on the mesh object
                    print(f"Error Handler: Missing properties for {mesh_name} ({cvars.PROP_ORIG_VERTS=}, {cvars.PROP_HEIGHT=})")


# --- UI Panel ---
# Defines the panel in the 3D Viewport Sidebar (N-Panel)
class VIEW3D_PT_piecegen_panel(bpy.types.Panel):
    """UI Panel for the PieceGen RMF Curve Deform addon"""
    # Panel identifiers and location
    bl_label = "PieceGen Deform"; bl_idname = "VIEW3D_PT_piecegen_deform" # Renamed ID
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'PieceGen' # Custom Category

    def draw(self, context):
        """Draws the UI elements in the panel."""
        layout = self.layout
        scene = context.scene
        active_obj = context.active_object

        # Ensure the addon's property group exists on the scene
        if not hasattr(scene, 'ccdg_props'):
            layout.label(text="Error: Addon properties not found!", icon='ERROR')
            return
        # Get the property group instance from the scene
        props = scene.ccdg_props

        # --- Section 1: Object Generation ---
        # (Code for generation section remains the same)
        box_gen = layout.box(); col_gen = box_gen.column(align=True)
        col_gen.label(text="1. Object Generation:")
        col_gen.prop(props, "num_cap_verts"); col_gen.prop(props, "cap_radius"); col_gen.prop(props, "height")
        col_gen.prop(props, "num_height_segs"); col_gen.prop(props, "cap_fill_type"); col_gen.prop(props, "num_cpts_curve")
        col_gen.operator(OBJECT_OT_generate_cylinder_with_curve.bl_idname, icon='MESH_CYLINDER')

        layout.separator()

        # --- Section 2: Realtime Python Deformer ---
        box_rt = layout.box()
        col_rt = box_rt.column(align=True)
        col_rt.label(text="2. Realtime Deformer:")

        # Show controls only if a mesh object is active
        if active_obj and active_obj.type == 'MESH':
            col_rt.label(text="Target Mesh: " + active_obj.name)
            # Curve selection using PointerProperty (linked to scene props)
            col_rt.prop(props, "target_curve")
            # RMF Steps setting (linked to scene props)
            col_rt.prop(props, "rmf_steps_deform")

            # Check if deformation is currently enabled for this active object
            is_enabled = active_obj.get(cvars.PROP_ENABLED, False) # Use imported constant
            linked_curve_name = active_obj.get(cvars.PROP_CURVE_NAME, "") # Use imported constant

            # --- Toggle Button ---
            # Determine button text/icon based on 'is_enabled'
            op_row = col_rt.row() # Use a row for the operator button
            # Set the 'active' state of the ROW before drawing the operator
            # Disable the "Start" button if no target curve is selected in the panel property
            # The button is always active if deformation is already enabled (to allow stopping)
            op_row.active = is_enabled or (props.target_curve is not None)
            # Draw the operator button
            op = op_row.operator(
                OBJECT_OT_toggle_realtime_bend.bl_idname, # Use bl_idname from imported class
                text="Stop Realtime Deform" if is_enabled else "Start Realtime Deform",
                icon='PAUSE' if is_enabled else 'PLAY'
            )

            if is_enabled:
                if active_obj.name in cvars.MONITORED_MESH_OBJECTS: col_rt.label(text=f"Deforming with '{linked_curve_name}'", icon='CHECKMARK'); col_rt.label(text="Edit curve now.", icon='INFO')
                else: col_rt.label(text="State Error: Try Toggling Off/On?", icon='ERROR')
                col_rt.label(text="WARNING: Can be slow!", icon='ERROR')
            else:
                if props.target_curve is None: col_rt.label(text="Select Target Curve above to enable.", icon='INFO')
                else: col_rt.label(text="Ready to start deformation.", icon='INFO')
        else: col_rt.label(text="Select Mesh Object to deform.", icon='INFO')


# --- Visualization Operator (Improved Readability) ---
class VISUALIZE_OT_curve_frames(bpy.types.Operator):
    """Visualizes the Tangent, Normal, and Binormal RMF frames along the active curve"""
    bl_idname = "object.visualize_curve_frames"
    bl_label = "Visualize RMF Frames"
    bl_options = {'REGISTER', 'UNDO'} # Allow undo after running

    # Operator properties (appear in redo panel)
    num_steps: bpy.props.IntProperty(
        name="Steps",
        description="Number of evaluation points (frames) along the curve",
        default=20, min=2, max=500 # Increased max steps
    )
    scale: bpy.props.FloatProperty(
        name="Vector Scale",
        description="Length of the visualized vector markers",
        default=0.2, min=0.01, max=10.0, subtype='DISTANCE', unit='LENGTH'
    )
    vector_radius: bpy.props.FloatProperty(
        name="Vector Radius",
        description="Radius of the cylinder markers used for vectors",
        default=0.015, min=0.001, max=1.0, subtype='DISTANCE', unit='LENGTH'
    )

    @classmethod
    def poll(cls, context):
        """Check if the operator can run in the current context."""
        active_obj = context.active_object
        # Require an active object that is a Curve...
        return active_obj and active_obj.type == 'CURVE' and \
               active_obj.data and active_obj.data.splines and \
               len(active_obj.data.splines) > 0 and \
               active_obj.data.splines[0].type == 'BEZIER' and \
               len(active_obj.data.splines[0].bezier_points) >= 2 # ...with at least 2 points

    def create_gradient_material(self, name, base_color):
        """
        Creates or retrieves a node-based material with a gradient.
        The gradient runs along the local Z-axis of the object,
        from the specified base_color to white.

        Args:
            name (str): The desired name for the material.
            base_color (tuple): An RGB or RGBA tuple for the gradient start color.

        Returns:
            bpy.types.Material: The created or updated material.
        """
        # Get existing material or create new one
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)
        else:
            # Clear existing nodes if reusing material to ensure clean setup
            if mat.node_tree:
                mat.node_tree.nodes.clear()

        mat.use_nodes = True # Ensure node tree is used
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear() # Clear any default nodes (like Principled BSDF)

        # Create necessary shader nodes
        tex_coord_node = nodes.new(type='ShaderNodeTexCoord')
        separate_xyz_node = nodes.new(type='ShaderNodeSeparateXYZ')
        map_range_node = nodes.new(type='ShaderNodeMapRange') # For robust 0-1 mapping
        color_ramp_node = nodes.new(type='ShaderNodeValToRGB')
        principled_bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_node = nodes.new(type='ShaderNodeOutputMaterial')

        # Configure Map Range node:
        # Assume Generated Z coord runs 0 to 1 (common for primitives), map it explicitly
        map_range_node.inputs['From Min'].default_value = 0.0
        map_range_node.inputs['From Max'].default_value = 1.0
        map_range_node.inputs['To Min'].default_value = 0.0
        map_range_node.inputs['To Max'].default_value = 1.0
        map_range_node.use_clamp = True # Ensure output is strictly within 0-1

        # Configure Color Ramp for gradient: Base Color -> Base Color -> White
        color_ramp = color_ramp_node.color_ramp
        # Start color stop (at position 0.0)
        color_ramp.elements[0].position = 0.0
        color_ramp.elements[0].color = (*base_color[:3], 1.0) # Use input base color (ensure alpha is 1)
        # End color stop (at position 1.0) - Adjusted to 0.8 for more base color visibility
        color_ramp.elements[1].position = 0.8 # White starts fading in later
        color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0) # White color
        # Add an intermediate stop to hold the base color longer
        # Check if element exists before creating (important for script reload)
        if len(color_ramp.elements) < 3:
            new_stop = color_ramp.elements.new(position=0.7) # Add stop at 70%
        else:
            new_stop = color_ramp.elements[2]
            new_stop.position = 0.7 # Ensure position is correct on reload
        new_stop.color = (*base_color[:3], 1.0) # Make intermediate stop the base color

        # Position nodes for better readability in Shader Editor (optional)
        tex_coord_node.location = (-800, 0)
        separate_xyz_node.location = (-600, 0)
        map_range_node.location = (-400, 0)
        color_ramp_node.location = (-200, 0)
        principled_bsdf_node.location = (0, 0)
        output_node.location = (200, 0)

        # Link nodes together:
        # Generated Coords -> Separate XYZ -> [Z] -> MapRange -> ColorRamp Fac -> BSDF Base Color -> Output
        links.new(tex_coord_node.outputs['Generated'], separate_xyz_node.inputs['Vector'])
        links.new(separate_xyz_node.outputs['Z'], map_range_node.inputs['Value']) # Use Z axis for gradient along cylinder length
        links.new(map_range_node.outputs['Result'], color_ramp_node.inputs['Fac']) # Mapped 0-1 value drives the Color Ramp
        links.new(color_ramp_node.outputs['Color'], principled_bsdf_node.inputs['Base Color']) # Ramp output sets the shader color
        links.new(principled_bsdf_node.outputs['BSDF'], output_node.inputs['Surface']) # Connect shader to output

        return mat

    def create_vector_marker(self, op_context, name, origin_world, direction_world, length, radius, material, parent_obj):
        """
        Creates a cylinder mesh object to represent a vector.

        Args:
            op_context: The operator context.
            name (str): Name for the new marker object.
            origin_world (Vector): World-space start position for the marker.
            direction_world (Vector): World-space direction vector (will be normalized).
            length (float): Length of the cylinder marker.
            radius (float): Radius of the cylinder marker.
            material (bpy.types.Material): Material to assign to the marker.
            parent_obj (bpy.types.Object): Object to parent the marker to.

        Returns:
            bpy.types.Object or None: The created marker object, or None if direction is zero.
        """
        # Avoid creating marker for zero-length direction vectors
        if direction_world.length_squared < 1e-12: # Use squared length for efficiency
             print(f"Skipping marker '{name}': Zero direction vector.")
             return None

        # Normalize direction for rotation calculation
        dir_norm = direction_world.normalized()
        # Calculate rotation quaternion to align local Z-axis with the direction vector
        # 'Y' is used as the tracking axis to lock rotation around the vector
        rot_quat = dir_norm.to_track_quat('Z', 'Y')

        # Create mesh data using BMesh (potentially faster than bpy.ops for many objects)
        mesh = bpy.data.meshes.new(name + "_mesh") # Give mesh data a unique name
        bm = bmesh.new()
        # Create a simple cylinder (or cone for arrow-like appearance)
        bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=8,
                              radius1=radius, radius2=radius, # Use same radius for cylinder
                              depth=length)
        # Translate geometry so its base is at (0,0,0) in local space
        # (create_cone origin is center, so move up by half depth)
        bmesh.ops.translate(bm, verts=bm.verts, vec=(0, 0, length / 2.0))
        bm.to_mesh(mesh) # Write BMesh data to mesh
        bm.free() # Free BMesh instance

        # Create object from mesh data
        marker = bpy.data.objects.new(name, mesh)

        # Assign material
        if marker.data.materials: marker.data.materials[0] = material
        else: marker.data.materials.append(material)

        # Set object transform: Rotate first, then translate to origin
        marker.matrix_world = Matrix.Translation(origin_world) @ rot_quat.to_matrix().to_4x4()
        # Set parent
        marker.parent = parent_obj
        # Link object to the scene collection provided by the operator context
        op_context.collection.objects.link(marker)
        return marker

    def execute(self, context):
        """Calculates and visualizes RMF frames for the active curve."""
        curve_obj = context.active_object
        curve_data = curve_obj.data
        # Validation already done by poll, but double-check spline
        if not curve_data.splines or len(curve_data.splines) == 0:
             self.report({'ERROR'}, "Curve has no splines.")
             return {'CANCELLED'}
        spline = curve_data.splines[0] # Assume first spline is the target

        # --- Setup ---
        # Create materials with gradients
        mat_tangent  = self.create_gradient_material("Vis_Tangent_Grad", (1.0, 0.0, 0.0)) # Red base
        mat_normal   = self.create_gradient_material("Vis_Normal_Grad", (0.0, 1.0, 0.0))   # Green base
        mat_binormal = self.create_gradient_material("Vis_Binormal_Grad", (0.0, 0.0, 1.0)) # Blue base

        # Create or clear parent empty for visualization objects
        old_empty_name = f"{curve_obj.name}_FramesViz"
        old_empty = bpy.data.objects.get(old_empty_name)
        if old_empty:
            # Store current selection state
            current_active = context.view_layer.objects.active
            current_selected = context.selected_objects[:]
            # Delete existing hierarchy robustly
            objs_to_delete = list(old_empty.children) + [old_empty]
            bpy.ops.object.select_all(action='DESELECT')
            for obj in objs_to_delete:
                # Check object still exists (might have been deleted manually)
                if obj.name in bpy.data.objects:
                     obj.select_set(True)
            # Delete selected objects (parent and children)
            bpy.ops.object.delete(use_global=False) # Use use_global=False for safety
            # Restore original selection
            bpy.ops.object.select_all(action='DESELECT')
            for obj in current_selected:
                 if obj.name in bpy.data.objects: # Check if original selection still exists
                     obj.select_set(True)
            # Restore active object if it still exists
            if current_active and current_active.name in bpy.data.objects:
                 context.view_layer.objects.active = current_active

        # Create new parent empty at world origin
        viz_empty = bpy.data.objects.new(old_empty_name, None)
        context.collection.objects.link(viz_empty)
        viz_empty.matrix_world = Matrix.Identity(4)

        # --- Calculate RMF Frames ---
        # Call the function from the imported bezier module
        rmf_frames = bezier.calculate_rmf_frames(spline, self.num_steps)
        if not rmf_frames:
            self.report({'ERROR'}, "Failed to calculate RMF frames.")
            return {'CANCELLED'}

        # --- Get World Matrices and Inverse-Transpose ---
        curve_world_matrix = curve_obj.matrix_world
        curve_rot_scale_matrix = curve_world_matrix.to_3x3()
        try:
            inv_matrix = curve_rot_scale_matrix.inverted_safe()
            inv_trans_matrix = inv_matrix.transposed()
            transform_normals_correctly = True
        except ValueError:
            inv_trans_matrix = Matrix.Identity(3)
            transform_normals_correctly = False
            print("Visualize Warning: Curve matrix not invertible, normal transform may be inaccurate.")

        # --- Visualize Each Calculated Frame ---
        for i, frame_data in enumerate(rmf_frames):
            # Unpack frame data (local to the curve object)
            pos_local, tan_local, norm_rmf_local, frame_t = frame_data

            # --- Apply Tilt for Visualization ---
            # Get interpolated tilt value at this frame's specific parameter t
            interpolated_tilt = bezier.get_interpolated_tilt(spline, frame_t) # Use bezier module
            # Create rotation matrix around the local tangent
            tilt_rotation_matrix = Matrix.Rotation(interpolated_tilt, 3, tan_local)
            # Rotate the calculated RMF normal by the tilt
            norm_final_local = tilt_rotation_matrix @ norm_rmf_local
            norm_final_local.normalize() # Ensure unit length
            # Calculate the corresponding binormal: B = T x N
            bino_final_local = tan_local.cross(norm_final_local).normalized()

            # --- Transform Frame Vectors to World Space ---
            pos_world = curve_world_matrix @ pos_local # Transform position
            tan_world  = (curve_rot_scale_matrix @ tan_local).normalized() # Transform tangent

            # Transform final (tilted) normal and binormal using inverse-transpose
            if transform_normals_correctly:
                norm_world = (inv_trans_matrix @ norm_final_local).normalized()
                bino_world = (inv_trans_matrix @ bino_final_local).normalized()
            else: # Fallback if curve matrix was non-invertible
                norm_world = (curve_rot_scale_matrix @ norm_final_local).normalized()
                bino_world = (curve_rot_scale_matrix @ bino_final_local).normalized()

            # --- Re-orthogonalize World Frame for Clean Visualization ---
            # Ensures displayed markers are perfectly perpendicular, even if transforms introduced slight skew.
            # Recalculate Binormal based on world T and potentially skewed world N
            bino_world = tan_world.cross(norm_world).normalized()
            # Avoid creating zero vector if T and N became parallel after transform
            if bino_world.length_squared > 1e-9:
                 # Recalculate Normal based on world B and T for perfect orthogonality
                 norm_world = bino_world.cross(tan_world).normalized()
            # else: keep the potentially skewed norm_world if B is zero

            # Optional: Print dot products for debugging orthogonality in world space
            # dot_tn = tan_world.dot(norm_world); dot_tb = tan_world.dot(bino_world); dot_nb = norm_world.dot(bino_world)
            # if abs(dot_tn) > 1e-4 or abs(dot_tb) > 1e-4 or abs(dot_nb) > 1e-4: print(f"Step {i}: Non-orthogonality! T.N={dot_tn:.5f}, T.B={dot_tb:.5f}, N.B={dot_nb:.5f}")

            # --- Create Visual Markers ---
            # Use the helper function to create cylinder objects representing T, N, B
            self.create_vector_marker(context, f"T_{i:03d}", pos_world, tan_world, self.scale, self.vector_radius, mat_tangent, viz_empty)
            self.create_vector_marker(context, f"N_{i:03d}", pos_world, norm_world, self.scale, self.vector_radius, mat_normal, viz_empty)
            self.create_vector_marker(context, f"B_{i:03d}", pos_world, bino_world, self.scale, self.vector_radius, mat_binormal, viz_empty)

        self.report({'INFO'}, f"Generated {self.num_steps} RMF frame visualizations.")
        return {'FINISHED'}


class TEST_OT_test_op(bpy.types.Operator):
    bl_idname = "test.test_op"
    bl_label = "Test"
    bl_description = "A test operator"
    bl_options = {'REGISTER'}

    mouse_x : bpy.props.IntProperty()
    mouse_y : bpy.props.IntProperty()

    @classmethod
    def poll(cls, context):
        return True

    def invoke(self, context, event):
        self.mouse_x = event.mouse_x
        self.mouse_y = event.mouse_y
        return self.execute(context)

    def execute(self, context):
        print(f"Test operator: {self.mouse_x}, {self.mouse_y}")
        return {'FINISHED'}


class TEST_GT_test_gizmo(bpy.types.Gizmo):
    bl_target_properties = (
        {"id": "mouse_x", "type": 'INT'},
        {"id": "mouse_y", "type": 'INT'},
    )

    def draw(self, context):
        matrix = mathutils.Matrix.Translation((0.5, 0.5, 0.5))
        self.draw_preset_box(matrix, select_id=0)

    def draw_select(self, context, select_id=0):
        matrix = mathutils.Matrix.Translation((0.5, 0.5, 0.5))
        self.draw_preset_box(matrix, select_id=select_id)

    def setup(self):
        print(f"{self}: setup")

    def modal(self, context, event, tweak):
        self.target_set_value("mouse_x", event.mouse_x)
        self.target_set_value("mouse_y", event.mouse_y)
        return {'RUNNING_MODAL'}

    def invoke(self, context, event):
        print(f"{self}: invoke(event={event})")
        return {'RUNNING_MODAL'}

    def exit(self, context, cancel):
        print(f"{self}: exit(cancel={cancel})")


class TEST_GGT_test_group(bpy.types.GizmoGroup):
    bl_label = "Test Widget"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D'} # 'TOOL_INIT' also sounds appropriate, but then the gizmo doesn't appear!
    bl_operator = "test.test_op" # Just for the tooltip

    @classmethod
    def poll(cls, context):
        return context.workspace.tools.from_space_view3d_mode(context.mode).idname == PIECEGEN_TOOL_test.bl_idname
        # o = context.object
        # return (o and o.select_get() and o.type == 'MESH')

    def setup(self, context):
        print(f"{self}: setup")
        o = context.object
        giz = self.gizmos.new("TEST_GT_test_gizmo")
        giz.use_draw_modal = True # Is this necessary? (Makes no difference I can see)
        giz.target_set_operator("test.test_op", index=0)
        giz.matrix_basis = o.matrix_world.normalized()
        giz.color = 1.0, 1.0, 1.0
        giz.alpha = 0.5
        giz.color_highlight = 1.0, 0.0, 0.0
        giz.alpha_highlight = 1.0
        self.gizmo = giz

    def refresh(self, context):
        print(f"{self}: refresh")
        o = context.object
        giz = self.gizmo
        giz.matrix_basis = o.matrix_world.normalized()

# --- Custom Workspace Tool (Minimal Definition) ---
# https://github.com/blender/blender/blob/main/scripts/templates_py/ui_tool_simple.py
# https://b3d.interplanety.org/en/creating-custom-tool-in-blender/
class PIECEGEN_TOOL_test(bpy.types.WorkSpaceTool):
    """Minimal tool definition for testing visibility."""
    # 1. Define where the tool appears
    bl_space_type = 'VIEW_3D'
    # https://docs.blender.org/api/current/bpy_types_enum_items/context_mode_items.html
    bl_context_mode = 'EDIT_CURVE'

    # 2. Unique identifier
    bl_idname = "piecegen.radius_edit_tool"

    # 3. User-visible name and icon
    bl_label = "PieceGen Edit Tool" # Simple label
    bl_description = (
        'Curve Deformer\n'
    )
    bl_icon = "brush.generic" # Use a guaranteed icon

    bl_widget = "TEST_GGT_test_group"
    # gizmo_group_properties = [
    #     ("radius", 75.0),
    #     ("backdrop_fill_alpha", 0.0),
    # ]
    # bl_keymap = (
    #     # ('view3d.select_box',   {'type': 'LEFTMOUSE', 'value': 'CLICK_DRAG'},   None),
    #     ("transform.translate", {"type": 'LEFTMOUSE', "value": 'PRESS'},        None),
    # )

    # 8. Optional: Draw settings in the header when active
    def draw_settings(context, layout, tool):
        layout.label(text="PieceGen Tool Active")


# --- Registration ---
# List of all classes from all modules that need to be registered with Blender
classes = (
    CCDG_Properties,                        # Defined in this file
    OBJECT_OT_generate_cylinder_with_curve, # From op_generate
    OBJECT_OT_toggle_realtime_bend,         # From op_deform (replaces enable/disable)
    VIEW3D_PT_piecegen_panel,               # Defined in this file (renamed)
    VISUALIZE_OT_curve_frames,              # Defined in this file
)
# Keep a reference to the handler function for registration/unregistration
_handler_ref = ccdg_depsgraph_handler

def register():
    """Registers all addon classes, properties, and the depsgraph handler."""
    
    print(f"Registering PieceGen addon (Toggle Workflow)..")
    
    if HAS_NUMPY: print(f"- Numpy {np.__version__ if hasattr(np,'__version__') else ''} detected.")
    else: print("- Numpy not detected.")
    
    for cls in classes:
        try: bpy.utils.register_class(cls)
        except ValueError: pass
    gizmo.register()
    
    # bpy.utils.register_tool(PIECEGEN_TOOL_test, after={'builtin.transform'}, separator=True, group=True)
    setattr(bpy.types.Scene, 'ccdg_props', bpy.props.PointerProperty(type=CCDG_Properties))
    
    cvars.MONITORED_MESH_OBJECTS.clear()
    cvars.original_coords_cache.clear()
    
    if _handler_ref not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_handler_ref)
        print("- Realtime handler added.")
    print(f"Registered.")


def unregister():
    """Unregisters all addon classes, properties, and removes the depsgraph handler."""
    # (Unregistration logic unchanged from previous version)
    print(f"Unregistering PieceGen addon..")
    
    if _handler_ref in bpy.app.handlers.depsgraph_update_post:
        try: bpy.app.handlers.depsgraph_update_post.remove(_handler_ref); print("- Realtime handler removed.")
        except ValueError: print(f"- Warning: Realtime handler was already removed or not found.")
    
    cvars.original_coords_cache.clear()
    cvars.MONITORED_MESH_OBJECTS.clear()
    
    if hasattr(bpy.types.Scene, 'ccdg_props'):
        try: del bpy.types.Scene.ccdg_props
        except (AttributeError, Exception) as e: print(f"Warning: Could not delete scene property 'ccdg_props': {e}")

    # bpy.utils.unregister_tool(PIECEGEN_TOOL_test)

    gizmo.unregister()
    for cls in reversed(classes):
         if hasattr(bpy.types, cls.__name__):
            try: bpy.utils.unregister_class(cls)
            except (RuntimeError, Exception) as e: print(f"Warning: Could not unregister class {cls.__name__}: {e}")
    print(f"Unregistered.")


# Standard execution guard
if __name__ == "__main__":
    # Avoid auto-registering from __main__ in multi-file setup
    pass
