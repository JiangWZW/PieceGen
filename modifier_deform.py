# <pep8 compliant>
# -*- coding: utf-8 -*-
# File: modifier.py
# Contains the Custom Modifier logic for RMF Curve Deform

import bpy
from mathutils import Vector, Matrix
# Import RMF functions from bezier.py using relative import
from . import bezier # Assuming bezier.py is in the same folder

# --- Modifier Update Function ---
# Defined outside class for property update callback
def update_modifier_dependency(self, context):
    """Callback to tag object data for depsgraph update when modifier props change."""
    if context and context.object:
        # Tagging the object's data seems sufficient to trigger re-evaluation
        context.object.data.update()
        # Trigger viewport update if needed (might be redundant)
        # if context.view_layer:
        #     context.view_layer.update()

# --- Custom Modifier Definition ---
class CurveDeformModifier(bpy.types.Modifier):
    """Deforms mesh along a curve using RMF and Tilt"""
    # Required properties
    bl_idname = "PIECEGEN_MT_curve_deform" # Unique ID name
    bl_label = "Curve Deform"
    bl_description = "Deform mesh along a Bezier curve"
    bl_options = {'REGISTER', 'UNDO'} # Enable undo for modifier changes
    bl_type = 'DEFORM' # This modifier changes geometry

    # --- Modifier Properties ---
    # These properties will appear in the Modifier UI panel
    curve_object: bpy.props.PointerProperty(
        name="Curve Object",
        type=bpy.types.Object,
        # Poll function to ensure only valid Bezier curve objects can be selected
        poll=lambda self, obj: obj is not None and obj.type == 'CURVE' and obj.data and \
                               obj.data.splines and len(obj.data.splines) > 0 and \
                               obj.data.splines[0].type == 'BEZIER' and \
                               len(obj.data.splines[0].bezier_points) >= 2,
        description="Bezier curve object to deform along (must have >= 2 points)",
        update=update_modifier_dependency # Trigger update when changed
    )

    rmf_steps: bpy.props.IntProperty(
        name="RMF Steps",
        default=50,
        min=3, # Min 3 for stable RMF calculation seems reasonable
        max=1000, # Increased max for potentially complex curves
        description="Resolution/Steps for RMF calculation (higher = smoother but slower)",
        update=update_modifier_dependency
    )

    base_height: bpy.props.FloatProperty(
        name="Base Height",
        default=1.0, # Default, should be set on creation or manually
        min=0.001,
        subtype='DISTANCE', # Use distance subtype for units
        unit='LENGTH',
        precision=4,
        description="Original height of the mesh (along local Z) used for mapping to curve parameter 't'. Set automatically by 'Generate Objects' or adjust manually.",
        update=update_modifier_dependency
    )

    # --- Placeholder for future interpolation ---
    # use_interpolation: bpy.props.BoolProperty(
    #     name="Use Frame Interpolation (WIP)",
    #     default=False,
    #     description="Interpolate between RMF frames instead of using nearest (Requires Slerp - WIP)",
    #     update=update_modifier_dependency
    # )

    # --- Modifier UI Layout ---
    def draw(self, context, layout):
        """Defines how the modifier properties are drawn in the UI."""
        layout.use_property_split = True # Align labels and properties
        layout.use_property_decorate = False # No animation keying icons

        layout.prop(self, "curve_object")
        layout.prop(self, "rmf_steps")
        layout.prop(self, "base_height")
        # layout.prop(self, "use_interpolation") # Uncomment when interpolation is implemented

    # --- Modifier Evaluation Logic ---
    # This is the core function called by Blender's dependency graph
    def modifier_update(self, context, depsgraph):
        """
        Called by Blender's dependency graph when the modifier needs to evaluate.
        Modifies the evaluated geometry based on the modifier settings.
        """
        # Get the object this modifier is applied to (as evaluated by depsgraph)
        obj_eval = depsgraph.objects.get(context.object.name)
        if not obj_eval or not obj_eval.data or not isinstance(obj_eval.data, bpy.types.Mesh):
            print(f"{self.bl_label} Error: Could not get evaluated mesh data for {context.object.name}")
            return

        # Get the original object (needed for original vertex coords)
        obj_orig = context.object

        # Get the evaluated curve object from the dependency graph
        curve_eval = None
        if self.curve_object:
            try:
                # Use the evaluated version of the curve object
                curve_eval = depsgraph.get_evaluated_object(self.curve_object)
            except Exception as e:
                # Handle cases where object might not be in depsgraph (rare)
                print(f"{self.bl_label} Warning: Could not get evaluated curve object '{self.curve_object.name}': {e}. Using original.")
                curve_eval = self.curve_object # Fallback to original

        # --- Input Validation ---
        if not curve_eval or not curve_eval.data or not isinstance(curve_eval.data, bpy.types.Curve) or \
           not curve_eval.data.splines or len(curve_eval.data.splines) == 0 or \
           curve_eval.data.splines[0].type != 'BEZIER' or \
           len(curve_eval.data.splines[0].bezier_points) < 2:
            # Silently return if curve invalid - UI poll should prevent selection mostly.
            # This avoids console spam during edits before a valid curve is selected.
            return

        if self.base_height <= 1e-6:
            # Avoid division by zero or nonsensical mapping
            print(f"{self.bl_label} Warning: Base Height is zero or negative. Deformation skipped.")
            return

        spline = curve_eval.data.splines[0] # Use the first spline of the evaluated curve data

        # --- Get Mesh Data ---
        mesh_eval = obj_eval.data # The mesh data *after* previous modifiers
        verts_eval = mesh_eval.vertices

        # IMPORTANT: Get original coordinates for mapping 't' and XY offset.
        # Accessing obj_orig.data assumes the base mesh topology hasn't changed
        # *before* this modifier in the stack. A more robust approach might involve
        # storing original data or using UVs for mapping if topology changes.
        try:
            mesh_orig = obj_orig.data
            vert_count = len(mesh_orig.vertices)
            # Check if evaluated mesh has the same number of vertices
            if len(verts_eval) != vert_count:
                 print(f"{self.bl_label} Warning: Vertex count mismatch between original ({vert_count}) and evaluated ({len(verts_eval)}). Deformation skipped. Ensure modifier is applied to base mesh or handle topology changes.")
                 return
            # Create a list of original local coordinates
            verts_orig_co = [v.co.copy() for v in mesh_orig.vertices]
        except (AttributeError, ReferenceError) as e:
             print(f"{self.bl_label} Error: Could not access original mesh data for {obj_orig.name}: {e}")
             return

        if vert_count == 0: return # Nothing to deform

        # --- Pre-calculate RMF Frames (using evaluated curve spline) ---
        rmf_frames = bezier.calculate_rmf_frames(spline, self.rmf_steps)
        if not rmf_frames:
            print(f"{self.bl_label} Error: RMF frame calculation failed.")
            return
        num_rmf_frames = len(rmf_frames)

        # --- Get Transformation Matrices ---
        curve_world_matrix = curve_eval.matrix_world # Use evaluated curve matrix
        # Modifier output needs to be in the object's local space
        target_inv_matrix  = obj_eval.matrix_world.inverted()

        # --- Calculate Inverse-Transpose for Normal Transforms ---
        curve_rot_scale_matrix = curve_world_matrix.to_3x3()
        try:
            inv_matrix = curve_rot_scale_matrix.inverted_safe()
            inv_trans_matrix = inv_matrix.transposed()
            transform_normals_correctly = True
        except ValueError:
            inv_trans_matrix = Matrix.Identity(3) # Fallback
            transform_normals_correctly = False
            # Only print warning once per update? Could get spammy.
            # print(f"{self.bl_label} Warning: Curve matrix not invertible, normal transform may be incorrect.")


        # --- Prepare Coordinate Array (for evaluated mesh) ---
        # We will write the new coordinates directly into the evaluated mesh data
        new_coords_flat = [0.0] * (vert_count * 3)

        # --- Process Each Vertex ---
        for i in range(vert_count):
            original_co = verts_orig_co[i] # Use original coord for mapping

            # Calculate curve parameter 't' (0-1) based on original local Z and base_height property
            vertex_t = original_co.z / self.base_height
            vertex_t = max(0.0, min(1.0, vertex_t)) # Clamp t

            # --- Find NEAREST pre-calculated RMF frame (Simplification) ---
            # TODO: Implement Slerp interpolation based on self.use_interpolation
            nearest_idx = round(vertex_t * (num_rmf_frames - 1))
            nearest_idx = max(0, min(num_rmf_frames - 1, nearest_idx)) # Clamp index
            # Get frame data (local to the curve object)
            curve_p_local, curve_t_local, curve_n_rmf_local, frame_t = rmf_frames[nearest_idx]

            # --- Apply User Tilt ---
            # Get interpolated tilt value at the vertex's specific parameter t
            interpolated_tilt = bezier.get_interpolated_tilt(spline, vertex_t)
            # Create rotation matrix around the RMF tangent (local to curve)
            tilt_rotation_matrix = Matrix.Rotation(interpolated_tilt, 3, curve_t_local)
            # Apply tilt to the RMF normal (local to curve)
            curve_n_final_local = tilt_rotation_matrix @ curve_n_rmf_local
            curve_n_final_local.normalize() # Ensure unit length

            # Calculate final binormal based on tilted frame (local to curve)
            curve_b_final_local = curve_t_local.cross(curve_n_final_local).normalized()

            # --- Transform Frame to World Space ---
            curve_p_world = curve_world_matrix @ curve_p_local
            curve_t_world = (curve_rot_scale_matrix @ curve_t_local).normalized()

            if transform_normals_correctly:
                curve_n_world = (inv_trans_matrix @ curve_n_final_local).normalized()
                curve_b_world = (inv_trans_matrix @ curve_b_final_local).normalized()
            else: # Fallback if curve matrix was non-invertible
                 curve_n_world = (curve_rot_scale_matrix @ curve_n_final_local).normalized()
                 curve_b_world = (curve_rot_scale_matrix @ curve_b_final_local).normalized()
                 # Optionally re-orthogonalize in world space if needed as fallback
                 # curve_b_world = curve_t_world.cross(curve_n_world).normalized()
                 # curve_n_world = curve_b_world.cross(curve_t_world).normalized()

            # Construct world space transformation matrix for the final tilted frame at 't'
            # Assuming T=Z, N=Y, B=X convention for how cylinder verts map
            mat_rot = Matrix((curve_b_world, curve_n_world, curve_t_world)).transposed()
            mat_frame_world = mat_rot.to_4x4()
            mat_frame_world.translation = curve_p_world

            # Original vertex position relative to the cylinder spine (XY plane offset)
            original_xy_offset_vec = Vector((original_co.x, original_co.y, 0.0))

            # Transform offset vector to world space using the calculated frame
            world_pos = mat_frame_world @ original_xy_offset_vec

            # Transform the final world position back into the target object's local space
            local_pos = target_inv_matrix @ world_pos

            # Store result in the flat list
            idx = i * 3
            new_coords_flat[idx]     = local_pos.x
            new_coords_flat[idx + 1] = local_pos.y
            new_coords_flat[idx + 2] = local_pos.z

        # --- Update Mesh Vertices Efficiently (on evaluated mesh) ---
        try:
            # Directly set the coordinates of the evaluated mesh vertices
            verts_eval.foreach_set("co", new_coords_flat)
        except Exception as e:
            # Handle potential errors during coordinate setting
            print(f"{self.bl_label} Error: Failed to set vertex coordinates: {e}")