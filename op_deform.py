# <pep8 compliant>
# -*- coding: utf-8 -*-
# File: op_deform.py
# Contains operators to enable/disable deformation and the core deform logic.

# Use imports from common_imports
from .common_imports import (
    bpy, np, HAS_NUMPY, Vector, Matrix, Quaternion
)
import math

# Use variables/constants from common_vars
from . import common_vars as cvars
# Use serialization functions
from .prop_serialization import pack_verts_to_string, unpack_verts_from_string
# Use bezier/RMF functions
from . import bezier

# --- Core Deformation Function ---
# This function performs the actual mesh deformation based on the curve
def deform_mesh_along_curve(target_mesh_obj: bpy.types.Object,
                            curve_guide_obj: bpy.types.Object,
                            original_verts_coords: list,
                            cyl_height: float,
                            rmf_steps: int = 50): # Default RMF steps
    """
    Applies curve deformation using pre-calculated RMF frames.
    Uses NEAREST pre-calculated frame (Simplification).
    Modifies mesh data directly. Uses numpy for speed if available.
    """
    mesh = target_mesh_obj.data

    # --- Input Validation ---
    if not curve_guide_obj or not curve_guide_obj.data or not isinstance(curve_guide_obj.data, bpy.types.Curve):
        print(f"Error: Deform failed - Invalid curve object provided.")
        return False
    curve_data = curve_guide_obj.data

    if not curve_data.splines or len(curve_data.splines) == 0 or curve_data.splines[0].type != 'BEZIER':
        print(f"Error: Deform failed - Curve's first spline must be BEZIER.")
        return False
    spline = curve_data.splines[0] # Use the first spline

    # Check if spline has enough points
    if len(spline.bezier_points) < 2:
        print(f"Error: Deform failed - Bezier spline needs at least 2 control points.")
        return False

    vert_count = len(mesh.vertices)
    if vert_count == 0 or not original_verts_coords or len(original_verts_coords) != vert_count:
        print(f"Error: Deform failed - Vertex count mismatch or missing original data.")
        return False
    if cyl_height <= 1e-6:
        print(f"Error: Deform failed - Invalid cylinder height ({cyl_height}).")
        return False
    if rmf_steps < 2: rmf_steps = 2 # Ensure minimum steps for RMF logic

    # --- Pre-calculate RMF Frames ---
    # print(f"Calculating {rmf_steps} RMF frames...") # Optional debug print
    # Call function from bezier module
    rmf_frames = bezier.calculate_rmf_frames(spline, rmf_steps)
    if not rmf_frames:
        # Error message printed inside calculate_rmf_frames
        return False
    # print("RMF calculation complete.") # Optional debug print
    num_rmf_frames = len(rmf_frames)

    # --- Get Transformation Matrices ---
    curve_world_matrix = curve_guide_obj.matrix_world
    target_inv_matrix  = target_mesh_obj.matrix_world.inverted()

    # --- Calculate Inverse-Transpose for Normal Transforms ---
    curve_rot_scale_matrix = curve_world_matrix.to_3x3()
    try:
        inv_matrix = curve_rot_scale_matrix.inverted_safe()
        inv_trans_matrix = inv_matrix.transposed()
        transform_normals_correctly = True
    except ValueError:
        inv_trans_matrix = Matrix.Identity(3) # Fallback
        transform_normals_correctly = False
        # print("Warning: Curve matrix not invertible, normal transform may be incorrect.")

    # --- Prepare Coordinate Array ---
    # Use numpy array for potentially faster operations if available
    if HAS_NUMPY:
        new_coords_np = np.empty((vert_count, 3), dtype=np.float32)
    else:
        # Fallback to standard list if numpy not installed
        new_coords_flat = [0.0] * (vert_count * 3)

    # --- Process Each Vertex ---
    for i in range(vert_count):
        original_co = original_verts_coords[i] # Original local coordinate

        # Calculate curve parameter 't' (0-1) based on original local Z and height
        vertex_t = original_co.z / cyl_height
        vertex_t = max(0.0, min(1.0, vertex_t)) # Clamp t to [0, 1]

        # --- Find Bracketing RMF Frames and Interpolation Factor ---
        t: float = vertex_t * (num_rmf_frames - 1)
        prev_frame_id = math.floor(t)
        next_frame_id = math.ceil(t)

        if prev_frame_id == next_frame_id or num_rmf_frames <= next_frame_id: 
            # Handle edge cases where vertex_t lands exactly on a frame
            prev_frame_id = max(0, min(num_rmf_frames - 1, prev_frame_id))
            interp_factor = .0
            frame_prev = rmf_frames[prev_frame_id]
            frame_next = frame_prev
        else: 
            frame_prev, frame_next = rmf_frames[prev_frame_id], rmf_frames[next_frame_id]
            # Calculate interpolation factor between frame_prev.t and frame_next.t
            t_prev_param, t_next_param = frame_prev[3], frame_next[3] # Global t of prev/next frame
            if abs(t_next_param - t_prev_param) < 1e-9: # Avoid division by zero if frame t's are identical
                interp_factor = 0.0
            else: interp_factor = (vertex_t - t_prev_param) / (t_next_param - t_prev_param)
            interp_factor = max(0.0, min(1.0, interp_factor)) # Clamp factor
        
        # --- Interpolate Frame Data ---
        # Unpack frame data
        p_prev, t_prev, n_prev, _ = frame_prev
        p_next, t_next, n_next, _ = frame_next

        # 1. Linear Interpolate Position (Lerp)
        curve_p_local = p_prev.lerp(p_next, interp_factor)

        # 2. Spherical Linear Interpolate Orientation (Slerp)
        #    Convert frame orientations (T, N) to Quaternions for Slerp
        def quat_from_bnt(n:Vector, t:Vector): 
            b = t.cross(n).normalized()
            n = b.cross(t).normalized()
            mat = Matrix((b, n, t)).transposed() # b, n, t at each column
            quat = mat.to_quaternion()
            return quat
        quat_prev, quat_next = quat_from_bnt(n_prev, t_prev), quat_from_bnt(n_next, t_next)

        # Perform Slerp
        quat_interp = quat_prev.slerp(quat_next, interp_factor)

        # Convert interpolated quaternion back to matrix to extract T, N vectors
        mat_interp = quat_interp.to_matrix()
        curve_t_local = mat_interp.col[2].normalized()
        curve_n_rmf_local = mat_interp.col[1].normalized()

        # --- Apply User Tilt ---
        # Get interpolated tilt value at the vertex's specific parameter t
        interpolated_tilt = bezier.get_interpolated_tilt(spline, vertex_t)
        # Create rotation matrix around the RMF tangent (local to curve)
        tilt_rotation_matrix = Matrix.Rotation(interpolated_tilt, 3, curve_t_local)
        # Apply tilt to the RMF normal (local to curve)
        curve_n_final_local = tilt_rotation_matrix @ curve_n_rmf_local
        curve_n_final_local.normalize() # Ensure unit length after rotation

        # Calculate final binormal based on tilted frame (local to curve)
        curve_b_final_local = curve_t_local.cross(curve_n_final_local).normalized()

        # --- Transform Frame to World Space ---
        curve_p_world = curve_world_matrix @ curve_p_local
        # Transform tangent using standard matrix
        curve_t_world = (curve_rot_scale_matrix @ curve_t_local).normalized()
        # Transform normal and binormal using inverse-transpose
        if transform_normals_correctly:
            curve_n_world = (inv_trans_matrix @ curve_n_final_local).normalized()
            curve_b_world = (inv_trans_matrix @ curve_b_final_local).normalized()
        else: # Fallback if curve matrix was non-invertible
             curve_n_world = (curve_rot_scale_matrix @ curve_n_final_local).normalized()
             curve_b_world = (curve_rot_scale_matrix @ curve_b_final_local).normalized()
             # Optional: Re-orthogonalize in world space if needed as fallback
             # curve_b_world = curve_t_world.cross(curve_n_world).normalized()
             # curve_n_world = curve_b_world.cross(curve_t_world).normalized()

        # Construct world space transformation matrix for the final tilted frame at 't'
        # Using the convention: X=Binormal, Y=Normal, Z=Tangent
        mat_rot = Matrix((curve_b_world, curve_n_world, curve_t_world)).transposed()
        mat_frame_world = mat_rot.to_4x4()
        mat_frame_world.translation = curve_p_world

        # Original vertex position relative to the cylinder spine (XY plane offset)
        original_xy_offset_vec = Vector((original_co.x, original_co.y, 0.0))

        # Transform the XY offset vector by the world frame matrix to get world position
        world_pos = mat_frame_world @ original_xy_offset_vec

        # Transform the final world position back into the target object's local space
        local_pos = target_inv_matrix @ world_pos

        # Store result
        if HAS_NUMPY:
            new_coords_np[i] = local_pos # Store Vector directly
        else:
            idx = i * 3
            new_coords_flat[idx]     = local_pos.x
            new_coords_flat[idx + 1] = local_pos.y
            new_coords_flat[idx + 2] = local_pos.z

    # --- Update Mesh Vertices Efficiently ---
    try:
        # Use foreach_set for faster vertex coordinate updates
        if HAS_NUMPY:
            # Ravel the numpy array to a flat list for foreach_set
            mesh.vertices.foreach_set("co", new_coords_np.ravel())
        else:
            mesh.vertices.foreach_set("co", new_coords_flat)
        # Mark mesh data as updated
        mesh.update()
        return True # Indicate success
    except Exception as e:
        # Handle potential errors during coordinate setting
        print(f"Error: Failed to set vertex coordinates: {e}")
        return False # Indicate failure


# --- Operator: Enable Realtime Deform ---
class OBJECT_OT_enable_realtime_bend(bpy.types.Operator):
    """Enables realtime deformation handler for the active mesh, linking it to the selected curve."""
    bl_idname = "object.enable_realtime_bend"
    bl_label = "Enable Realtime Deform"
    bl_description = "Links active mesh to selected curve for realtime updates (Can be slow!)"
    bl_options = {'REGISTER', 'UNDO'}

    # Removed class variable for cache - using global cache from common_vars

    @classmethod
    def poll(cls, context):
        """Enable operator only if an unlinked mesh is active and a curve is selected."""
        active_obj = context.active_object
        # Check if PROP_ENABLED exists and is False/None before enabling
        is_already_enabled = active_obj.get(cvars.PROP_ENABLED, False) if active_obj else False
        return (context.mode == 'OBJECT' and
                active_obj and active_obj.type == 'MESH' and
                len(context.selected_objects) == 2 and # Require exactly mesh + curve selected
                not is_already_enabled and # Check the fetched property
                any(obj.type == 'CURVE' for obj in context.selected_objects if obj != active_obj))

    def find_curve_object(self, context):
        """Finds the selected curve object (assumes only one other selected object)."""
        active_obj = context.active_object
        for obj in context.selected_objects:
            if obj != active_obj and obj.type == 'CURVE':
                return obj
        return None

    def execute(self, context):
        target_obj = context.active_object
        curve_obj = self.find_curve_object(context)

        # Access scene properties (needed for default height if not stored)
        if not hasattr(context.scene, 'ccdg_props'):
             self.report({'ERROR'}, "Addon properties not found on scene.")
             return {'CANCELLED'}
        props = context.scene.ccdg_props

        if not curve_obj:
            self.report({'ERROR'}, "No valid curve object selected alongside the active mesh.")
            return {'CANCELLED'}

        # --- Store original vertex data ---
        # Copy current vertex coordinates before deformation
        original_verts = [v.co.copy() for v in target_obj.data.vertices]
        # Pack coordinates into a string for storage in custom property
        packed_verts = pack_verts_to_string(original_verts) # Uses function from prop_serialization
        if not packed_verts:
            self.report({'ERROR'}, "Failed to pack original vertex data.")
            return {'CANCELLED'}

        # Determine the base height for mapping 't'
        # Prefer height stored previously, fallback to current panel setting
        stored_height = target_obj.get(cvars.PROP_HEIGHT)
        if stored_height is None:
            stored_height = props.height # Get height from scene properties
            print(f"Warning: Using panel height ({stored_height:.3f}) for deformation mapping as no original height was stored on '{target_obj.name}'.")
            # TODO: Optionally calculate bounds here as a more robust fallback if needed

        # --- Store state in global cache and object custom properties ---
        cvars.original_coords_cache[target_obj.name] = original_verts # Store in global cache
        target_obj[cvars.PROP_ENABLED] = True # Mark as enabled
        target_obj[cvars.PROP_CURVE_NAME] = curve_obj.name # Store linked curve name
        target_obj[cvars.PROP_ORIG_VERTS] = packed_verts # Store packed original verts
        target_obj[cvars.PROP_HEIGHT] = stored_height # Store the height used for mapping

        # Add object name to the set monitored by the depsgraph handler
        cvars.MONITORED_MESH_OBJECTS.add(target_obj.name)

        # --- Apply initial deformation ---
        self.report({'INFO'}, "Applying initial deformation...")
        # Call the deformation function defined in this module
        success = deform_mesh_along_curve(target_obj, curve_obj, original_verts, stored_height)

        if success:
            self.report({'INFO'}, f"Enabled realtime deform for '{target_obj.name}' linked to '{curve_obj.name}'.")
            context.area.tag_redraw() # Update UI to reflect enabled state
            return {'FINISHED'}
        else:
            # Clean up if initial deformation failed
            target_obj.pop(cvars.PROP_ENABLED, None)
            target_obj.pop(cvars.PROP_CURVE_NAME, None)
            target_obj.pop(cvars.PROP_ORIG_VERTS, None)
            target_obj.pop(cvars.PROP_HEIGHT, None)
            cvars.original_coords_cache.pop(target_obj.name, None) # Clear cache
            cvars.MONITORED_MESH_OBJECTS.discard(target_obj.name) # Remove from monitored set
            self.report({'ERROR'}, "Initial deformation failed.")
            return {'CANCELLED'}

# --- Operator: Disable Realtime Deform ---
class OBJECT_OT_disable_realtime_bend(bpy.types.Operator):
    """Disables realtime deformation handler and restores original mesh shape."""
    bl_idname = "object.disable_realtime_bend"
    bl_label = "Disable Realtime Deform"
    bl_description = "Stops realtime updates and restores original mesh shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        """Enable operator only if the active object is a mesh and currently enabled."""
        return (context.active_object and context.active_object.type == 'MESH' and
                context.active_object.get(cvars.PROP_ENABLED, False)) # Check enabled flag

    def execute(self, context):
        target_obj = context.active_object
        mesh = target_obj.data
        obj_name = target_obj.name

        # --- Restore original shape ---
        # Retrieve packed original vertex data from custom property
        packed_verts = target_obj.get(cvars.PROP_ORIG_VERTS)
        if packed_verts:
            # Unpack the vertex data
            original_verts = unpack_verts_from_string(packed_verts) # Uses function from prop_serialization
            # Check if unpacking was successful and vertex count matches current mesh
            if original_verts and len(original_verts) == len(mesh.vertices):
                 try:
                     # Prepare flat list for foreach_set
                     if HAS_NUMPY: # Use numpy if available
                          flat_coords = np.array(original_verts, dtype=np.float32).ravel()
                     else:
                          flat_coords = [c for v_co in original_verts for c in v_co]
                     # Apply original coordinates back to the mesh
                     mesh.vertices.foreach_set("co", flat_coords)
                     mesh.update() # Update mesh state
                     self.report({'INFO'}, f"Restored original shape for '{obj_name}'.")
                 except Exception as e:
                     self.report({'ERROR'}, f"Failed to restore vertex coords for '{obj_name}': {e}")
            elif original_verts:
                # Vertex count mismatch - cannot restore reliably
                self.report({'WARNING'}, f"Could not restore '{obj_name}': Vertex count changed since deform was enabled.")
            else:
                # Unpacking failed
                self.report({'WARNING'}, f"Could not restore '{obj_name}': Failed to unpack original vertex data.")
        else:
            # No original data found - maybe it was cleared or never set
            self.report({'WARNING'}, f"No original shape data found for '{obj_name}'. Cannot restore.")

        # --- Cleanup ---
        # Remove object from monitored set
        cvars.MONITORED_MESH_OBJECTS.discard(obj_name)
        # Remove custom properties from the object
        target_obj.pop(cvars.PROP_ENABLED, None)
        target_obj.pop(cvars.PROP_CURVE_NAME, None)
        target_obj.pop(cvars.PROP_ORIG_VERTS, None)
        target_obj.pop(cvars.PROP_HEIGHT, None)
        # Clear entry from the global coordinate cache
        cvars.original_coords_cache.pop(obj_name, None)

        self.report({'INFO'}, f"Disabled realtime deform for '{obj_name}'.")
        context.area.tag_redraw() # Update UI
        return {'FINISHED'}
