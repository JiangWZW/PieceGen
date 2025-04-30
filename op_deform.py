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

# --- Core Deformation Function (Multi-Frame Weighted Blend Version) ---
def deform_mesh_along_curve(target_mesh_obj: bpy.types.Object,
                            curve_guide_obj: bpy.types.Object,
                            original_verts_coords: list,
                            cyl_height: float,
                            rmf_steps: int = 50,
                            influence_count: int = 4): # Number of frames to blend
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
    influence_count = max(2, influence_count) # Need at least 2 for blending

    # --- Pre-calculate RMF Frames ---
    rmf_frames = bezier.calculate_rmf_frames(spline, rmf_steps)
    if not rmf_frames: return False
    num_rmf_frames = len(rmf_frames)
    if num_rmf_frames < 2: return False # Cannot blend with fewer than 2 frames

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

    # --- Pre-calculate World Frame Data (Position & Quaternion) ---
    # Store world position and world orientation quaternion for each RMF frame.
    world_frame_data = []
    print(f"Pre-calculating {num_rmf_frames} world frame positions & quaternions...") # Debug
    for frame_data in rmf_frames:
        pos_local, tan_local, norm_rmf_local, frame_t = frame_data

        # Apply tilt locally
        interpolated_tilt = bezier.get_interpolated_tilt(spline, frame_t)
        tilt_rot = Matrix.Rotation(interpolated_tilt, 3, tan_local)
        norm_final_local = tilt_rot @ norm_rmf_local
        norm_final_local.normalize()
        bino_final_local = tan_local.cross(norm_final_local).normalized()
        # Ensure normal is orthogonal after potential precision loss
        norm_final_local = bino_final_local.cross(tan_local).normalized()

        # Transform position to world
        pos_world = curve_world_matrix @ pos_local

        # Transform orientation vectors to world
        tan_world = (curve_rot_scale_matrix @ tan_local).normalized()
        if transform_normals_correctly:
            norm_world = (inv_trans_matrix @ norm_final_local).normalized()
            bino_world = (inv_trans_matrix @ bino_final_local).normalized()
        else:
            norm_world = (curve_rot_scale_matrix @ norm_final_local).normalized()
            bino_world = (curve_rot_scale_matrix @ bino_final_local).normalized()

        # Re-orthogonalize world frame just in case
        bino_world = tan_world.cross(norm_world).normalized()
        if bino_world.length_squared > 1e-9:
             norm_world = bino_world.cross(tan_world).normalized()

        # Create world rotation matrix and convert to quaternion
        mat_rot_world = Matrix((bino_world, norm_world, tan_world)).transposed()
        quat_world = mat_rot_world.to_quaternion()

        # Store world position, world quaternion, and original frame t
        world_frame_data.append({'pos': pos_world, 'quat': quat_world, 't': frame_t})
    print("World frame data calculated.") # Debug


    # --- Prepare Coordinate Array ---
    # Use numpy array for potentially faster operations if available
    if HAS_NUMPY: new_coords_np = np.empty((vert_count, 3), dtype=np.float32)
    else: new_coords_flat = [0.0] * (vert_count * 3) # Fallback to standard list if numpy not installed

    # --- Process Each Vertex ---
    for i in range(vert_count):
        original_co = original_verts_coords[i] # Original local coordinate
        # Calculate curve parameter 't' (0-1) based on original local Z and height
        vertex_t = original_co.z / cyl_height
        vertex_t = max(0.0, min(1.0, vertex_t)) # Clamp t to [0, 1]

        # --- Find Influencing Frames and Calculate Weights ---
        # Determine the 'central' index based on t
        t_scaled = vertex_t * (num_rmf_frames - 1)
        center_idx_float = t_scaled
        center_idx_int = int(round(center_idx_float)) # Nearest frame index

        # Determine the range of indices to consider (e.g., 4 nearest)
        # Calculate start index, ensuring it doesn't go below 0
        half_influence = influence_count // 2
        start_idx = max(0, center_idx_int - half_influence + (1 if influence_count % 2 == 0 else 0) ) # Adjust for even counts
        # Calculate end index, ensuring it doesn't exceed bounds
        end_idx = min(num_rmf_frames, start_idx + influence_count)
        # Adjust start index again if end index hit the boundary
        start_idx = max(0, end_idx - influence_count)

        # Get the indices of the influencing frames
        influence_indices = range(start_idx, end_idx)
        num_influences = len(influence_indices)

        # Calculate weights based on inverse distance in parameter space
        weights = []
        total_inv_dist = 0.0
        epsilon = 1e-6 # To prevent division by zero

        for k in influence_indices:
            frame_t_param = world_frame_data[k]['t']
            dist_sq = (vertex_t - frame_t_param)**2
            inv_dist = 1.0 / (math.sqrt(dist_sq) + epsilon)
            weights.append(inv_dist)
            total_inv_dist += inv_dist

        # Normalize weights
        if total_inv_dist > epsilon:
            for k in range(num_influences):
                weights[k] /= total_inv_dist
        else:
            # Handle case where all distances are zero (vertex_t matches a frame_t exactly)
            # Assign full weight to the matching frame
            exact_match_idx_local = -1
            for k_idx, k_frame_idx in enumerate(influence_indices):
                 if abs(vertex_t - world_frame_data[k_frame_idx]['t']) < epsilon:
                      exact_match_idx_local = k_idx
                      break
            if exact_match_idx_local != -1:
                 for k in range(num_influences): weights[k] = 0.0
                 weights[exact_match_idx_local] = 1.0
            else: # Should not happen if epsilon is small, but fallback: equal weights
                 for k in range(num_influences): weights[k] = 1.0 / num_influences


        # --- Apply Weighted Transformation ---
        original_xy_offset_vec = Vector((original_co.x, original_co.y, 0.0))

        # Blend Positions (Linear Interpolation)
        final_world_pos = Vector((0.0, 0.0, 0.0))
        # Blend Orientations (NLERP - Normalized Linear Quaternion Blending)
        final_world_quat = Quaternion((0.0, 0.0, 0.0, 0.0)) # Accumulator
        ref_quat = None # Reference for consistent quaternion signs

        for k_idx, frame_idx in enumerate(influence_indices):
            weight = weights[k_idx]
            frame_world_pos = world_frame_data[frame_idx]['pos']
            frame_world_quat = world_frame_data[frame_idx]['quat'].copy() # Use copy

            # --- Position Blending ---
            final_world_pos += frame_world_pos * weight

            # --- Orientation Blending (NLERP) ---
            # Ensure quaternion signs are consistent for averaging
            # Compare with the first contributing quaternion (or the accumulator)
            if ref_quat is None:
                 ref_quat = frame_world_quat
            if ref_quat.dot(frame_world_quat) < 0.0:
                frame_world_quat.negate() # Flip sign for shortest path

            # Accumulate weighted quaternion
            final_world_quat.w += frame_world_quat.w * weight
            final_world_quat.x += frame_world_quat.x * weight
            final_world_quat.y += frame_world_quat.y * weight
            final_world_quat.z += frame_world_quat.z * weight

        # Normalize the accumulated quaternion for NLERP
        final_world_quat.normalize()

        # Construct the final world transformation matrix from blended components
        mat_frame_world = final_world_quat.to_matrix().to_4x4()
        mat_frame_world.translation = final_world_pos

        # Transform the original offset by the blended world matrix
        world_pos_deformed = mat_frame_world @ original_xy_offset_vec

        # Transform the final world position back into the target object's local space
        local_pos = target_inv_matrix @ world_pos_deformed

        # Store result
        if HAS_NUMPY:
            new_coords_np[i] = local_pos
        else:
            idx = i * 3
            new_coords_flat[idx:idx+3] = local_pos.x, local_pos.y, local_pos.z

    # --- Update Mesh Vertices Efficiently ---
    try:
        if HAS_NUMPY: mesh.vertices.foreach_set("co", new_coords_np.ravel())
        else: mesh.vertices.foreach_set("co", new_coords_flat)
        mesh.update(); return True
    except Exception as e: print(f"Error: Failed to set vertex coordinates: {e}"); return False

# --- Operator: Toggle Realtime Deform ---
class OBJECT_OT_toggle_realtime_bend(bpy.types.Operator):
    """Toggles realtime deformation handler for the active mesh using the chosen curve."""
    bl_idname = "object.toggle_realtime_bend" # New ID for the toggle operator
    bl_label = "Toggle Realtime Deform"
    bl_description = "Starts or stops realtime curve deformation for the active mesh"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        """Enable operator only if the active object is a mesh."""
        # Curve validity is checked during execute when enabling
        return context.active_object and context.active_object.type == 'MESH'

    def execute(self, context):
        target_obj = context.active_object
        scene = context.scene
        # Access the scene property group where the panel stores settings
        if not hasattr(scene, 'ccdg_props'):
            self.report({'ERROR'}, "Addon properties not found on scene.")
            return {'CANCELLED'}
        props = scene.ccdg_props

        # Check current state of the active object
        is_enabled = target_obj.get(cvars.PROP_ENABLED, False)

        if not is_enabled:
            # --- Enable Logic ---
            # Get the curve selected in the panel's PointerProperty
            curve_obj = props.target_curve
            if not curve_obj:
                self.report({'ERROR'}, "No Target Curve selected in the panel.")
                return {'CANCELLED'}

            # Validate the selected curve object *before* proceeding
            if curve_obj.type != 'CURVE' or not curve_obj.data or \
               not curve_obj.data.splines or len(curve_obj.data.splines) == 0 or \
               curve_obj.data.splines[0].type != 'BEZIER' or \
               len(curve_obj.data.splines[0].bezier_points) < 2:
                self.report({'ERROR'}, f"Selected Target Curve '{curve_obj.name}' is not a valid Bezier curve with >= 2 points.")
                return {'CANCELLED'}

            # Store original vertex data
            self.report({'INFO'}, f"Storing original shape for '{target_obj.name}'...")
            original_verts = [v.co.copy() for v in target_obj.data.vertices]
            # Use packing function from prop_serialization module
            packed_verts = pack_verts_to_string(original_verts)
            if not packed_verts:
                self.report({'ERROR'}, "Failed to pack original vertex data.")
                return {'CANCELLED'}

            # Determine base height for mapping t
            # Prefer height stored previously on the object, fallback to current generation panel setting
            stored_height = target_obj.get(cvars.PROP_HEIGHT)
            if stored_height is None:
                # Use height from generation panel settings if not found on object
                stored_height = props.height # Get height from scene properties
                print(f"Warning: Using panel height ({stored_height:.3f}) for deform mapping on '{target_obj.name}'. Ensure this matches the mesh.")

            # Store state in global cache (common_vars) and object custom properties (common_vars)
            cvars.original_coords_cache[target_obj.name] = original_verts
            target_obj[cvars.PROP_ENABLED] = True
            target_obj[cvars.PROP_CURVE_NAME] = curve_obj.name # Store linked curve name
            target_obj[cvars.PROP_ORIG_VERTS] = packed_verts
            target_obj[cvars.PROP_HEIGHT] = stored_height # Store the height used for mapping

            # Add object name to the set monitored by the depsgraph handler
            cvars.MONITORED_MESH_OBJECTS.add(target_obj.name)

            # Apply initial deformation
            self.report({'INFO'}, "Applying initial deformation...")
            # Get RMF steps from the panel property
            rmf_steps = props.rmf_steps_deform
            # Call the deformation function defined in this module
            success = deform_mesh_along_curve(target_obj, curve_obj, original_verts, stored_height, rmf_steps)

            if success:
                self.report({'INFO'}, f"Started realtime deform for '{target_obj.name}' with '{curve_obj.name}'.")
                # --- Switch context for immediate curve editing ---
                try:
                    # Deselect mesh, select curve, make curve active
                    target_obj.select_set(False)
                    curve_obj.select_set(True)
                    context.view_layer.objects.active = curve_obj
                    # Switch to Edit mode if not already there
                    if context.mode != 'EDIT_CURVE':
                        bpy.ops.object.mode_set(mode='EDIT')
                except Exception as e:
                    # Don't cancel if context switch fails, just warn
                    print(f"Warning: Could not switch context to curve edit mode: {e}")
                # Update the UI panel
                context.area.tag_redraw()
                return {'FINISHED'}
            else:
                # Clean up if initial deformation failed
                target_obj.pop(cvars.PROP_ENABLED, None); target_obj.pop(cvars.PROP_CURVE_NAME, None)
                target_obj.pop(cvars.PROP_ORIG_VERTS, None); target_obj.pop(cvars.PROP_HEIGHT, None)
                cvars.original_coords_cache.pop(target_obj.name, None)
                cvars.MONITORED_MESH_OBJECTS.discard(target_obj.name)
                self.report({'ERROR'}, "Initial deformation failed."); return {'CANCELLED'}

        else:
            # --- Disable Logic ---
            mesh = target_obj.data
            obj_name = target_obj.name
            self.report({'INFO'}, f"Stopping realtime deform for '{obj_name}'.")

            # Restore original shape
            packed_verts = target_obj.get(cvars.PROP_ORIG_VERTS)
            if packed_verts:
                # Use unpacking function from prop_serialization module
                original_verts = unpack_verts_from_string(packed_verts)
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
                         # print(f"Restored original shape for '{obj_name}'.") # Report below
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

            # --- Cleanup State ---
            # Remove object from monitored set (common_vars)
            cvars.MONITORED_MESH_OBJECTS.discard(obj_name)
            # Remove custom properties from the object (using keys from common_vars)
            target_obj.pop(cvars.PROP_ENABLED, None)
            target_obj.pop(cvars.PROP_CURVE_NAME, None)
            target_obj.pop(cvars.PROP_ORIG_VERTS, None)
            target_obj.pop(cvars.PROP_HEIGHT, None)
            # Clear entry from the global coordinate cache (common_vars)
            cvars.original_coords_cache.pop(obj_name, None)

            # --- Switch context back to Object mode ---
            if context.mode != 'OBJECT':
                try:
                    bpy.ops.object.mode_set(mode='OBJECT')
                except Exception as e:
                    # Don't cancel if mode switch fails, just warn
                    print(f"Warning: Could not switch back to object mode: {e}")

            self.report({'INFO'}, f"Stopped realtime deform for '{obj_name}'.")
            # Update the UI panel
            context.area.tag_redraw()
            return {'FINISHED'}


