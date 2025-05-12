import bpy
import gpu
from gpu_extras.batch import batch_for_shader # For easier drawing
from mathutils import Vector, Matrix

from . import common_vars as cvars


class PIECEGEN_OT_modal_point_scale(bpy.types.Operator):
    """Scales custom per-point attributes on a Bezier curve modally"""
    bl_idname = "piecegen.modal_point_scale"
    bl_label = "PieceGen Point Scale (Modal)"
    bl_options = {'REGISTER', 'UNDO'}

    # --- Python Instance Attributes (for internal modal state) ---
    # These will be created in invoke()
    active_curve_ob_name: str
    selected_point_indices: list
    initial_scales_per_point: list # List of [sx, sy, sz] lists
    
    initial_mouse_x: int
    initial_mouse_y: int
    pivot_point_world: Vector
    constraint_axis_str: str
    live_scale_factors: Vector
    
    _draw_handler = None
    _draw_args = None
    
    # --- Shader (Class attribute, initialized once) ---
    shader_builtin_flat_color = None


    # --- Static Draw Callback Wrapper ---
    @staticmethod
    def _draw_callback_wrapper(op_instance, context_from_handler_args):
        # This static method is what draw_handler_add gets.
        # op_instance is 'self' (the operator instance) that was passed in _draw_args.
        # context_from_handler_args is the 'context' that was passed in _draw_args.
        if op_instance:
            op_instance.draw_scale_visuals(context_from_handler_args) # Call the instance method

    # --- Instance method for actual drawing logic ---
    def draw_scale_visuals(self, context):
        # Ensure class-level shader is initialized
        if PIECEGEN_OT_modal_point_scale.shader_builtin_flat_color is None:
            return

        shader = PIECEGEN_OT_modal_point_scale.shader_builtin_flat_color
        
        rv3d = context.space_data.region_3d
        projection_matrix = rv3d.window_matrix 
        view_matrix = rv3d.view_matrix
        mvp_matrix = projection_matrix @ view_matrix

        # Visual properties
        base_visual_length = 0.5  # Base length of axis lines when scale factor is 1.0
        line_width = 2.0 
        pivot_point_size = 6.0
        end_cap_point_size = 20.0 # Slightly larger for end caps
        inactive_length_factor = 0.25 # How much shorter inactive axes are
        inactive_alpha_factor = 0.4   # How much dimmer inactive axes are

        color_map = {
            'X': Vector((1.0, 0.2, 0.2, 0.9)), 
            'Y': Vector((0.2, 1.0, 0.2, 0.9)), 
            'Z': Vector((0.2, 0.2, 1.0, 0.9)),
            'PIVOT': Vector((0.9, 0.9, 0.2, 1.0)) # Brighter pivot
        }
        
        pivot_vec = self.pivot_point_world 
        # live_scale_factors (sfx, sfy, sfz) are multipliers for the initial data scale
        # For visuals, they show how much the unit guide lines are stretched
        s_factor_x, s_factor_y, s_factor_z = self.live_scale_factors 
        
        current_constraint = self.constraint_axis_str.upper()

        world_axes = {
            'X': Vector((1.0, 0.0, 0.0)),
            'Y': Vector((0.0, 1.0, 0.0)),
            'Z': Vector((0.0, 0.0, 1.0))
        }

        line_positions = []
        line_colors = []
        point_positions = [pivot_vec] # Start with pivot point
        point_colors = [color_map['PIVOT'].to_tuple()] # Ensure tuple for color data

        for axis_char in ['X', 'Y', 'Z']:
            is_active = axis_char in current_constraint or current_constraint == "XYZ"
            
            scale_factor_for_axis = 1.0
            if axis_char == 'X': scale_factor_for_axis = s_factor_x
            elif axis_char == 'Y': scale_factor_for_axis = s_factor_y
            elif axis_char == 'Z': scale_factor_for_axis = s_factor_z

            current_length = base_visual_length * (scale_factor_for_axis if is_active else inactive_length_factor)
            
            # Ensure current_length is not extremely small or negative if scale_factors can be
            current_length = max(0.001, current_length)

            line_end_positive = pivot_vec + world_axes[axis_char] * current_length
            # Blender's gizmo often shows lines in both positive and negative directions from pivot
            line_end_negative = pivot_vec - world_axes[axis_char] * current_length 
            
            current_color_vec = color_map[axis_char].copy() # Get base color
            if not is_active:
                current_color_vec.w *= inactive_alpha_factor # Dim alpha for inactive axes
                # Optionally, desaturate color too for inactive
                # current_color_vec.xyz = current_color_vec.xyz * 0.5 + Vector((0.5,0.5,0.5)) * 0.5


            # Positive direction line
            line_positions.extend([pivot_vec, line_end_positive])
            line_colors.extend([current_color_vec.to_tuple(), current_color_vec.to_tuple()])
            
            # Negative direction line (optional, but common for scale gizmo)
            line_positions.extend([pivot_vec, line_end_negative])
            line_colors.extend([current_color_vec.to_tuple(), current_color_vec.to_tuple()])

            if is_active: # Add end caps for active axes
                point_positions.append(line_end_positive)
                point_colors.append(current_color_vec.to_tuple())
                point_positions.append(line_end_negative)
                point_colors.append(current_color_vec.to_tuple())

        # --- Drawing ---
        shader.bind()
        shader.uniform_float("ModelViewProjectionMatrix", mvp_matrix)

        gpu.state.blend_set('ALPHA') # Enable blending for transparency
        
        # Draw all lines
        if line_positions:
            gpu.state.line_width_set(line_width)
            line_content = {"pos": line_positions, "color": line_colors}
            batch_lines = batch_for_shader(shader, 'LINES', line_content)
            batch_lines.draw(shader)
            gpu.state.line_width_set(1.0) # Reset line width

        # Draw all points (pivot + end caps)
        if point_positions:
            # Note: point_size applies to all points in this batch.
            # If you want different sizes for pivot vs caps, you'd need separate batches or a more complex shader.
            gpu.state.point_size_set(end_cap_point_size) # Use end_cap_point_size for all for simplicity
            point_content = {"pos": point_positions, "color": point_colors}
            batch_points = batch_for_shader(shader, 'POINTS', point_content)
            batch_points.draw(shader)
            # Reset point size if it could affect other drawings: gpu.state.point_size_set(1.0) 

        gpu.state.blend_set('NONE') # Reset blend state # Reset blend state
        # gpu.state.point_size_set(1.0) # Optional: reset point size if it affects other drawings


    def invoke(self, context, event):
        if not self._is_valid_state(context, check_level='invoke_pre_init'):
            return {'CANCELLED'}
        
        obj = context.active_object
        curve_data = obj.data
        spline = curve_data.splines[0]

        # --- Initialize or retrieve custom scale property from curve data ---
        scales_collection: bpy.props.CollectionProperty = curve_data.piecegen_custom_scales
        num_bezier_points = len(spline.bezier_points)
        # Ensure collection has enough items
        while len(scales_collection) < num_bezier_points:
            prop_pt_scale: cvars.PieceGenPointScaleValues = scales_collection.add()
            prop_pt_scale.scale = (1, 1, 1)
        # Trim if too many (e.g., if curve points were deleted)
        while len(scales_collection) > num_bezier_points:
            scales_collection.remove(len(scales_collection) - 1)


        # --- Initialize Python instance attributes for modal state ---
        self.active_curve_ob_name = obj.name
        self.initial_mouse_x = event.mouse_x # Standard Python int
        self.initial_mouse_y = event.mouse_y

        self.selected_point_indices = [i for i, bp in enumerate(spline.bezier_points) if bp.select_control_point]
        if not self.selected_point_indices:
            self.report({'INFO'}, "No curve points selected to scale.")
            return {'CANCELLED'}

        self.initial_scales_per_point = [] 
        selected_points_coords_local = []

        for idx in self.selected_point_indices:
            # Ensure index is within bounds for all_point_scales_prop
            if idx < len(scales_collection):
                self.initial_scales_per_point.append(list(scales_collection[idx]))
            else: 
                self.initial_scales_per_point.append([1.0, 1.0, 1.0]) # Fallback default
            
            if idx < len(spline.bezier_points):
                 selected_points_coords_local.append(spline.bezier_points[idx].co.copy())
            
        pivot_local = sum(selected_points_coords_local, Vector()) / len(selected_points_coords_local)
        self.pivot_point_world = obj.matrix_world @ pivot_local
        
        self.live_scale_factors = Vector((1.0, 1.0, 1.0)) 
        self.constraint_axis_str = "XYZ"

        # --- Initialize shader (once per operator class, if not already) ---
        if PIECEGEN_OT_modal_point_scale.shader_builtin_flat_color is None:
            PIECEGEN_OT_modal_point_scale.shader_builtin_flat_color = gpu.shader.from_builtin('FLAT_COLOR')

        # --- Add draw handler ---
        self._draw_args = (self, context) 
        self._draw_handler = bpy.types.SpaceView3D.draw_handler_add(
            PIECEGEN_OT_modal_point_scale._draw_callback_wrapper,
            self._draw_args, 'WINDOW', 'POST_VIEW'
        )

        context.window_manager.modal_handler_add(self)
        self._update_header_text(context)
        return {'RUNNING_MODAL'}

    def _update_header_text(self, context):
        # Ensure live_scale_factors exists and is a Vector or tuple/list of 3 floats
        sfx, sfy, sfz = (1.0, 1.0, 1.0) # Defaults
        if hasattr(self, 'live_scale_factors') and self.live_scale_factors:
            sfx, sfy, sfz = self.live_scale_factors

        mode_txt = self.constraint_axis_str if hasattr(self, 'constraint_axis_str') and self.constraint_axis_str else "Uniform"
        
        header_txt = (
            f"PieceGen Scale: Mode: {mode_txt} | "
            f"Factor: X:{sfx:.2f} Y:{sfy:.2f} Z:{sfz:.2f} | "
            f"LMB/Enter: Confirm, RMB/Esc: Cancel, X/Y/Z: Constrain, Shift+X/Y/Z: Planar Constraint"
        )
        context.area.header_text_set(header_txt)

    def _cleanup(self, context, cancelled=False):
        context.area.header_text_set(None)
        bpy.types.SpaceView3D.draw_handler_remove(self._draw_handler, 'WINDOW')
        self._draw_handler = None
        
        if cancelled:
            obj = context.scene.objects.get(self.active_curve_ob_name if hasattr(self, 'active_curve_ob_name') else None)
            if obj and obj.type == 'CURVE' and obj.data: 
                scales_collection = obj.data.piecegen_custom_scales
                for i, point_idx in enumerate(self.selected_point_indices):
                    initial_s_vec = self.initial_scales_per_point[i]
                    scales_collection[point_idx].scale = initial_s_vec # Restore the Vector
                    
                obj.data.update_tag()
                context.area.tag_redraw()

    def modal(self, context, event):
        if not self._is_valid_state(context, check_level='modal'):
            self._cleanup(context, cancelled=True) 
            # _is_valid_state already reports, so just cancel
            return {'CANCELLED'}

        obj = context.scene.objects.get(self.active_curve_ob_name)
        
        curve_data = obj.data
        scales_collection = curve_data.piecegen_custom_scales

        context.area.tag_redraw() # Request redraw for draw handler

        # Handle mouse movement for scaling
        # --- Handle Confirmation Events FIRST ---
        if event.type in {'SPACE', 'RET', 'NUMPAD_ENTER'}:
            if event.value == 'PRESS': # Or 'RELEASE' if you prefer confirm on release
                self._cleanup(context) # Perform cleanup (remove draw handler, etc.)
                bpy.ops.ed.undo_push(message="PieceGen Point Scale") # Add to undo stack
                self.report({'INFO'}, "Scale confirmed.")
                return {'FINISHED'}

        # --- Handle Cancellation Events ---
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            if event.value == 'PRESS': # Usually cancel on press for these keys
                self._cleanup(context, cancelled=True) # True to restore initial scales
                self.report({'INFO'}, "Scale cancelled.")
                return {'CANCELLED'}
            
        # --- Handle Mouse Movement for Scaling ---
        elif event.type == 'MOUSEMOVE':
            delta_x = event.mouse_x - self.initial_mouse_x
            delta_y = event.mouse_y - self.initial_mouse_y # Calculate Y delta

            # Base sensitivity - you might want different sensitivities for X and Y mouse movements
            # For example, delta_x might represent horizontal screen space, delta_y vertical.
            # Let's assume a general "drag magnitude" or use delta_x for X/Uniform and delta_y for Y/Z.
            
            scale_factor_from_dx = 1.0 + (delta_x / 200.0)
            scale_factor_from_dy = 1.0 + (delta_y / 200.0) # Note: mouse Y often inverted; adjust if needed (e.g. -delta_y)

            effective_dx_scale = max(0.01, scale_factor_from_dx)
            effective_dy_scale = max(0.01, scale_factor_from_dy)

            sfx, sfy, sfz = 1.0, 1.0, 1.0 # Initialize to no change from current live_scale_factors state
                                        # These factors are MULTIPLIERS for initial_scales_per_point
            current_mode = self.constraint_axis_str.upper()

            if current_mode == "XYZ": # Uniform scaling driven by horizontal mouse movement
                sfx = sfy = sfz = effective_dx_scale
            elif current_mode == 'X':
                sfx = effective_dx_scale
            elif current_mode == 'Y':
                # Option: Y-scale driven by horizontal mouse movement (like X)
                sfy = effective_dx_scale
                # Option: Y-scale driven by vertical mouse movement
                # sfy = effective_dy_scale
            elif current_mode == 'Z':
                # Z-scale driven by vertical mouse movement
                sfz = effective_dy_scale # Use dy for Z
            elif current_mode == 'XY': 
                sfx = effective_dx_scale
                sfy = effective_dx_scale # Or make sfy use effective_dy_scale if preferred
                sfz = 1.0
            elif current_mode == 'XZ': 
                sfx = effective_dx_scale
                sfz = effective_dy_scale # Z uses dy
                sfy = 1.0
            elif current_mode == 'YZ': 
                sfy = effective_dx_scale # Or dy
                sfz = effective_dy_scale # Z uses dy
                sfx = 1.0
            
            self.live_scale_factors = Vector((sfx, sfy, sfz))

            for i, point_idx in enumerate(self.selected_point_indices):
                if point_idx < len(scales_collection) and i < len(self.initial_scales_per_point):
                    initial_s = self.initial_scales_per_point[i]
                    scales_collection[point_idx].scale = (
                        initial_s[0] * self.live_scale_factors.x, 
                        initial_s[1] * self.live_scale_factors.y, 
                        initial_s[2] * self.live_scale_factors.z
                    )
            
            curve_data.update_tag() # Essential for depsgraph update
            self._update_header_text(context)

        # --- Handle Axis Constraint Key Presses ---
        elif event.value == 'PRESS':
            new_constraint_mode = None
            key = event.type
            
            if key == 'X': new_constraint_mode = 'X' if not event.shift else 'YZ' # X alone, or Shift+X for YZ plane
            elif key == 'Y': new_constraint_mode = 'Y' if not event.shift else 'XZ'
            elif key == 'Z': new_constraint_mode = 'Z' if not event.shift else 'XY'

            if new_constraint_mode:
                if self.constraint_axis_str == new_constraint_mode: # Pressing same constraint again
                    # Cycle: Axis -> Plane -> Uniform (or just Axis -> Uniform)
                    if len(self.constraint_axis_str) == 1 and not event.shift: # Was single axis, no shift -> uniform
                         self.constraint_axis_str = "XYZ"
                    elif len(self.constraint_axis_str) == 2 and event.shift: # Was planar, shift + same axis origin -> uniform
                         self.constraint_axis_str = "XYZ"
                    else: # Go to the new mode (e.g. X to YZ, or XYZ to X)
                        self.constraint_axis_str = new_constraint_mode
                else:
                    self.constraint_axis_str = new_constraint_mode
                
                self.initial_mouse_x = event.mouse_x 
                self.initial_mouse_y = event.mouse_y
                self.live_scale_factors = Vector((1.0,1.0,1.0))
                # Restore scales to initial_scales before applying new constraint from neutral factors
                for i, point_idx in enumerate(self.selected_point_indices):
                     if point_idx < len(scales_collection) and i < len(self.initial_scales_per_point):
                        initial_s = self.initial_scales_per_point[i]
                        scales_collection[point_idx].scale = (initial_s[0], initial_s[1], initial_s[2])
                        
                curve_data.update_tag()
                self._update_header_text(context)

        return {'RUNNING_MODAL'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        if not (obj and obj.type == 'CURVE' and obj.data and obj.mode == 'EDIT'):
            return False
        if not (obj.data.splines and obj.data.splines[0].type == 'BEZIER'):
            return False
        # Check if any bezier point is selected (select_control_point for the main point)
        return any(bp.select_control_point for bp in obj.data.splines[0].bezier_points)
    


    def _is_valid_state(self, context, check_level='modal'):
        """
        Checks if the operator is in a valid state to proceed.
        Levels: 'invoke_pre_init', 'invoke_post_init', 'modal', 'draw'.
        Returns True if valid, False otherwise. Reports errors for critical failures.
        """
        if check_level in ['invoke_pre_init', 'invoke_post_init', 'modal', 'draw']:
            obj = context.active_object if check_level == 'invoke_pre_init' else context.scene.objects.get(getattr(self, 'active_curve_ob_name', None))
            if not (obj and obj.type == 'CURVE' and obj.data and obj.mode == 'EDIT'):
                if check_level != 'draw': # Don't spam reports from draw
                    self.report({'WARNING'}, "Curve not active or not in Edit Mode.")
                return False
            
            curve_data = obj.data
            if not (curve_data.splines and curve_data.splines[0].type == 'BEZIER'):
                if check_level != 'draw':
                    self.report({'WARNING'}, "Active curve is not a Bezier curve.")
                return False
            self.current_curve_object = obj # Store for easy access if valid
            self.current_curve_data = curve_data
            self.current_spline = curve_data.splines[0]

        if check_level == 'draw': # Specific checks for drawing visuals
            if PIECEGEN_OT_modal_point_scale.shader_builtin_flat_color is None:
                return False
            if not context.space_data or not hasattr(context.space_data, 'region_3d'):
                return False # No region_3d to draw into

        return True