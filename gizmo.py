# <pep8 compliant>
# -*- coding: utf-8 -*-
# File: gizmos.py
# Contains Gizmo definitions for PieceGen addon using target_set_handler
# DEBUGGING: Focus on why poll() is not called.

import bpy
import math
from mathutils import Vector, Matrix, Euler
from bpy_extras import view3d_utils # Needed for mouse projection

from . import common_vars as cvars
from . import op_custom_ui

PIXELS_PER_WORLD_UNIT = 30.0  # Gizmo circle will try to span this many pixels for each unit of data radius
MIN_VISUAL_HANDLE_SIZE_PX = 8.0   # Minimum visual size of the handle on screen, in pixels
MAX_VISUAL_HANDLE_SIZE_PX = 100.0 # Maximum visual size of the handle on screen, in pixels

TARGET_SCREEN_RADIUS_PX = 45.0  # Desired pixel radius for the gizmo circle on screen
MIN_RADIUS_VALUE = 0.05         # Define the minimum allowed radius value
MAX_RADIUS_VALUE = 20.0         # Define the maximum allowed radius value

# --- Helper Function for Disk Geometry ---
def generate_unit_disk_geometry(segments: int):
    """
    Generates vertices and faces for a unit radius disk (filled circle)
    centered at the origin, lying in the XY plane.

    Args:
        segments (int): Number of segments for the circle's circumference.

    Returns:
        tuple: (list_of_vectors_for_verts, list_of_tuples_for_faces)
    """
    verts = []
    faces = []
    radius = 1.0  # Unit radius; actual size will come from the gizmo's matrix_basis

    # Add center vertex (local origin)
    verts.append(Vector((0.0, 0.0, 0.0)))  # Vertex index 0

    # Add outer vertices for the circumference
    for i in range(segments):
        angle = (i / segments) * 2 * math.pi
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        verts.append(Vector((x, y, 0.0)))  # Vertex indices 1 to 'segments'

    # Create faces (triangle fan connecting center to outer edges)
    # Each face is a tuple of vertex indices
    for i in range(segments):
        v_outer_current_idx = i + 1  # Index of current outer vertex (1 to 'segments')
        # Index of next outer vertex, wrapping around for the last triangle
        v_outer_next_idx = ((i + 1) % segments) + 1
        
        # Triangles are (center_vertex_idx, outer_vertex_1_idx, outer_vertex_2_idx)
        faces.append(0)
        faces.append(v_outer_current_idx)
        faces.append(v_outer_next_idx)

    flattened_verts = [verts[iv] for iv in faces]
    return verts, faces, flattened_verts


# --- Operator for Gizmo Interaction ---
class PIECEGEN_OT_set_radius(bpy.types.Operator):
    """Operator to set the radius for a specific curve point via Gizmo"""
    bl_idname = "piecegen.set_radius"
    bl_label = "Set PieceGen Point Radius"
    bl_options = {'REGISTER', 'UNDO'}

    point_index: bpy.props.IntProperty(name="Point Index", default=-1)
    # Gizmo system updates this 'value' property during drag
    value: bpy.props.FloatProperty(name="Radius Value")
    initial_radius: float = 1.0

    @classmethod
    def poll(cls, context):
        obj = context.object
        return (context.mode == 'EDIT_CURVE' and
                obj and obj.type == 'CURVE' and
                obj.data and cvars and cvars.PROP_RADIUS_ARRAY in obj.data)

    def invoke(self, context, event):
        """Initialize the operator when gizmo interaction starts."""
        obj = context.object
        if not obj or self.point_index < 0 or not obj.data or cvars.PROP_RADIUS_ARRAY not in obj.data:
            return {'CANCELLED'}
        curve_data = obj.data
        radius_array_prop = curve_data.get(cvars.PROP_RADIUS_ARRAY)
        if not radius_array_prop or self.point_index >= len(radius_array_prop):
            return {'CANCELLED'}

        self.initial_radius = float(radius_array_prop[self.point_index])
        self.value = self.initial_radius # Initialize operator value

        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        """Handle mouse events during gizmo drag."""
        obj = context.object
        if not obj or self.point_index < 0: return {'CANCELLED'}
        curve_data = obj.data
        radius_array_prop = curve_data.get(cvars.PROP_RADIUS_ARRAY)
        if not radius_array_prop or self.point_index >= len(radius_array_prop): return {'CANCELLED'}
        
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            # Read the value updated by the gizmo system
            new_radius = max(0.01, self.value)
            try: # Apply the change
                curve_data[cvars.PROP_RADIUS_ARRAY][self.point_index] = new_radius
                curve_data.update_tag()
                for window in context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D': area.tag_redraw()
            except Exception as e:
                 print(f"Modal Error: Failed to set radius: {e}")
                 return {'CANCELLED'}
            context.area.header_text_set(f"Radius: {new_radius:.3f}")

        elif event.type in {'LEFTMOUSE', 'RET'} and event.value == 'RELEASE':
            context.area.header_text_set(None)
            return {'FINISHED'}
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            context.area.header_text_set(None)
            try: # Restore initial value on cancel
                curve_data[cvars.PROP_RADIUS_ARRAY][self.point_index] = self.initial_radius
                curve_data.update_tag()
                for window in context.window_manager.windows:
                    for area in window.screen.areas:
                        if area.type == 'VIEW_3D': area.tag_redraw()
            except Exception as e: print(f"Modal Cancel Error: {e}")
            return {'CANCELLED'}

        return {'RUNNING_MODAL'}


# --- Custom Gizmo Shape Definition ---
class PIECEGEN_GIZMO_radius_handle(bpy.types.Gizmo):
    """
    Gizmo shape for radius control, mimicking built-in radius tool.
    Draws a circle whose visual radius matches the point radius.
    Interaction scales radius based on mouse distance from center.
    Invokes PIECEGEN_OT_set_radius.
    """
    bl_idname = "piecegen.radius_gizmo_shape"

    # Store internal state for modal operation
    init_radius: float = 1.0 # Stores the radius value at the start of invoke
    init_mouse_pos_px: Vector = None # Stores initial mouse screen position (Vector)
    origin_px: Vector = None # Stores projected gizmo origin in screen pixels
    piecegen_point_index: int = None
    custom_drawing_shape = None
    current_screen_radius_for_select = None

    # --- Store raw geometry at class level ---
    _unit_disk_segments = 32  # Number of segments for the circle
    _unit_disk_verts, _unit_disk_faces, _disk_verts_flattened = generate_unit_disk_geometry(_unit_disk_segments)

    # --- Appearance ---
    def draw(self, context):
        """Draw the filled gizmo circle in the viewport."""
        # 1. Draw the filled disk using the current gizmo color and alpha
        # (these are self.color, self.alpha or self.color_highlight, self.alpha_highlight
        #  depending on the gizmo's state, handled by the system)
        if self.custom_drawing_shape:
            self.draw_custom_shape(
                shape=self.custom_drawing_shape,
                matrix=self.matrix_basis
            )

        # 2. Prepare to draw the orange outline
        # Store the current color and alpha so we can restore them later.
        # This ensures that if other drawing operations happened after this for the gizmo,
        # or for other gizmos, they use the correct original colors.
        original_color = self.color[:]  # Create a copy of the color tuple/list
        original_alpha = self.alpha

        # Define orange colors
        orange_color = (1.0, 0.5, 0.0)          # Standard orange
        highlight_orange_color = (1.0, 0.7, 0.2) # A slightly brighter/different orange for highlight

        # Set the color and alpha for the outline
        if self.is_highlight:
            self.color = highlight_orange_color
            self.alpha = self.alpha_highlight # Use the gizmo's highlight alpha for the outline too
        else:
            self.color = orange_color
            # You can choose the alpha for the outline:
            # Option A: Use the gizmo's current normal alpha
            self.alpha = original_alpha 
            # Option B: Use a fixed alpha for the outline, e.g., fully opaque
            # self.alpha = 1.0 

        # 3. Draw the circular outline
        # draw_preset_circle draws an unfilled circle.
        # It will use the self.color and self.alpha we just set.
        # The self.matrix_basis already provides the correct transformation
        # for screen-space constant size and orientation.
        # The default radius=1.0 for draw_preset_circle is correct as matrix_basis handles scale.
        self.draw_preset_circle(
            matrix=self.matrix_basis,
        )

        # 4. Restore the original color and alpha
        self.color = original_color
        self.alpha = original_alpha

    def draw_select(self, context, select_id):
        """Draw the filled gizmo circle for selection."""
        # The Gizmo system calls this with the 'select_id' to use
        # for rendering this gizmo into the selection buffer.
        self.draw_custom_shape(
            shape=self.custom_drawing_shape, # Pass the compiled shape object
            matrix=self.matrix_basis
        )

    # --- Interaction Setup ---
    def setup(self):
        """Configure gizmo properties for interaction."""
        # Set the property name the gizmo should modify on the target operator
        self.target_property = "value"
        # No grab_axis needed, interaction is based on 2D distance
        self.use_grab_axis = False
        # Adjust sensitivity if needed (might not be directly applicable for distance scaling)
        # self.scale_basis = 1.0 # Default
        if self.custom_drawing_shape is None: 
            self.custom_drawing_shape = self.new_custom_shape(
                type='TRIS', 
                verts=PIECEGEN_GIZMO_radius_handle._disk_verts_flattened
            )

    # --- Interaction Methods ---
    def invoke(self, context, event):
        """Initialize gizmo state on click."""
        # Get the initial value from the operator property (set by GizmoGroup setup)
        op_props = self.target_get_operator(PIECEGEN_OT_set_radius.bl_idname)
        if not op_props: return {'CANCELLED'} # Should not happen if setup worked
        self.init_radius = max(0.01, op_props.value) # Get initial radius, ensure positive

        # Store initial mouse screen position
        self.init_mouse_pos_px = Vector((event.mouse_region_x, event.mouse_region_y))

        # Project gizmo origin to screen space for distance calculation
        origin_world = self.matrix_basis.translation
        region = context.region
        rv3d = context.space_data.region_3d
        self.origin_px = view3d_utils.world_to_region_2d(region, rv3d, origin_world)

        if self.origin_px is None:
            print("Gizmo Invoke Warning: Could not project origin to screen.")
            return {'CANCELLED'} # Cannot calculate distance if origin projection fails

        # Push undo step before modal operation begins
        bpy.ops.ed.undo_push(message="Set Radius (Gizmo)")

        return {'RUNNING_MODAL'}

    def exit(self, context, cancel):
        """Cleanup on gizmo exit."""
        context.area.header_text_set(None)
        # Operator handles final state / undo

    def modal(self, context, event, tweak):
        """Handle gizmo interaction during drag based on screen distance."""

        if self.origin_px is None: # Should have been set in invoke
            return {'CANCELLED'}

        current_mouse_pos_px = Vector((event.mouse_region_x, event.mouse_region_y))

        # Initial mouse vector from projected origin
        init_vec_px = self.init_mouse_pos_px - self.origin_px
        init_dist_px = init_vec_px.length
        
        # Prevent division by zero or extreme sensitivity near the center
        # If initial click is very close to center, mouse must move further for scaling.
        if init_dist_px < 2.0: # Use a small pixel threshold
            init_dist_px = 2.0 # Effectively makes initial scale factor 1 until mouse moves further

        # Current mouse vector from projected origin
        current_vec_px = current_mouse_pos_px - self.origin_px
        current_dist_px = current_vec_px.length

        # Calculate scale factor based on distance change
        # If current_dist_px is also very small, scale_factor will be small.
        scale_factor = current_dist_px / init_dist_px

        # Calculate new value based on initial value and scale factor
        # self.init_value is the radius when the drag started.
        calculated_value = self.init_value * scale_factor

        # Apply min/max thresholds (replaces the old max(0.01, ...))
        new_value = max(MIN_RADIUS_VALUE, min(calculated_value, MAX_RADIUS_VALUE))

        # Apply tweak modifiers (optional, snapping might be useful)
        if 'SNAP' in tweak: # Snap to increments (e.g., 0.1 units)
            # Example: Snap to 2 decimal places if value is small, 1 if larger
            if new_value < 1.0:
                new_value = round(new_value * 100) / 100
            else:
                new_value = round(new_value * 10) / 10
        
        # Update the linked operator's 'value' property
        op_props = self.target_get_operator(PIECEGEN_OT_set_radius.bl_idname)
        if op_props:
            op_props.value = new_value
            # The operator's modal function will read this new value, apply it to the curve,
            # and handle UI updates like the header text.
        else:
            # Should not happen if invoke succeeded
            return {'CANCELLED'}

        # Check for standard exit keys (operator's modal will also handle these)
        if event.type in {'LEFTMOUSE', 'RET'} and event.value == 'RELEASE':
            # Operator's modal will return {'FINISHED'}
            pass 
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
            # Operator's modal will return {'CANCELLED'}
            pass

        return {'RUNNING_MODAL'} # Continue gizmo modal operation

    # --- Selection testing methods ---
    def test_select(self, context, location):
        """
        Test selection based on distance from the projected gizmo center
        in screen space, using the known TARGET_SCREEN_RADIUS_PX.
        'location' is a 2D vector of the mouse coordinates.
        """
        region = context.region
        rv3d = context.space_data.region_3d

        # Get the gizmo's 3D world origin from its current matrix_basis
        origin_world = self.matrix_basis.translation

        # Project the 3D world origin to 2D screen coordinates
        origin_px = view3d_utils.location_3d_to_region_2d(region, rv3d, origin_world)

        # 'location' is already in screen pixels (it's event.mouse_region_x, event.mouse_region_y)
        # Ensure it's a Vector for distance calculation
        mouse_loc_px = Vector(location)

        # Calculate the 2D screen-space distance between the mouse
        # and the projected center of the gizmo.
        dist_px = (mouse_loc_px - origin_px).length

        # Define a tolerance for easier selection
        select_tolerance_px = 5.0  # Adjust as needed (e.g., make it a small fraction of TARGET_SCREEN_RADIUS_PX or a fixed value)

        # Use the dynamically calculated screen radius for this gizmo instance.
        # This assumes 'current_screen_radius_for_select' is set on the instance
        # by the GizmoGroup during matrix calculation.
        # If not directly set, this logic would need to re-calculate it based on self.value
        # or the current data_radius from the curve, which is more complex here.
        # The cleaner way is for the group to store it on 'gz' when updating matrix_basis.
        effective_screen_radius = self.current_screen_radius_for_select

        # Perform hit test:
        # Check if the mouse click is within the circle's apparent screen radius.
        # This is for a "solid" circle selection (clicking anywhere inside).
        if dist_px < effective_screen_radius + select_tolerance_px:
            return self.select_id  # Hit! (self.select_id is set by the GizmoGroup system)

        # Optional: If you wanted an "annulus" selection (clicking on the ring itself):
        # ring_visual_thickness_px = 2.0 # Example: How thick the ring appears
        # if abs(dist_px - TARGET_SCREEN_RADIUS_PX) < (ring_visual_thickness_px / 2.0 + select_tolerance_px):
        #     return self.select_id # Hit!

        return -1  # Miss

    def select(self, context, event):
        self.use_draw_modal = True; return True

    def deselect(self, context):
        self.use_draw_modal = False; return True


# --- Gizmo Group Definition ---
class PIECEGEN_GGT_radius_control(bpy.types.GizmoGroup):
    """
    Gizmo Group that displays radius handles on selected Bezier points.
    Links gizmo interaction to the PIECEGEN_OT_set_radius operator.
    """
    # bl_idname = "piecegen.radius_gizmo_group"
    bl_label = "PieceGen Radius Control Gizmo"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'WINDOW'
    bl_options = {'3D', 'SHOW_MODAL_ALL', 'SELECT'}
    
    @classmethod
    def poll(cls, context):
        """Check if the associated WorkSpaceTool is active and context is valid."""
        active_tool = None
        try: active_tool = context.workspace.tools.from_space_view3d_mode(context.mode, create=False)
        except Exception: pass
        
        tool_ok = active_tool and active_tool.idname == PIECEGEN_TOOL_radius_edit.bl_idname
        if not tool_ok: return False
        
        mode_ok = context.mode == 'EDIT_CURVE'
        obj = context.object
        obj_ok = obj and obj.type == 'CURVE'
        data_ok = obj_ok and obj.data and obj.data.splines and len(obj.data.splines) > 0 and obj.data.splines[0].type == 'BEZIER'
        prop_ok = data_ok and cvars is not None and cvars.PROP_RADIUS_ARRAY in obj.data
        return mode_ok and prop_ok

    def calc_gizmo_matrices(self, context, pos_ws_co: Vector, radius_co: Vector):
        # Matrix for gizmo's world position
        mat_world_gz = Matrix.Translation(pos_ws_co)

        rv3d = context.space_data.region_3d # RegionView3D

        # --- Calculate Orientation Matrix (mat_rot_gz) ---
        # This matrix orients the gizmo's local XY plane to be parallel to the screen's XY plane.
        # The rotational part of the inverse view matrix aligns local axes with view axes.
        mat_rot_gz = rv3d.view_matrix.inverted().to_3x3().to_4x4()

        # --- Calculate target screen radius based on data_radius ---
        # This makes the visual size of the handle proportional to the data it represents.
        target_visual_radius_on_screen_px = PIXELS_PER_WORLD_UNIT * radius_co
        
        # Clamp the calculated visual screen radius to make the handle always usable
        target_visual_radius_on_screen_px = max(
            MIN_VISUAL_HANDLE_SIZE_PX,
            min(target_visual_radius_on_screen_px, MAX_VISUAL_HANDLE_SIZE_PX)
        )

        # --- Calculate Non-Uniform Scale for Screen-Space Circle ---
        # Default world radii if projection fails or rv3d is None.
        # These defaults will result in a small circle in world units.
        world_display_radius_x = 0.1
        world_display_radius_y = 0.1

        region = context.region # Viewport region
        gizmo_screen_origin = view3d_utils.location_3d_to_region_2d(region, rv3d, pos_ws_co)

        # Calculate world scale needed for the screen X-axis projection
        offset_screen_point_x = gizmo_screen_origin + Vector((target_visual_radius_on_screen_px, 0.0))
        offset_world_point_x = view3d_utils.region_2d_to_location_3d(region, rv3d, offset_screen_point_x, pos_ws_co)
        calculated_radius_x = (offset_world_point_x - pos_ws_co).length
        world_display_radius_x = calculated_radius_x

        # Calculate world scale needed for the screen Y-axis projection
        offset_screen_point_y = gizmo_screen_origin + Vector((0.0, target_visual_radius_on_screen_px))
        offset_world_point_y = view3d_utils.region_2d_to_location_3d(region, rv3d, offset_screen_point_y, pos_ws_co)
        calculated_radius_y = (offset_world_point_y - pos_ws_co).length
        world_display_radius_y = calculated_radius_y
    
        # --- Create the Scale Matrix ---
        # This matrix will scale the unit circle drawn by draw_preset_circle.
        # It scales local X by world_display_radius_x and local Y by world_display_radius_y.
        # The Z scale is less critical for a 2D circle; using an average or 1.0 is fine.
        # `draw_preset_circle` by default draws in the XY plane (normal along Z).
        mat_scale_gz = Matrix.Diagonal((
            world_display_radius_x,
            world_display_radius_y,
            (world_display_radius_x + world_display_radius_y) / 2.0, # Or 1.0, or min(sx,sy)
            1.0
        ))

        # The final matrix_basis for the gizmo instance
        return mat_world_gz @ mat_rot_gz @ mat_scale_gz, target_visual_radius_on_screen_px

    def realloc_from_curve(self, context):
        """Create gizmo instances for selected points and link to operator."""
        obj = context.object
        if not self.poll(context): self.gizmos.clear(); return

        curve_data = obj.data
        spline            = curve_data.splines[0]
        radius_array_prop = curve_data.get(cvars.PROP_RADIUS_ARRAY)

        # --- Reallocate Gizmos ---
        self.gizmos.clear() 
        mat_world = obj.matrix_world
        selected_point_indices = {i for i, bp in enumerate(spline.bezier_points) if bp.select_control_point}

        for i, bp in enumerate(spline.bezier_points):
            if i in selected_point_indices:
                gz = self.gizmos.new(PIECEGEN_GIZMO_radius_handle.bl_idname)
                gz.piecegen_point_index = i # Store the point index on the gizmo instance itself

                # --- Link Gizmo Interaction to Operator ---
                # This returns an object representing the operator's properties
                # that will be used when this gizmo instance (gz) is activated.
                op_props = gz.target_set_operator(PIECEGEN_OT_set_radius.bl_idname)
                if op_props:
                    # Set the 'point_index' property ON THE OPERATOR PROPERTIES OBJECT
                    op_props.point_index = i
                    # print(f"  Gizmo for point {i} linked to operator.") # Debug
                else:
                    print(f"  Error: Could not get operator properties for point {i}")
                    continue # Skip this gizmo if linking failed

                # --- Calculate Scaling ---
                current_radius = max(0.01, float(radius_array_prop[i]))
                # --- Set initial 'value' on the operator properties ---
                # This ensures the operator starts with the correct radius when invoked by this gizmo
                op_props.value = current_radius

                # --- Set Gizmo Matrix ---
                gz.matrix_basis, gz.current_screen_radius_for_select = self.calc_gizmo_matrices(
                    context, 
                    pos_ws_co=mat_world @ bp.co, 
                    radius_co=current_radius
                )

                # --- Configure Appearance ---'
                gz.color = 0.5, 0.4, 0.1; gz.alpha = 0.5
                gz.color_highlight = 1.0, 1.0, 0.5; gz.alpha_highlight = 1.0
                gz.use_draw_scale = False

                # --- Call Gizmo's Setup ---
                gz.setup() # Configures target_property, grab_axis etc.

    def update_from_curve(self, context):
        """Updates the matrix_basis of active gizmos before drawing."""
        obj = context.active_object
        
        curve_data = obj.data
        spline = curve_data.splines[0]
            
        radius_array_prop = curve_data.get(cvars.PROP_RADIUS_ARRAY)

        mat_world = obj.matrix_world

        for gz in self.gizmos:
            if not hasattr(gz, 'piecegen_point_index') or gz.piecegen_point_index < 0:
                continue 

            point_idx = gz.piecegen_point_index
            if point_idx >= len(spline.bezier_points): continue # Point index out of bounds

            bp = spline.bezier_points[point_idx]
            # --- Recalculate Orientation and Scale ---
            current_radius_val = 1.0
            try:
                if point_idx < len(list(radius_array_prop)): # Check bounds for radius_array_prop
                    current_radius_val = max(0.01, float(radius_array_prop[point_idx]))
            except IndexError: pass
            
            # Update gizmos transform
            gz.matrix_basis, gz.current_screen_radius_for_select = self.calc_gizmo_matrices(
                context, 
                pos_ws_co=mat_world @ bp.co, 
                radius_co=current_radius_val
            )

            # --- Call Gizmo's Setup ---
            gz.setup() # Configures target_property, grab_axis etc.

    def setup(self, context):
        self.realloc_from_curve(context)

    def refresh(self, context):
        self.realloc_from_curve(context)

    def draw_prepare(self, context):    
        self.update_from_curve(context)

    def invoke_prepare(self, context, event):
        print(f"GizmoGroup {self.bl_label} invoke_prepareD with event {event.type}")
        # Now, iterate self.gizmos and manually do the test_select / select / invoke sequence
        for gz in self.gizmos:
            if isinstance(gz, PIECEGEN_GIZMO_radius_handle):
                mouse_region_loc = (event.mouse_region_x, event.mouse_region_y)
                hit_id = gz.test_select(context, mouse_region_loc)
                if hit_id != -1 and hit_id == gz.select_id:
                    if gz.select(context, event):
                        return gz.invoke(context, event) # Let the individual gizmo run its modal
        return {'PASS_THROUGH'}
    


# --- Custom Workspace Tool ---
# https://github.com/blender/blender/blob/main/scripts/templates_py/ui_tool_simple.py
# https://b3d.interplanety.org/en/creating-custom-tool-in-blender/
class PIECEGEN_TOOL_radius_edit(bpy.types.WorkSpaceTool):
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

    # bl_widget = "PIECEGEN_GGT_radius_control"
    # bl_widget_properties = [
    #     # ("radius", 75.0),
    #     # ("backdrop_fill_alpha", 0.0),
    # ]
    bl_keymap = (
        # This entry tells Blender to pass LEFTMOUSE PRESS events to the active gizmo group.
        # The gizmo group will then determine if any of its gizmos (like your radius handle)
        # are hit (via test_select) and then call their invoke/modal methods.
        (op_custom_ui.PIECEGEN_OT_modal_point_scale.bl_idname,
            {"type": 'S', "value": 'PRESS', 'alt': True}, # User presses alt + 'S'
            {'properties': []}
        ),
        # ("transform.scale",     {'type':'S', 'value':'PRESS'}),

        # Keep standard selection & transform hotkeys
        ("view3d.select",
            {'type': 'LEFTMOUSE', 'value': 'CLICK'}, 
            { 'properties': [
                    ("deselect_all",True), # Deselect everything else on single click
                    ("toggle", False), # Don't toggle selection
                ]
            },
        ),
        ("view3d.select_box",   
            {'type': 'LEFTMOUSE', 'value': 'CLICK_DRAG'}, 
            {'properties': [('wait_for_input',False), ('mode','SET')]}
        ), 
        ("transform.rotate", 
            {'type':'R', 'value':'PRESS'}, 
            {'properties': []}
        ),
        ("transform.transform", 
            {'type':'W', 'value':'PRESS'}
            , {'properties': []}
        ),
        # (PIECEGEN_OT_set_radius.bl_idname, {'type': 'S', 'value': 'PRESS', 'alt': True}),
    )

    # 8. Optional: Draw settings in the header when active
    def draw_settings(context, layout, tool):
        layout.label(text="PieceGen Tool Active")




# --- Registration Helper ---
gizmo_classes = (
    op_custom_ui.PIECEGEN_OT_modal_point_scale, 
    # PIECEGEN_OT_set_radius, 
    # PIECEGEN_GIZMO_radius_handle,
    # PIECEGEN_GGT_radius_control,
)

def register():
    for cls in gizmo_classes:
        try: bpy.utils.register_class(cls)
        except ValueError: pass

    bpy.utils.register_tool(PIECEGEN_TOOL_radius_edit, after={'builtin.transform'}, separator=True, group=True)

def unregister():
    bpy.utils.unregister_tool(PIECEGEN_TOOL_radius_edit)

    for cls in reversed(gizmo_classes):
        if hasattr(bpy.types, cls.__name__):
            try: bpy.utils.unregister_class(cls)
            except (RuntimeError, Exception) as e: print(f"Warning: Could not unregister class {cls.__name__}: {e}")
