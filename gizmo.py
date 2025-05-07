# <pep8 compliant>
# -*- coding: utf-8 -*-
# File: gizmos.py
# Contains Gizmo definitions for PieceGen addon using target_set_handler
# DEBUGGING: Focus on why poll() is not called.

import bpy
import math
from mathutils import Vector, Matrix, Euler
from bpy_extras import view3d_utils # Needed for mouse projection

# Import shared variables (needed for property key)
try:
    from . import common_vars as cvars
    # Import the bezier module if needed for tangent/normal calculations
    from . import bezier
except ImportError:
    # Fallback if run directly (shouldn't happen in addon context)
    print("Error importing common_vars or bezier in gizmos.py")
    cvars = None # Define dummy if needed
    bezier = None




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
    init_value: float = 1.0 # Stores the radius value at the start of invoke
    init_mouse_pos_px: Vector = None # Stores initial mouse screen position (Vector)
    origin_px: Vector = None # Stores projected gizmo origin in screen pixels

    # --- Appearance ---
    def draw(self, context):
        """Draw the gizmo shape in the viewport."""
        # Draw a circle with base radius 1.0, oriented in its XY plane.
        # The actual size is controlled by the matrix_basis scale.
        # The GizmoGroup setup aligns the gizmo's Z with curve tangent,
        # so the circle lies in the Normal/Binormal plane.
        self.draw_preset_circle(
            matrix=self.matrix_basis # Matrix.Identity(4) # No extra rotation needed relative to gizmo's matrix_basis
        )

    def draw_select(self, context, select_id):
        """Draw the selection highlight for the gizmo."""
        # Use a slightly thicker line or slightly larger radius for selection
        # For simplicity, just draw the same circle with the select_id
        self.draw_preset_circle(
            matrix=Matrix.Identity(4),
            select_id=select_id
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

    # --- Interaction Methods ---
    def invoke(self, context, event):
        """Initialize gizmo state on click."""
        # Get the initial value from the operator property (set by GizmoGroup setup)
        op_props = self.target_get_operator(PIECEGEN_OT_set_radius.bl_idname)
        if not op_props: return {'CANCELLED'} # Should not happen if setup worked
        self.init_value = max(0.01, op_props.value) # Get initial radius, ensure positive

        # Store initial mouse screen position
        self.init_mouse_pos_px = Vector((event.mouse_region_x, event.mouse_region_y))

        # Project gizmo origin to screen space for distance calculation
        origin_world = self.matrix_basis.translation
        region = context.region
        rv3d = context.region_data
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

        # Current mouse position
        current_mouse_pos_px = Vector((event.mouse_region_x, event.mouse_region_y))

        # Initial mouse vector from projected origin
        init_vec_px = self.init_mouse_pos_px - self.origin_px
        init_dist_px = init_vec_px.length
        # Prevent division by zero or extreme sensitivity near the center
        if init_dist_px < 2.0: # Use a small pixel threshold
            init_dist_px = 2.0

        # Current mouse vector from projected origin
        current_vec_px = current_mouse_pos_px - self.origin_px
        current_dist_px = current_vec_px.length

        # Calculate scale factor based on distance change
        scale_factor = current_dist_px / init_dist_px

        # Calculate new value based on initial value and scale factor
        new_value = max(0.01, self.init_value * scale_factor) # Ensure positive

        # Apply tweak modifiers (optional, snapping might be useful)
        # if 'PRECISE' in tweak: # Precise mode might feel odd here
        if 'SNAP' in tweak: # Snap to increments (e.g., 0.1 units)
             new_value = round(new_value * 10) / 10

        # --- Call the Operator ---
        # The gizmo system automatically updates the 'value' property on the linked operator
        # based on internal calculations derived from mouse movement and setup (target_property).
        # For distance-based scaling, we manually set the operator's value.
        op_props = self.target_get_operator(PIECEGEN_OT_set_radius.bl_idname)
        if op_props:
            op_props.value = new_value
            # The operator's modal function will read this new value and apply it.
        else:
            # Should not happen if invoke succeeded
             return {'CANCELLED'}

        # No need to call self.target_set_value here if operator handles it

        # Modal exit conditions (let operator handle finish/cancel logic)
        # The operator's modal will return FINISHED or CANCELLED
        # We just keep running until the operator finishes.
        # However, we need to check if the operator is *still* running,
        # as it might finish/cancel based on its own event handling.
        # This check is complex; simpler to let operator manage exit.

        # Check for standard exit keys locally in gizmo as well for responsiveness
        if event.type in {'LEFTMOUSE', 'RET'} and event.value == 'RELEASE':
             # Let operator finalize state in its modal return
             pass
        elif event.type in {'RIGHTMOUSE', 'ESC'}:
             # Let operator cancel and restore state in its modal return
             pass


        return {'RUNNING_MODAL'} # Continue gizmo modal

    # --- Selection testing methods ---
    def test_select(self, context, location):
        """Test selection based on distance from center in screen space."""
        if not self.is_select_test_enabled(context):
            return -1

        # Project origin to screen space
        origin_world = self.matrix_basis.translation
        region = context.region
        rv3d = context.region_data
        origin_px = view3d_utils.world_to_region_2d(region, rv3d, origin_world)

        if origin_px is None: return -1 # Cannot test if origin not visible

        # Calculate screen distance from mouse location to projected origin
        mouse_loc_px = Vector(location)
        dist_px = (mouse_loc_px - origin_px).length

        # Determine the visual radius of the gizmo circle in pixels
        # Get world radius (from matrix scale) and project a point on the circumference
        world_radius = self.matrix_basis.col[0].length # Approx radius from scale
        # Use a point along the gizmo's local X axis scaled by world_radius
        point_on_circle_world = origin_world + self.matrix_basis.col[0].normalized() * world_radius
        point_on_circle_px = view3d_utils.world_to_region_2d(region, rv3d, point_on_circle_world)

        if point_on_circle_px is None: return -1 # Cannot determine pixel radius

        visual_radius_px = (point_on_circle_px - origin_px).length

        # Add a small tolerance for easier selection
        select_tolerance_px = 5.0
        # Check if mouse distance is close to the visual radius
        if abs(dist_px - visual_radius_px) < select_tolerance_px:
            return self.select_id # Hit!

        return -1 # Miss

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
    bl_options = {'3D', 'SHOW_MODAL_ALL'}

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

    def calc_curve_pt_matrix(i, num_points, spline, bp, mat_world, mat_world_normal): 
        t_param = i / max(1, num_points - 1) if num_points > 1 else 0.0
        tangent_local = bezier.evaluate_spline(spline, t_param, order=1)
        if tangent_local.length_squared < 1e-9:
                if i < num_points - 1: tangent_local = spline.bezier_points[i+1].co - bp.co
                elif i > 0: tangent_local = bp.co - spline.bezier_points[i-1].co
                else: tangent_local = bp.handle_right - bp.co
        if tangent_local.length_squared > 1e-9: tangent_local.normalize()
        else: tangent_local = Vector((0,0,1))
        up = Vector((0,0,1))
        if abs(tangent_local.dot(up)) > 0.999: up = Vector((0,1,0))
        binormal_local = tangent_local.cross(up)
        if binormal_local.length_squared < 1e-9: up = Vector((1,0,0)); binormal_local = tangent_local.cross(up)
        if binormal_local.length_squared > 1e-9:
                binormal_local.normalize(); normal_local = binormal_local.cross(tangent_local).normalized()
        else: normal_local = Vector((0,1,0)); binormal_local = Vector((1,0,0))
        tangent_world = (mat_world.to_3x3() @ tangent_local).normalized()
        normal_world = (mat_world_normal @ normal_local).normalized()
        binormal_world = (mat_world_normal @ binormal_local).normalized()
        if tangent_world.length > 1e-6 and normal_world.length > 1e-6:
            binormal_world = tangent_world.cross(normal_world).normalized()
            normal_world = binormal_world.cross(tangent_world).normalized()
        else: tangent_world=Vector((0,0,1)); normal_world=Vector((0,1,0)); binormal_world=Vector((1,0,0))
        mat_rot = Matrix((binormal_world, normal_world, tangent_world)).transposed()

        return mat_rot

    def setup(self, context):
        """Create gizmo instances for selected points and link to operator."""
        obj = context.object
        if not self.poll(context): self.gizmos.clear(); return

        curve_data = obj.data
        spline = curve_data.splines[0]
        radius_array_prop = curve_data.get(cvars.PROP_RADIUS_ARRAY)
        num_points = len(spline.bezier_points)

        if radius_array_prop is None: self.gizmos.clear(); return
        try: array_len = len(list(radius_array_prop))
        except TypeError: self.gizmos.clear(); return
        if array_len != num_points: self.gizmos.clear(); return

        mat_world = obj.matrix_world
        try: mat_world_normal = mat_world.inverted_safe().transposed().to_3x3()
        except ValueError: mat_world_normal = Matrix.Identity(3)

        selected_point_indices = {i for i, bp in enumerate(spline.bezier_points) if bp.select_control_point}
        self.gizmos.clear()

        for i, bp in enumerate(spline.bezier_points):
            if i in selected_point_indices:
                gz = self.gizmos.new(PIECEGEN_GIZMO_radius_handle.bl_idname)

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

                # --- Calculate Orientation Frame ---
                t_param = i / max(1, num_points - 1) if num_points > 1 else 0.0
                tangent_local = bezier.evaluate_spline(spline, t_param, order=1)
                if tangent_local.length_squared < 1e-9:
                        if i < num_points - 1: tangent_local = spline.bezier_points[i+1].co - bp.co
                        elif i > 0: tangent_local = bp.co - spline.bezier_points[i-1].co
                        else: tangent_local = bp.handle_right - bp.co
                if tangent_local.length_squared > 1e-9: tangent_local.normalize()
                else: tangent_local = Vector((0,0,1))
                up = Vector((0,0,1))
                if abs(tangent_local.dot(up)) > 0.999: up = Vector((0,1,0))
                binormal_local = tangent_local.cross(up)
                if binormal_local.length_squared < 1e-9: up = Vector((1,0,0)); binormal_local = tangent_local.cross(up)
                if binormal_local.length_squared > 1e-9:
                        binormal_local.normalize(); normal_local = binormal_local.cross(tangent_local).normalized()
                else: normal_local = Vector((0,1,0)); binormal_local = Vector((1,0,0))
                tangent_world = (mat_world.to_3x3() @ tangent_local).normalized()
                normal_world = (mat_world_normal @ normal_local).normalized()
                binormal_world = (mat_world_normal @ binormal_local).normalized()
                if tangent_world.length > 1e-6 and normal_world.length > 1e-6:
                    binormal_world = tangent_world.cross(normal_world).normalized()
                    normal_world = binormal_world.cross(tangent_world).normalized()
                else: tangent_world=Vector((0,0,1)); normal_world=Vector((0,1,0)); binormal_world=Vector((1,0,0))
                mat_rot = Matrix((binormal_world, normal_world, tangent_world)).transposed()

                # --- Set Gizmo Matrix ---
                try: current_radius = max(0.01, float(radius_array_prop[i]))
                except IndexError: current_radius = 1.0

                # --- Set initial 'value' on the operator properties ---
                # This ensures the operator starts with the correct radius when invoked by this gizmo
                if op_props:
                    op_props.value = current_radius

                world_co = mat_world @ bp.co
                mat_scale = Matrix.Scale(current_radius, 4)
                gz.matrix_basis = Matrix.Translation(world_co) @ mat_rot.to_4x4() @ mat_scale

                # --- Configure Appearance ---'
                gz.color = 0.8, 0.5, 0.2; gz.alpha = 0.7
                gz.color_highlight = 1.0, 1.0, 0.5; gz.alpha_highlight = 1.0
                gz.use_draw_scale = False

                # --- Call Gizmo's Setup ---
                gz.setup() # Configures target_property, grab_axis etc.

    def refresh(self, context):
        self.setup(context)



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

    bl_widget = "PIECEGEN_GGT_radius_control"
    bl_widget_properties = [
        # ("radius", 75.0),
        # ("backdrop_fill_alpha", 0.0),
    ]
    bl_keymap = (
        # Keep standard transform hotkeys
        ("transform.transform", 
            {'type':'H', 'value':'PRESS'}
            , {'properties': []}
        ),
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
        # ("transform.rotate",    {'type':'R', 'value':'PRESS'}),
        # ("transform.scale",     {'type':'S', 'value':'PRESS'}),
        # Example: Alt+S for radius adjustment (like built-in)
        # (PIECEGEN_OT_set_radius.bl_idname, {'type': 'S', 'value': 'PRESS', 'alt': True}),
    )

    # 8. Optional: Draw settings in the header when active
    def draw_settings(context, layout, tool):
        layout.label(text="PieceGen Tool Active")


# --- Registration Helper ---
gizmo_classes = (
    PIECEGEN_OT_set_radius, 
    PIECEGEN_GIZMO_radius_handle,
    PIECEGEN_GGT_radius_control,
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
