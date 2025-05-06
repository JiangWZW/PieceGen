# <pep8 compliant>
# -*- coding: utf-8 -*-
# File: op_generate.py
# Contains OBJECT_OT_generate_cylinder_with_curve operator

# --- Standard Imports ---
import bpy
import bmesh
from mathutils import Vector, Matrix

# --- Local Imports ---
from . import common_vars as cvars
from .bezier import ensure_radius_array

class OBJECT_OT_generate_cylinder_with_curve(bpy.types.Operator):
    """Creates/Recreates a cylinder mesh and a Bézier curve based on Scene settings."""
    bl_idname = "object.generate_cylinder_with_curve"
    bl_label = "1. Create/Recreate Objects"
    bl_description = "Generates the base cylinder and curve objects using panel settings"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        # Operator can run only in Object mode
        return context.mode == 'OBJECT'

    def execute(self, context):
        scene = context.scene
        # Access addon properties stored on the scene
        # These properties are defined in __init__.py and attached to the Scene type
        if not hasattr(scene, 'ccdg_props'):
             self.report({'ERROR'}, "Addon properties not found on scene. Please ensure addon is enabled.")
             return {'CANCELLED'}
        props = scene.ccdg_props
        cursor_loc = scene.cursor.location.copy() # Remember user's cursor location

        # Read properties from the panel settings stored in the PropertyGroup
        radius = props.cap_radius
        height = props.height
        fill_type = props.cap_fill_type
        num_height_segs = props.num_height_segs
        num_curve_pts = props.num_cpts_curve

        # Remember active object and mode to restore later if possible
        prev_active = context.view_layer.objects.active
        prev_mode = context.object.mode if context.object else 'OBJECT'
        if prev_mode != 'OBJECT':
            try:
                bpy.ops.object.mode_set(mode='OBJECT')
            except RuntimeError as e:
                # Handle cases where mode set might fail (e.g., no active object)
                print(f"Warning: Could not switch to Object mode: {e}")
                # Continue if possible, but be aware context might be unexpected

        # --- Create Cylinder ---
        try:
            # Create cylinder at world origin first for easier origin setting later
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=props.num_cap_verts,
                radius=radius,
                depth=height, # Depth corresponds to height along Z
                end_fill_type=fill_type,
                location=(0,0,0), # Create at world origin
                scale=(1,1,1)
            )
        except Exception as e:
             # Attempt to restore previous state if creation fails
             if prev_active: context.view_layer.objects.active = prev_active
             if prev_mode != 'OBJECT':
                 try: bpy.ops.object.mode_set(mode=prev_mode)
                 except RuntimeError: pass # Ignore if context changed
             self.report({'ERROR'}, f"Cylinder creation failed: {e}")
             return {'CANCELLED'}

        cyl_obj = context.active_object # The newly created cylinder is now active
        if not cyl_obj or cyl_obj.type != 'MESH':
            self.report({'ERROR'}, "Failed to get created cylinder object.")
            # Attempt to restore previous state
            if prev_active: context.view_layer.objects.active = prev_active
            if prev_mode != 'OBJECT':
                 try: bpy.ops.object.mode_set(mode=prev_mode)
                 except RuntimeError: pass
            return {'CANCELLED'}

        # Assign meaningful names
        cyl_obj.name = "DeformCylinder"
        mesh_data = cyl_obj.data
        mesh_data.name = "DeformCylinderMesh"

        # --- Subdivide Height using BMesh ---
        # (Important: Do this *before* setting the origin, while geometry is centered)
        if num_height_segs > 1:
            bm = None # Initialize bmesh variable
            try:
                bm = bmesh.new() # Create a new BMesh
                bm.from_mesh(mesh_data) # Load mesh data into BMesh
                bm.verts.ensure_lookup_table() # Ensure fast vertex access
                bm.edges.ensure_lookup_table() # Ensure fast edge access

                # Collect vertical edges for splitting (more robust check)
                vertical_edges = []
                # Use relative tolerances based on dimensions for robustness
                height_epsilon = height * 0.01 # Tolerance for height check (1% of height)
                xy_epsilon = radius * 0.01   # Tolerance for XY alignment check (1% of radius)

                for edge in bm.edges:
                    v1_co = edge.verts[0].co
                    v2_co = edge.verts[1].co
                    delta = v1_co - v2_co
                    # Check if edge length is close to cylinder height
                    # AND check if X and Y differences are very small (indicating vertical edge)
                    if abs(delta.length - height) < height_epsilon and \
                       abs(delta.x) < xy_epsilon and \
                       abs(delta.y) < xy_epsilon:
                        vertical_edges.append(edge)

                if vertical_edges:
                    # Perform subdivision on the collected vertical edges
                    bmesh.ops.subdivide_edges(bm,
                                              edges=vertical_edges,
                                              cuts=num_height_segs - 1, # cuts = segments - 1
                                              use_grid_fill=False) # Simple subdivision
                    # Write the modified BMesh data back to the mesh object
                    bm.to_mesh(mesh_data)
                    mesh_data.update() # Update mesh data state for Blender
                else:
                    # Report if no suitable edges were found (e.g., unexpected geometry)
                    self.report({'WARNING'}, "No suitable vertical edges found for subdivision.")
            except Exception as e:
                # Report any errors during BMesh operations
                self.report({'WARNING'}, f"BMesh subdivision failed: {e}")
            finally:
                # Ensure BMesh is freed to prevent memory leaks
                if bm:
                    bm.free()

        # --- Set Origin to Base Center ---
        # Calculate the desired origin point in current local space (center of the base)
        base_center_local = Vector((0.0, 0.0, -height / 2.0))
        # Transform this local point to world space using the object's current matrix_world
        # Since object was created at origin, matrix_world is identity initially
        base_center_world = cyl_obj.matrix_world @ base_center_local

        # Store current object world location before changing the origin
        original_world_location = cyl_obj.location.copy()

        # Select only the cylinder object
        bpy.ops.object.select_all(action='DESELECT')
        cyl_obj.select_set(True)
        context.view_layer.objects.active = cyl_obj

        # Temporarily move the 3D cursor to the target origin point
        context.scene.cursor.location = base_center_world
        # Set the object's origin to the 3D cursor location
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='MEDIAN')
        # Move the object so its new origin is back at the original world location
        # This ensures the object visually stays in place after the origin change
        cyl_obj.location = original_world_location
        # This ensures that Blender updates the internal state 
        # and the cyl_obj.matrix_world accurately reflects the final position 
        # and origin change before we use it for the curve.
        context.view_layer.update() 


        # --- Create Bézier Curve ---
        curve_data = bpy.data.curves.new('BendCurveData', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 12 # Keep higher resolution for smooth appearance
        spline = curve_data.splines.new(type='BEZIER')
        # Add the specified number of points (need num_curve_pts - 1 additional points)
        spline.bezier_points.add(count=num_curve_pts - 1)

        # Calculate segment length for placing points along the original height
        # Ensure division by zero is avoided if num_curve_pts is 1 (though poll prevents this)
        seg_len = height / (num_curve_pts - 1) if num_curve_pts > 1 else height

        # Configure each Bezier point
        for i, bp in enumerate(spline.bezier_points):
            # Place points along local Z axis relative to object origin (which is now base)
            z = i * seg_len
            bp.co = Vector((0.0, 0.0, z))
            # Calculate handle offset relative to segment length for initial straight shape
            h_offset = seg_len / 3.0 # Common heuristic for handles
            bp.handle_left = Vector((0.0, 0.0, z - h_offset))
            bp.handle_right = Vector((0.0, 0.0, z + h_offset))
            # Use AUTO handles for potentially smoother initial curve editing experience
            bp.handle_left_type = 'AUTO'
            bp.handle_right_type = 'AUTO'
            bp.tilt = 0.0 # Default tilt to zero

        # Create the curve object
        curve_obj = bpy.data.objects.new('BendCurveObject', curve_data)
        # Place curve object at the same location and orientation as the cylinder's origin
        # This ensures they start aligned
        curve_obj.matrix_world = cyl_obj.matrix_world.copy()
        # Link the new curve object to the current scene collection
        context.collection.objects.link(curve_obj)

        # --- Initialize Radius Array using imported function ---
        if not ensure_radius_array(curve_obj.data):
            # Function reported an error
            self.report({'WARNING'}, f"Could not initialize/verify radius array on '{curve_obj.name}'. Gizmos might not work.")
            # Don't cancel the whole operator, just warn.

        # --- Final Selection & Cleanup ---
        bpy.ops.object.select_all(action='DESELECT') # Deselect all objects first
        context.view_layer.objects.active = cyl_obj # Make cylinder the active object
        cyl_obj.select_set(True) # Select cylinder
        curve_obj.select_set(True) # Select curve as well
        context.scene.cursor.location = cursor_loc # Restore original cursor location

        # Restore original mode if necessary (handle potential context changes robustly)
        # if prev_mode != 'OBJECT' and context.object and context.object.mode != prev_mode:
        #    try:
        #        bpy.ops.object.mode_set(mode=prev_mode)
        #    except RuntimeError as e:
        #        print(f"Warning: Could not restore previous mode '{prev_mode}': {e}")

        self.report({'INFO'}, f"Created '{cyl_obj.name}' and '{curve_obj.name}'.")
        return {'FINISHED'}
