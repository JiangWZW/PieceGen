# <pep8 compliant>
# -*- coding: utf-8 -*-

# Main Addon File: cylinder_curve_deform/__init__.py

bl_info = {
    "name": "Cylinder Curve Deform (Python Module)",
    "author": "AI Assistant (Gemini) & User - Simplified", # Simplified version
    "version": (1, 20, 0), # Incremented for simplification refactor
    "blender": (3, 0, 0),
    "location": "View3D > Sidebar (N Panel) > Create Tab",
    "description": "Generates cylinder/curve, applies Python curve deform with real-time updates.",
    "warning": "PYTHON-BASED REALTIME DEFORM CAN BE SLOW FOR DENSE MESHES!",
    "doc_url": "",
    "category": "Add Mesh",
}

# --- Standard Imports ---
import bpy
import bmesh
import math
import mathutils
import base64
import zlib
from mathutils import Vector, Matrix

# --- Optional Imports ---
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False
    # Warning printed during registration

# --- Constants ---
# Custom Property Keys (used on the mesh object being deformed)
PROP_ENABLED = "ccdg_enabled"             # (Boolean) Is realtime deform active?
PROP_CURVE_NAME = "ccdg_curve_name"       # (String) Name of the guide curve object
PROP_ORIG_VERTS = "ccdg_original_verts_b64" # (String) Packed original vertex data
PROP_HEIGHT = "ccdg_original_height"      # (Float) Original cylinder height (renamed from PROP_DEPTH)

# --- Global State ---
# Stores names of mesh objects being actively deformed by the handler.
MONITORED_MESH_OBJECTS = set()

# --- Scene Properties ---
class CCDG_Properties(bpy.types.PropertyGroup):
    """Stores settings for the Addon Panel, attached to the Scene context."""
    num_cap_verts: bpy.props.IntProperty(
        name="Cap Vertices", description="Num vertices in cylinder cap ring",
        default=32, min=3, max=256)
    cap_radius: bpy.props.FloatProperty(
        name="Radius", description="Cylinder radius",
        default=1.0, min=0.01, unit='LENGTH', precision=3)
    height: bpy.props.FloatProperty(
        name="Height", description="Cylinder height (local Z)",
        default=2.0, min=0.01, unit='LENGTH', precision=3)
    num_height_segs: bpy.props.IntProperty(
        name="Height Segments", description="Num subdivisions along cylinder height",
        default=10, min=1, max=200)
    num_cpts_curve: bpy.props.IntProperty(
        name="Curve Points", description="Num control points for initial Bézier curve",
        default=4, min=2, max=32)
    cap_fill_type: bpy.props.EnumProperty(
        name="Cap Fill Type",
        items=[('NGON', "N-Gon", "N-Gon"),
               ('TRIANGLE_FAN', "Tri Fan", "Tri Fan"),
               ('NOTHING', "Nothing", "Nothing")],
        default='NGON', description="How to fill cylinder caps")

# --- Helper Functions ---

# --- Helper: Bezier Curve Evaluation ---
def evaluate_bezier_spline_with_tilt(spline: bpy.types.Spline, t: float):
    """
    Evaluates position, tangent, and tilt-aware normal of a Bezier spline
    at parameter t (0-1, representing fraction of total length).

    This function manually calculates the Bezier curve properties using
    standard mathematical formulas (Bernstein polynomials for position,
    its derivative for tangent) and incorporates Blender's 'Tilt' property
    for calculating the curve's normal vector at point t.

    Args:
        spline: The bpy.types.Spline object (must be BEZIER type).
        t: Parameter along the curve (0.0 to 1.0).

    Returns:
        tuple: (position_local, tangent_local, normal_local) Vectors in curve's local space.
               Returns default vectors if spline is invalid.
    """
    # --- Input Validation ---
    if not spline or not spline.bezier_points or spline.type != 'BEZIER':
        print("Warning: evaluate_bezier_spline called with invalid spline.")
        return Vector((0,0,0)), Vector((0,0,1)), Vector((0,1,0)) # Default return

    num_points = len(spline.bezier_points)
    num_segments = num_points - 1 # Number of curves between control points

    # --- Handle Edge Case: Single Point Spline ---
    # If there's only one control point, we can't evaluate along a segment.
    # We estimate properties based on the single point and its handles.
    if num_segments <= 0:
        pt = spline.bezier_points[0]
        position_local = pt.co.copy()
        # Estimate tangent from the right handle (direction leaving the point)
        tangent_local = (pt.handle_right - pt.co)
        if tangent_local.length_squared > 1e-12:
            tangent_local.normalize()
        else:
            tangent_local = Vector((0,0,1)) # Default if handle is coincident

        # Estimate a base normal perpendicular to the tangent
        normal_local_initial = Vector((0.0, 1.0, 0.0)) # Start with world Y
        if abs(tangent_local.dot(normal_local_initial)) > 0.9999: # Check alignment
             normal_local_initial = Vector((1.0, 0.0, 0.0)) # Use world X if aligned with Y
        # Ensure perpendicularity using double cross product
        normal_local_initial = tangent_local.cross(normal_local_initial).cross(tangent_local).normalized()

        # Apply the single point's tilt value
        rotation_matrix = Matrix.Rotation(pt.tilt, 3, tangent_local) # Rotation around tangent
        normal_local = rotation_matrix @ normal_local_initial # Rotate the calculated normal
        return position_local, tangent_local, normal_local

    # --- Determine Target Segment and Local Parameter 'local_t' ---
    # Clamp input 't' to the valid range [0, 1]
    t_clamped = max(0.0, min(1.0, t))
    # Calculate which segment 't' falls into (0-based index)
    segment_float = t_clamped * num_segments
    segment_index = min(math.floor(segment_float), num_segments - 1) # Clamp index
    # Calculate the parameter within that specific segment (range 0-1)
    local_t = segment_float - segment_index
    # Handle floating point precision issues at the very end (t=1.0)
    if local_t < 1e-6 and t_clamped == 1.0:
        local_t = 1.0
        segment_index = num_segments - 1 # Ensure we are on the last segment

    # --- Get Control Points, Handles, and Tilt for the Target Segment ---
    start_point = spline.bezier_points[segment_index]
    end_point = spline.bezier_points[segment_index + 1]

    # Coordinates and handles defining the cubic Bezier curve segment:
    p0 = start_point.co             # Start point coordinate (P0)
    h0 = start_point.handle_right   # Handle controlling curve *leaving* P0 (H0)
    tilt0 = start_point.tilt        # Tilt value at P0

    p1 = end_point.co               # End point coordinate (P1)
    h1 = end_point.handle_left      # Handle controlling curve *entering* P1 (H1)
    tilt1 = end_point.tilt          # Tilt value at P1

    # --- Calculate Position using Bernstein Polynomials ---
    # The cubic Bezier formula: B(t) = (1-t)^3*P0 + 3*(1-t)^2*t*H0 + 3*(1-t)*t^2*H1 + t^3*P1
    # Pre-calculate powers of t and (1-t) for the formula
    omt = 1.0 - local_t  # (1-t)
    omt2 = omt * omt     # (1-t)^2
    omt3 = omt2 * omt    # (1-t)^3
    lt2 = local_t * local_t # t^2
    lt3 = lt2 * local_t   # t^3

    # Calculate position by summing the weighted control points/handles
    position_local = (omt3 * p0) + (3.0 * omt2 * local_t * h0) + (3.0 * omt * lt2 * h1) + (lt3 * p1)

    # --- Calculate Tangent (Derivative of Bernstein Polynomials) ---
    # The derivative B'(t) gives the direction (tangent) of the curve:
    # B'(t) = 3*(1-t)^2*(H0-P0) + 6*(1-t)*t*(H1-H0) + 3*t^2*(P1-H1)
    tangent_local = (3.0 * omt2 * (h0 - p0)) + \
                    (6.0 * omt * local_t * (h1 - h0)) + \
                    (3.0 * lt2 * (p1 - h1))

    # Normalize the tangent vector to get a unit direction vector
    # Handle potential zero-length tangents (if points/handles coincide)
    if tangent_local.length_squared > 1e-12: # Use length_squared for efficiency
        tangent_local.normalize()
    else:
        # Fallback 1: Use the direction between the segment's start and end points
        tangent_local = (p1 - p0)
        if tangent_local.length_squared > 1e-12:
            tangent_local.normalize()
        else:
            # Fallback 2: Use world Z-axis if start/end points are also coincident
            tangent_local = Vector((0, 0, 1))

    # --- Calculate Tilt-Aware Normal ---
    # This determines the "up" direction of the curve at point 't', respecting user-defined twist.

    # 1. Interpolate Tilt value linearly between the segment's endpoints.
    #    Formula: a * (1-t) + b * t
    interpolated_tilt = tilt0 * (1.0 - local_t) + tilt1 * local_t

    # 2. Calculate an Initial Reference Normal vector.
    #    This normal must be perpendicular to the tangent vector.
    #    We establish a reliable perpendicular direction before applying tilt.
    #    Start by preferring the World Z-axis as the reference "up".
    reference_up_vector = Vector((0.0, 0.0, 1.0))
    # If the tangent is parallel to World Z, use World Y instead.
    if abs(tangent_local.dot(reference_up_vector)) > 0.9999:
        reference_up_vector = Vector((0.0, 1.0, 0.0))

    # Calculate the cross product: tangent x reference_up. This gives a vector
    # perpendicular to both, essentially the curve's Binormal (local X-axis).
    binormal_vector = tangent_local.cross(reference_up_vector)

    # Handle edge case where tangent might *also* be parallel to the fallback reference_up.
    # If the cross product resulted in a zero vector, try World X as reference_up.
    if binormal_vector.length_squared < 1e-12:
         reference_up_vector = Vector((1.0, 0.0, 0.0))
         binormal_vector = tangent_local.cross(reference_up_vector)
         # If still zero (tangent must be zero, which is unlikely here but handled),
         # find *any* vector orthogonal to the tangent.
         if binormal_vector.length_squared < 1e-12:
              # mathutils.Vector.orthogonal() finds an arbitrary perpendicular vector.
              arbitrary_ortho = tangent_local.orthogonal()
              binormal_vector = tangent_local.cross(arbitrary_ortho) # Guarantees non-zero result if tangent isn't zero

    # Normalize the binormal vector (local X-axis).
    binormal_vector.normalize()

    # Calculate the initial Normal (local Y-axis) using another cross product:
    # Normal = Tangent x Binormal (following right-hand rule for Z-up: Y = Z x X)
    normal_local_initial = tangent_local.cross(binormal_vector).normalized()
    # Note: The previous double cross product (v.cross(t).cross(t)) is equivalent but maybe less intuitive.

    # 3. Rotate the Initial Normal around the Tangent by the Interpolated Tilt.
    #    Create a rotation matrix representing a rotation around the tangent vector.
    tilt_rotation_matrix = Matrix.Rotation(interpolated_tilt, 3, tangent_local)
    # Apply this rotation to the initially calculated normal vector.
    normal_local_final = tilt_rotation_matrix @ normal_local_initial
    # Ensure the final normal is perfectly unit length after rotation.
    normal_local_final.normalize()


    # Return the calculated position, tangent, and tilt-aware normal in local space
    return position_local, tangent_local, normal_local_final

def pack_verts_to_string(verts_coords_list: list[Vector]):
    """Converts list of Vector coords to compressed base64 string."""
    # (Function unchanged - already includes Numpy optimization)
    if not verts_coords_list: return ""
    if HAS_NUMPY:
        try:
            float_array = np.array(verts_coords_list, dtype=np.float32).ravel()
            packed_data = zlib.compress(float_array.tobytes())
            b64_data = base64.b64encode(packed_data).decode('ascii')
            return b64_data
        except Exception as e: print(f"Warning: Error packing verts with Numpy: {e}. Using fallback.")
    try: return ";".join(",".join(f"{c:.6f}" for c in v) for v in verts_coords_list)
    except Exception as e: print(f"Error: Error packing verts (fallback): {e}"); return ""

def unpack_verts_from_string(packed_string: str):
    """Converts compressed base64 string back to list of Vector coords."""
    # (Function unchanged - already includes Numpy optimization)
    if not packed_string: return None
    if HAS_NUMPY:
        try:
            packed_data = base64.b64decode(packed_string.encode('ascii'))
            decompressed_data = zlib.decompress(packed_data)
            num_floats = len(decompressed_data) // 4
            if num_floats * 4 != len(decompressed_data) or num_floats % 3 != 0: raise ValueError("Invalid data size.")
            float_array = np.frombuffer(decompressed_data, dtype=np.float32)
            coords = float_array.reshape((-1, 3))
            return [Vector(c) for c in coords]
        except Exception as e: print(f"Warning: Error unpacking verts with Numpy: {e}. Trying fallback.")
    try:
        verts = [Vector(map(float, v_str.split(','))) for v_str in packed_string.split(';') if v_str]
        if not verts or not all(len(v) == 3 for v in verts): raise ValueError("Invalid coordinate dimensions.")
        return verts
    except Exception as e: print(f"Error: Error unpacking verts (fallback): {e}"); return None

def deform_mesh_along_curve(target_mesh_obj: bpy.types.Object,
                            curve_guide_obj: bpy.types.Object,
                            original_verts_coords: list,
                            cyl_height: float):
    """
    Applies the curve deformation to the target mesh object's vertices.
    Modifies mesh data directly. Uses numpy for speed if available.
    """
    mesh = target_mesh_obj.data

    # --- Input Validation ---
    if not curve_guide_obj or not curve_guide_obj.data or not isinstance(curve_guide_obj.data, bpy.types.Curve): 
        print(f"Error: Deform failed - Invalid curve object.")
        return False
    curve_data = curve_guide_obj.data
    
    if not curve_data.splines or curve_data.splines[0].type != 'BEZIER': 
        print(f"Error: Deform failed - Curve's first spline must be BEZIER.")
        return False
    spline = curve_data.splines[0]

    vert_count = len(mesh.vertices)
    if vert_count == 0 or not original_verts_coords or len(original_verts_coords) != vert_count: 
        print(f"Error: Deform failed - Vertex count mismatch or missing original data.")
        return False
    if cyl_height <= 1e-6: 
        print(f"Error: Deform failed - Invalid cylinder height ({cyl_height}).")
        return False

    # --- Get Transformation Matrices ---
    curve_world_matrix = curve_guide_obj.matrix_world
    curve_world_mat_invT = curve_world_matrix.to_3x3().inverted_safe().transposed() # for normal transform under non-uniform scaling
    target_inv_matrix  = target_mesh_obj.matrix_world.inverted()

    # --- Prepare Coordinate Array ---
    if HAS_NUMPY: new_coords_np = np.empty((vert_count, 3), dtype=np.float32)
    else: new_coords_flat = [0.0] * (vert_count * 3)

    # --- Process Each Vertex ---
    for i in range(vert_count):
        original_co = original_verts_coords[i] # Original local coordinate

        # Calculate curve parameter 't' (0-1) based on original local Z
        t = original_co.z / cyl_height
        t = max(0.0, min(1.0, t)) # Clamp

        # Evaluate curve frame in curve's local space
        curve_p, curve_t, curve_n = evaluate_bezier_spline_with_tilt(spline, t)

        # Transform curve frame vectors to world space
        curve_p_world = curve_world_matrix @ curve_p
        curve_t_world = (curve_world_matrix.to_3x3() @ curve_t).normalized()
        curve_n_world = (curve_world_mat_invT @ curve_n).normalized()
        curve_b_world = curve_n_world.cross(curve_t_world).normalized() # Binormal (X)

        # Construct world space transformation matrix for the curve frame at 't'
        mat_rot = Matrix((curve_b_world, curve_n_world, curve_t_world)).transposed()
        mat_frame_world = mat_rot.to_4x4()
        mat_frame_world.translation = curve_p_world

        # Original vertex position relative to the cylinder spine (XY plane offset)
        original_xy_offset_vec = Vector((original_co.x, original_co.y, 0.0, 1.0))

        # --- SIMPLIFIED CALCULATION ---
        # Transform original XY offset into the curve's world frame, then transform
        # that result back into the target object's local space in one step.
        local_pos = target_inv_matrix @ mat_frame_world @ original_xy_offset_vec
        # --- END SIMPLIFIED CALCULATION ---

        # Store result
        if HAS_NUMPY: new_coords_np[i] = local_pos
        else: idx = i * 3; new_coords_flat[idx:idx+3] = local_pos

    # --- Update Mesh Vertices Efficiently ---
    try:
        if HAS_NUMPY: mesh.vertices.foreach_set("co", new_coords_np.ravel())
        else: mesh.vertices.foreach_set("co", new_coords_flat)
        mesh.update(); return True
    except Exception as e: print(f"Error: Failed to set vertex coordinates: {e}"); return False

# --- Operator 1: Generate Objects ---
class OBJECT_OT_generate_cylinder_with_curve(bpy.types.Operator):
    """Creates/Recreates a cylinder mesh and a Bézier curve based on Scene settings."""
    bl_idname = "object.generate_cylinder_with_curve";
    bl_label = "1. Create/Recreate Objects"
    bl_description = "Generates the base cylinder and curve objects using panel settings"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context): return context.mode == 'OBJECT'

    def execute(self, context):
        scene = context.scene; props = scene.ccdg_props
        cursor_loc = scene.cursor.location.copy()

        # Read properties using renamed identifiers
        radius = props.cap_radius
        height = props.height
        fill_type = props.cap_fill_type
        num_height_segs = props.num_height_segs
        num_curve_pts = props.num_cpts_curve

        # --- Create Cylinder ---
        try:
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=props.num_cap_verts, radius=radius, depth=height, end_fill_type=fill_type,
                location=(0,0,0), scale=(1,1,1))
        except Exception as e: self.report({'ERROR'}, f"Cylinder creation failed: {e}"); return {'CANCELLED'}

        cyl_obj = context.active_object
        if not cyl_obj or cyl_obj.type != 'MESH':
            self.report({'ERROR'}, "Failed to get created cylinder.")
            return {'CANCELLED'}
        cyl_obj.name = "DeformCylinder"
        mesh_data = cyl_obj.data
        mesh_data.name = "DeformCylinderMesh"

        # --- Subdivide Height using BMesh ---
        if num_height_segs > 1:
            bm = None
            try:
                bm = bmesh.new(); bm.from_mesh(mesh_data)
                bm.verts.ensure_lookup_table();
                bm.edges.ensure_lookup_table()
                # Collect vertical edges for spliting
                vertical_edges = []; 
                eps_len, eps_xy = height * 0.01, 0.01; 
                for edge in bm.edges:
                    v1, v2 = edge.verts[0].co, edge.verts[1].co
                    if abs((v1 - v2).length - height) < eps_len and \
                       abs(v1.x - v2.x) < eps_xy and abs(v1.y - v2.y) < eps_xy:
                        vertical_edges.append(edge)
                if vertical_edges:
                    bmesh.ops.subdivide_edges(bm, edges=vertical_edges, cuts=num_height_segs - 1, use_grid_fill=False)
                    bm.to_mesh(mesh_data); mesh_data.update()
                else: self.report({'WARNING'}, "No vertical edges found for subdivision.")
            except Exception as e: self.report({'WARNING'}, f"BMesh subdivision failed: {e}")
            finally:
                if bm: bm.free()

        # --- Set Origin to Base Center ---
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = cyl_obj
        cyl_obj.select_set(True)
        context.scene.cursor.location = (0.0, 0.0, -height / 2.0)
        bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
        cyl_obj.location = (0,0,0)

        # --- Create Bézier Curve ---
        curve_data = bpy.data.curves.new('BendCurveData', type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.resolution_u = 12
        spline = curve_data.splines.new(type='BEZIER')
        spline.bezier_points.add(count=num_curve_pts - 1)
        seg_len = height / (num_curve_pts - 1) if num_curve_pts > 1 else 0
        for i, bp in enumerate(spline.bezier_points):
            z = i * seg_len; bp.co = Vector((0.0, 0.0, z))
            h_offset = seg_len / 3.0
            bp.handle_left = Vector((0.0, 0.0, z - h_offset))
            bp.handle_right = Vector((0.0, 0.0, z + h_offset))
            # bp.handle_left_type = 'ALIGNED'; 
            # bp.handle_right_type = 'ALIGNED'; 
            bp.handle_left_type = 'AUTO'; 
            bp.handle_right_type = 'AUTO'; 
            bp.tilt = 0.0
        curve_obj = bpy.data.objects.new('BendCurveObject', curve_data)
        curve_obj.location = (0,0,0); context.collection.objects.link(curve_obj)

        # --- Final Selection & Cleanup ---
        bpy.ops.object.select_all(action='DESELECT')
        context.view_layer.objects.active = cyl_obj; cyl_obj.select_set(True)
        curve_obj.select_set(True); context.scene.cursor.location = cursor_loc
        self.report({'INFO'}, f"Created '{cyl_obj.name}' and '{curve_obj.name}'."); return {'FINISHED'}


# --- Operator 2: Enable Realtime Deform ---
class OBJECT_OT_enable_realtime_bend(bpy.types.Operator):
    """Enables realtime deformation handler for the active mesh..."""
    bl_idname = "object.enable_realtime_bend"; bl_label = "Enable Realtime Deform"
    bl_description = "Links active mesh to selected curve for realtime updates (Can be slow!)"
    bl_options = {'REGISTER', 'UNDO'}

    # Class variable for caching original coordinates
    original_coords_cache = {}

    @classmethod
    def poll(cls, context):
        active_obj = context.active_object
        return (context.mode == 'OBJECT' and active_obj and active_obj.type == 'MESH' and
                len(context.selected_objects) == 2 and
                not active_obj.get(PROP_ENABLED, False) and
                any(obj.type == 'CURVE' for obj in context.selected_objects if obj != active_obj))

    def find_curve_object(self, context):
        active_obj = context.active_object
        for obj in context.selected_objects:
            if obj != active_obj and obj.type == 'CURVE': return obj
        return None

    def execute(self, context):
        target_obj = context.active_object; curve_obj = self.find_curve_object(context)
        props = context.scene.ccdg_props
        if not curve_obj: self.report({'ERROR'}, "No valid curve object selected."); return {'CANCELLED'}

        # Store original vertex data
        original_verts = [v.co.copy() for v in target_obj.data.vertices]
        packed_verts = pack_verts_to_string(original_verts)
        if not packed_verts: self.report({'ERROR'}, "Failed to pack original vertex data."); return {'CANCELLED'}

        # Store in cache and custom properties
        OBJECT_OT_enable_realtime_bend.original_coords_cache[target_obj.name] = original_verts
        target_obj[PROP_ENABLED] = True
        target_obj[PROP_CURVE_NAME] = curve_obj.name
        target_obj[PROP_ORIG_VERTS] = packed_verts
        target_obj[PROP_HEIGHT] = props.height # Store original height

        # Add to monitored set
        MONITORED_MESH_OBJECTS.add(target_obj.name)

        # Apply initial deformation
        self.report({'INFO'}, "Applying initial deformation...")
        success = deform_mesh_along_curve(target_obj, curve_obj, original_verts, props.height)

        if success:
            self.report({'INFO'}, f"Enabled realtime deform for '{target_obj.name}' linked to '{curve_obj.name}'.")
            context.area.tag_redraw() # Update UI
            return {'FINISHED'}
        else: # Clean up on failure
            target_obj.pop(PROP_ENABLED, None); target_obj.pop(PROP_CURVE_NAME, None)
            target_obj.pop(PROP_ORIG_VERTS, None); target_obj.pop(PROP_HEIGHT, None)
            OBJECT_OT_enable_realtime_bend.original_coords_cache.pop(target_obj.name, None)
            MONITORED_MESH_OBJECTS.discard(target_obj.name)
            self.report({'ERROR'}, "Initial deformation failed."); return {'CANCELLED'}

# --- Operator 3: Disable Realtime Deform ---
class OBJECT_OT_disable_realtime_bend(bpy.types.Operator):
    """Disables realtime deformation handler and restores original mesh shape."""
    bl_idname = "object.disable_realtime_bend"; bl_label = "Disable Realtime Deform"
    bl_description = "Stops realtime updates and restores original mesh shape"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (context.active_object and context.active_object.type == 'MESH' and
                context.active_object.get(PROP_ENABLED, False))

    def execute(self, context):
        target_obj = context.active_object; mesh = target_obj.data; obj_name = target_obj.name

        # Restore original shape
        packed_verts = target_obj.get(PROP_ORIG_VERTS)
        if packed_verts:
            original_verts = unpack_verts_from_string(packed_verts)
            if original_verts and len(original_verts) == len(mesh.vertices):
                 try:
                     flat_coords = [c for v_co in original_verts for c in v_co]
                     mesh.vertices.foreach_set("co", flat_coords); mesh.update()
                     self.report({'INFO'}, f"Restored original shape for '{obj_name}'.")
                 except Exception as e: self.report({'ERROR'}, f"Failed to restore vertex coords: {e}")
            elif original_verts: self.report({'WARNING'}, f"Could not restore '{obj_name}': Vertex count changed.")
            else: self.report({'WARNING'}, f"Could not restore '{obj_name}': Failed to unpack original data.")
        else: self.report({'WARNING'}, f"No original shape data found for '{obj_name}'.")

        # Cleanup
        MONITORED_MESH_OBJECTS.discard(obj_name)
        target_obj.pop(PROP_ENABLED, None); target_obj.pop(PROP_CURVE_NAME, None)
        target_obj.pop(PROP_ORIG_VERTS, None); target_obj.pop(PROP_HEIGHT, None)
        OBJECT_OT_enable_realtime_bend.original_coords_cache.pop(obj_name, None) # Clear cache entry

        self.report({'INFO'}, f"Disabled realtime deform for '{obj_name}'.")
        context.area.tag_redraw() # Update UI
        return {'FINISHED'}

# --- Depsgraph Update Handler ---
#@bpy.app.handlers.persistent
def ccdg_depsgraph_handler(scene: bpy.types.Scene, depsgraph: bpy.types.Depsgraph):
    """Checks for updates to monitored curves and deforms linked meshes."""
    if not MONITORED_MESH_OBJECTS: return

    # Check if any relevant object was updated in the dependency graph
    relevant_update = False
    for update in depsgraph.updates:
        if isinstance(update.id, bpy.types.Object) and \
           (update.is_updated_geometry or update.is_updated_transform):
             relevant_update = True; break
    if not relevant_update: return

    # Check each monitored mesh
    for mesh_name in list(MONITORED_MESH_OBJECTS):
        mesh_obj = scene.objects.get(mesh_name)
        # Validate mesh object and its enabled state
        if not mesh_obj or mesh_obj.type != 'MESH' or not mesh_obj.get(PROP_ENABLED, False):
            MONITORED_MESH_OBJECTS.discard(mesh_name); continue

        # Validate linked curve object
        curve_name = mesh_obj.get(PROP_CURVE_NAME)
        curve_obj = scene.objects.get(curve_name) if curve_name else None
        if not curve_obj or curve_obj.type != 'CURVE': continue

        # --- Check if the linked curve was updated ---
        # Iterate through updates to reliably check if curve object or data was modified.
        curve_was_updated = False
        for update in depsgraph.updates: 
            if hasattr(update.id, "original") is False: continue
            if (update.id.original == curve_obj and (update.is_updated_transform or update.is_updated_geometry)): 
                curve_was_updated = True
                break
            if (update.id.original == curve_obj.data and update.is_updated_geometry): 
                # Check if the update ID matches the curve's data block
                curve_was_updated = True
                break

        # If curve updated, retrieve data and deform
        if curve_was_updated:
            # print(f"Handler: Curve '{curve_name}' updated, deforming '{mesh_name}'...") # Debug
            packed_verts = mesh_obj.get(PROP_ORIG_VERTS)
            cyl_height = mesh_obj.get(PROP_HEIGHT) # Use stored height
            if packed_verts and cyl_height is not None:
                original_verts = unpack_verts_from_string(packed_verts)
                if original_verts and len(original_verts) == len(mesh_obj.data.vertices):
                    deform_mesh_along_curve(mesh_obj, curve_obj, original_verts, cyl_height)
                # else: Error printed by deform_mesh_along_curve or unpack
            # else: Error: Missing custom props

# --- UI Panel ---
class VIEW3D_PT_cylinder_curve_deform(bpy.types.Panel):
    """UI Panel in the 3D Viewport Sidebar (N-Panel > Create Tab)"""
    bl_label = "Cylinder Curve Gen"; bl_idname = "VIEW3D_PT_cylinder_curve_deform"
    bl_space_type = 'VIEW_3D'; bl_region_type = 'UI'; bl_category = 'Create'

    def draw(self, context):
        layout = self.layout; scene = context.scene; active_obj = context.active_object
        props = scene.ccdg_props
        if not props: layout.label(text="Error: Addon properties not found!", icon='ERROR'); return

        # --- Section 1: Object Generation ---
        box_gen = layout.box(); col_gen = box_gen.column(align=True)
        col_gen.label(text="1. Object Generation Settings:")
        # Use RENAMED property identifiers
        col_gen.prop(props, "num_cap_verts"); col_gen.prop(props, "cap_radius")
        col_gen.prop(props, "height"); col_gen.prop(props, "num_height_segs")
        col_gen.prop(props, "cap_fill_type"); col_gen.prop(props, "num_cpts_curve")
        col_gen.operator(OBJECT_OT_generate_cylinder_with_curve.bl_idname, icon='MESH_CYLINDER')

        layout.separator()

        # --- Section 2: Realtime Python Deformer ---
        box_rt = layout.box(); col_rt = box_rt.column(align=True)
        col_rt.label(text="2. Realtime Python Deformer:")
        if active_obj and active_obj.type == 'MESH':
            is_enabled = active_obj.get(PROP_ENABLED, False)
            linked_curve_name = active_obj.get(PROP_CURVE_NAME, "None")
            if is_enabled:
                row = col_rt.row(align=True)
                row.label(text=f"Active: Linked to '{linked_curve_name}'", icon='CHECKMARK')
                row.operator(OBJECT_OT_disable_realtime_bend.bl_idname, text="", icon='X')
                col_rt.label(text="Edit linked curve for realtime updates.", icon='INFO')
                col_rt.label(text="WARNING: Can be slow!", icon='ERROR')
            else:
                col_rt.operator(OBJECT_OT_enable_realtime_bend.bl_idname, icon='MOD_CURVE')
                col_rt.label(text="Select Mesh (Active) and Curve, then Enable.", icon='INFO')
        else: col_rt.label(text="Select Mesh Object to see status.", icon='INFO')


# --- Visualization Operator ---
# Add this class to your __init__.py, assuming evaluate_bezier_spline_with_tilt is defined above.
# Also add VISUALIZE_OT_curve_frames to the 'classes' tuple near the end of the file.

class VISUALIZE_OT_curve_frames(bpy.types.Operator):
# --- How to Use ---
# 1. Make sure the `evaluate_bezier_spline_with_tilt` function is defined above this class.
# 2. Add `VISUALIZE_OT_curve_frames` to the `classes` tuple in your `register()` function.
# 3. Re-register the addon.
# 4. Select your Bézier curve object.
# 5. Press F3, search for "Visualize Curve Frames", and run the operator.
# 6. Adjust the "Steps" and "Scale" in the operator redo panel (bottom left) if needed.
    """Visualizes the Tangent, Normal, and Binormal frames along the active curve"""
    bl_idname = "object.visualize_curve_frames"
    bl_label = "Visualize Curve Frames"
    bl_options = {'REGISTER', 'UNDO'}

    num_steps: bpy.props.IntProperty(
        name="Steps",
        description="Number of evaluation points along the curve",
        default=20,
        min=2,
        max=200
    )
    scale: bpy.props.FloatProperty(
        name="Vector Scale",
        description="Length of the visualized vectors",
        default=0.2,
        min=0.01,
        max=10.0
    )
    vector_radius: bpy.props.FloatProperty(
        name="Vector Radius",
        description="Radius of the cylinder markers used for vectors",
        default=0.015,
        min=0.001,
        max=1.0
    )

    @classmethod
    def poll(cls, context):
        # Enable only if the active object is a Curve
        return context.active_object and context.active_object.type == 'CURVE'

    def execute(self, context):
        curve_obj = context.active_object
        curve_data = curve_obj.data

        if not curve_data.splines or curve_data.splines[0].type != 'BEZIER':
            self.report({'ERROR'}, "Active object's first spline must be BEZIER type.")
            return {'CANCELLED'}

        spline = curve_data.splines[0]

        # --- Create Materials for Colors ---
        def create_material(name, color):
            mat = bpy.data.materials.get(name)
            if mat is None:
                mat = bpy.data.materials.new(name=name)
                mat.use_nodes = False # Simple material
                mat.diffuse_color = color
            return mat

        mat_tangent = create_material("Vis_Tangent", (1.0, 0.0, 0.0, 1.0)) # Red
        mat_normal = create_material("Vis_Normal", (0.0, 1.0, 0.0, 1.0))   # Green
        mat_binormal = create_material("Vis_Binormal", (0.0, 0.0, 1.0, 1.0)) # Blue

        # --- Create Parent Empty ---
        # Remove previous visualization if it exists
        old_empty_name = f"{curve_obj.name}_FramesViz"
        old_empty = bpy.data.objects.get(old_empty_name)
        if old_empty:
            # Delete all children first
            for child in list(old_empty.children): # Iterate over a copy
                bpy.data.objects.remove(child, do_unlink=True)
            bpy.data.objects.remove(old_empty, do_unlink=True)

        # Create new empty
        viz_empty = bpy.data.objects.new(old_empty_name, None)
        context.collection.objects.link(viz_empty)
        viz_empty.matrix_world = Matrix.Identity(4) # Place at world origin


        # --- Create Visualization Geometry Function ---
        def create_vector_marker(name, origin_world, direction_world, length, radius, material, parent_obj):
            """Creates a cylinder representing the vector"""
            if direction_world.length < 1e-6: return None # Avoid zero vectors

            # Calculate rotation to align cylinder (default Z-axis) with direction
            rot_quat = direction_world.normalized().to_track_quat('Z', 'Y')

            # Create cylinder
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=8,
                radius=radius,
                depth=length,
                location=(0,0,0), # Create at origin first
                scale=(1,1,1)
            )
            marker = context.active_object
            marker.name = name

            # Assign material
            if marker.data.materials: marker.data.materials[0] = material
            else: marker.data.materials.append(material)

            # Set final transform
            # Position the base of the cylinder at origin_world, aligned with direction
            marker.matrix_world = Matrix.Translation(origin_world + direction_world * (length / 2.0)) @ rot_quat.to_matrix().to_4x4()

            # Parent to the main empty
            marker.parent = parent_obj
            return marker

        # --- Evaluate and Visualize ---
        curve_world_matrix = curve_obj.matrix_world
        curve_rot_matrix = curve_world_matrix.to_3x3()
        curve_rot_matrix_invT = curve_rot_matrix.inverted_safe().transposed()

        for i in range(self.num_steps):
            t = i / (self.num_steps - 1) if self.num_steps > 1 else 0.5 # Parameter 0.0 to 1.0

            # Evaluate using the function from your addon
            pos_local, tan_local, norm_local = evaluate_bezier_spline_with_tilt(spline, t) #

            # Calculate binormal locally (consistent with how norm_local was likely derived)
            # In evaluate_bezier_spline_with_tilt, norm_final = tilt_rot @ (tan x bino_initial)
            # where bino_initial = tan x ref_up.
            # Standard Frame: T, N, B (often X, Y, Z or Z, Y, X)
            # If T is Z-like, N is Y-like, then B = T x N (X-like)
            # Let's assume T=Z, N=Y, B=X (local frame)
            # If evaluate_bezier_spline_with_tilt gives T and N, calculate B = T.cross(N)
            bino_local = tan_local.cross(norm_local).normalized()


            # Transform to World Space
            pos_world = curve_world_matrix @ pos_local
            tan_world = (curve_rot_matrix @ tan_local).normalized()
            norm_world = (curve_rot_matrix_invT @ norm_local).normalized()
            bino_world = (curve_rot_matrix @ bino_local).normalized()


            # Create Markers
            create_vector_marker(f"T_{i:03d}", pos_world, tan_world, self.scale, self.vector_radius, mat_tangent, viz_empty)
            create_vector_marker(f"N_{i:03d}", pos_world, norm_world, self.scale, self.vector_radius, mat_normal, viz_empty)
            create_vector_marker(f"B_{i:03d}", pos_world, bino_world, self.scale, self.vector_radius, mat_binormal, viz_empty)

        self.report({'INFO'}, f"Generated {self.num_steps} frame visualizations.")
        return {'FINISHED'}


# --- Registration ---
classes = (
    CCDG_Properties,
    OBJECT_OT_generate_cylinder_with_curve,
    OBJECT_OT_enable_realtime_bend,
    OBJECT_OT_disable_realtime_bend,
    VIEW3D_PT_cylinder_curve_deform,
    VISUALIZE_OT_curve_frames
)
_handler_ref = ccdg_depsgraph_handler # Keep reference

def register():
    """Registers all addon classes and the depsgraph handler."""
    print(f"Registering ..")
    if HAS_NUMPY: print(f"- Numpy {np.__version__ if hasattr(np,'__version__') else ''} detected.")
    else: print("- Numpy not detected (using fallback packing).")
    for cls in classes:
        try: bpy.utils.register_class(cls)
        except ValueError: pass # Ignore if already registered
    # Add PropertyGroup pointer to Scene
    if not hasattr(bpy.types.Scene, 'ccdg_props'):
        bpy.types.Scene.ccdg_props = bpy.props.PointerProperty(type=CCDG_Properties)
    # Clear monitored set and add handler
    MONITORED_MESH_OBJECTS.clear()
    if _handler_ref not in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.append(_handler_ref)
        print("- Realtime handler added.")
    print(f"Registered.")

def unregister():
    """Unregisters all addon classes and removes the depsgraph handler."""
    print(f"Unregistering ..")
    # Remove handler first
    if _handler_ref in bpy.app.handlers.depsgraph_update_post:
        try: bpy.app.handlers.depsgraph_update_post.remove(_handler_ref); print("- Realtime handler removed.")
        except ValueError: print(f"- Warning: Realtime handler was already removed.")
    # Clear cache and monitored set
    if hasattr(OBJECT_OT_enable_realtime_bend, 'original_coords_cache'):
         OBJECT_OT_enable_realtime_bend.original_coords_cache.clear()
    MONITORED_MESH_OBJECTS.clear()
    # Delete Scene property
    if hasattr(bpy.types.Scene, 'ccdg_props'):
        try: delattr(bpy.types.Scene, 'ccdg_props')
        except Exception as e: print(f"Warning: Could not delete scene property 'ccdg_props': {e}")
    # Unregister classes
    for cls in reversed(classes):
         if hasattr(bpy.types, cls.__name__):
            try: bpy.utils.unregister_class(cls)
            except RuntimeError: pass # Ignore errors
    print(f"Unregistered.")

# Standard execution guard for running from Text Editor
if __name__ == "__main__":
    print(f"\n--- Running Script: {__file__} ---")
    try: print("\n--- Attempting Unregister from __main__ ---"); unregister()
    except Exception as e: print(f"Error during automatic unregister on script run: {e}")
    print("\n--- Attempting Register from __main__ ---"); register()
