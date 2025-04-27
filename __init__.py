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

# --- Local Addon Imports ---
from . import rmf_utils

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

# --- Modified Deform Function ---

def deform_mesh_along_curve(target_mesh_obj: bpy.types.Object,
                            curve_guide_obj: bpy.types.Object,
                            original_verts_coords: list,
                            cyl_height: float,
                            rmf_steps: int = 50): # Add RMF resolution parameter
    """
    Applies curve deformation using pre-calculated RMF frames.
    Uses NEAREST pre-calculated frame (less accurate than interpolation).
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
    if rmf_steps < 2: rmf_steps = 2 # Need at least 2 steps for RMF

    # --- Pre-calculate RMF Frames ---
    print(f"Calculating {rmf_steps} RMF frames...")
    rmf_frames = rmf_utils.calculate_rmf_frames(spline, rmf_steps)
    if not rmf_frames:
        print(f"Error: RMF frame calculation failed.")
        return False
    print("RMF calculation complete.")

    # --- Get Transformation Matrices ---
    curve_world_matrix = curve_guide_obj.matrix_world
    target_inv_matrix  = target_mesh_obj.matrix_world.inverted()

    # --- Prepare Coordinate Array ---
    if HAS_NUMPY: new_coords_np = np.empty((vert_count, 3), dtype=np.float32)
    else: new_coords_flat = [0.0] * (vert_count * 3)

    num_rmf_frames = len(rmf_frames)

    # --- Process Each Vertex ---
    for i in range(vert_count):
        original_co = original_verts_coords[i] # Original local coordinate

        # Calculate curve parameter 't' (0-1) based on original local Z
        vertex_t = original_co.z / cyl_height
        vertex_t = max(0.0, min(1.0, vertex_t)) # Clamp

        # --- Find NEAREST pre-calculated RMF frame ---
        # Map vertex_t (0-1) to the closest index in rmf_frames (0 to num_rmf_frames-1)
        nearest_idx = round(vertex_t * (num_rmf_frames - 1))
        nearest_idx = max(0, min(num_rmf_frames - 1, nearest_idx)) # Clamp index

        # Get frame data from the nearest pre-calculated frame
        curve_p_local, curve_t_local, curve_n_rmf_local, frame_t = rmf_frames[nearest_idx]

        # --- Apply User Tilt ---
        # Get interpolated tilt value at the vertex's specific parameter t
        interpolated_tilt = rmf_utils.get_interpolated_tilt(spline, vertex_t)
        # Create rotation matrix around the RMF tangent
        tilt_rotation_matrix = Matrix.Rotation(interpolated_tilt, 3, curve_t_local)
        # Apply tilt to the RMF normal
        curve_n_final_local = tilt_rotation_matrix @ curve_n_rmf_local
        curve_n_final_local.normalize() # Ensure unit length

        # Calculate final binormal based on tilted frame
        curve_b_final_local = curve_t_local.cross(curve_n_final_local).normalized()

        # --- Transform Frame to World Space ---
        # Use the correct inverse-transpose for normals/binormals if non-uniform scale
        curve_rot_scale_matrix = curve_world_matrix.to_3x3()
        try:
            inv_matrix = curve_rot_scale_matrix.inverted_safe()
            inv_trans_matrix = inv_matrix.transposed()
            transform_normals_correctly = True
        except ValueError:
            inv_trans_matrix = Matrix.Identity(3)
            transform_normals_correctly = False

        curve_p_world = curve_world_matrix @ curve_p_local
        curve_t_world = (curve_rot_scale_matrix @ curve_t_local).normalized()

        if transform_normals_correctly:
            curve_n_world = (inv_trans_matrix @ curve_n_final_local).normalized()
            curve_b_world = (inv_trans_matrix @ curve_b_final_local).normalized()
        else: # Fallback
             curve_n_world = (curve_rot_scale_matrix @ curve_n_final_local).normalized()
             curve_b_world = (curve_rot_scale_matrix @ curve_b_final_local).normalized()
             # Re-ensure orthogonality if needed: curve_b_world = curve_t_world.cross(curve_n_world).normalized() ?

        # Construct world space transformation matrix for the final tilted frame at 't'
        # Ensure axes form a valid RHS basis: T, N, B ? Check convention.
        # If T=Z, N=Y, B=X then mat = (B, N, T). Transposed for column vectors.
        # If T=X, N=Y, B=Z then mat = (T, N, B).
        # Let's assume T=Tangent(Z), N=Normal(Y), B=Binormal(X) convention for cylinder alignment
        mat_rot = Matrix((curve_b_world, curve_n_world, curve_t_world)).transposed()
        mat_frame_world = mat_rot.to_4x4()
        mat_frame_world.translation = curve_p_world

        # Original vertex position relative to the cylinder spine (XY plane offset)
        original_xy_offset_vec = Vector((original_co.x, original_co.y, 0.0)) # Use 3D vector

        # Transform original XY offset into the curve's world frame, then transform
        # that result back into the target object's local space.
        local_pos = target_inv_matrix @ mat_frame_world @ original_xy_offset_vec

        # Store result
        if HAS_NUMPY: new_coords_np[i] = local_pos[:3] # Ensure 3D vector
        else: idx = i * 3; new_coords_flat[idx:idx+3] = local_pos[:3]

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

    def create_gradient_material(self, name, base_color):
        """Creates a node-based material with a gradient along local Z."""
        mat = bpy.data.materials.get(name)
        if mat is None:
            mat = bpy.data.materials.new(name=name)
        else:
            # Clear existing nodes if reusing material
            if mat.node_tree:
                mat.node_tree.nodes.clear()

        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        # Clear default nodes
        nodes.clear()

        # Create nodes
        tex_coord = nodes.new(type='ShaderNodeTexCoord')
        separate_xyz = nodes.new(type='ShaderNodeSeparateXYZ')
        color_ramp = nodes.new(type='ShaderNodeValToRGB')
        principled_bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        output_node = nodes.new(type='ShaderNodeOutputMaterial')

        # Configure Color Ramp for gradient (Base Color -> White)
        color_ramp.color_ramp.elements[0].position = 0 # Base color at position 0
        color_ramp.color_ramp.elements[0].color = (*base_color[:3], 1.0) 
        color_ramp.color_ramp.elements[1].position = 1.0
        color_ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0) # White at position 1
        # Add a NEW stop in between that is also the base color
        new_stop = color_ramp.color_ramp.elements.new(position=0.7) # Add stop at 70%
        new_stop.color = (*base_color[:3], 1.0) # Make it the base color

        # Position nodes for clarity (optional)
        tex_coord.location = (-600, 0)
        separate_xyz.location = (-400, 0)
        color_ramp.location = (-200, 0)
        principled_bsdf.location = (0, 0)
        output_node.location = (200, 0)

        # Link nodes
        # Generated Coords -> Separate XYZ -> Z output -> ColorRamp Fac -> BSDF Base Color
        links.new(tex_coord.outputs['Generated'], separate_xyz.inputs['Vector'])
        links.new(separate_xyz.outputs['Z'], color_ramp.inputs['Fac']) # Use Z axis for gradient along length
        links.new(color_ramp.outputs['Color'], principled_bsdf.inputs['Base Color'])
        links.new(principled_bsdf.outputs['BSDF'], output_node.inputs['Surface'])

        return mat

    def execute(self, context):
        curve_obj = context.active_object
        curve_data = curve_obj.data

        if not curve_data.splines or curve_data.splines[0].type != 'BEZIER':
            self.report({'ERROR'}, "Active object's first spline must be BEZIER type.")
            return {'CANCELLED'}

        spline = curve_data.splines[0]

        # --- Create Gradient Materials ---
        # (Keep your create_gradient_material method and calls here)
        mat_tangent  = self.create_gradient_material("Vis_Tangent_Grad", (1.0, 0.0, 0.0))
        mat_normal   = self.create_gradient_material("Vis_Normal_Grad", (0.0, 1.0, 0.0))
        mat_binormal = self.create_gradient_material("Vis_Binormal_Grad", (0.0, 0.0, 1.0))

        # --- Create Parent Empty ---
        # (Keep the empty creation/deletion logic here)
        old_empty_name = f"{curve_obj.name}_FramesViz"
        # ... (rest of empty handling) ...
        viz_empty = bpy.data.objects.new(old_empty_name, None)
        context.collection.objects.link(viz_empty)
        viz_empty.matrix_world = Matrix.Identity(4)


        # --- Create Visualization Geometry Function ---
        def create_vector_marker(name, origin_world, direction_world, length, radius, material, parent_obj):
            """Creates a cylinder representing the vector"""
            if direction_world.length < 1e-6: return None # Avoid zero vectors

            # Calculate rotation to align cylinder (default Z-axis) with direction
            rot_quat = direction_world.normalized().to_track_quat('Z', 'Y')

            # Create cylinder
            # Ensure cylinder is created at origin BEFORE setting matrix_world
            bpy.ops.mesh.primitive_cylinder_add(
                vertices=8,
                radius=radius,
                depth=length,
                location=(0,0,0), # Create at origin
                scale=(1,1,1),
                rotation=(0,0,0) # No initial rotation
            )
            marker = context.active_object
            marker.name = name

            # Assign material
            if marker.data.materials: marker.data.materials[0] = material
            else: marker.data.materials.append(material)

            # Set final transform
            # Position the base of the cylinder slightly offset FOR GRADIENT (adjust if needed)
            # The cylinder's origin is its center; generated coords run -0.5 to 0.5 along Z depth
            # To have gradient base start near origin_world, we place center at origin + dir*len/2
            marker.matrix_world = Matrix.Translation(origin_world + direction_world.normalized() * (length / 2.0)) @ rot_quat.to_matrix().to_4x4()


            # Parent to the main empty
            marker.parent = parent_obj
            return marker

        # --- Calculate RMF Frames ---
        rmf_frames = rmf_utils.calculate_rmf_frames(spline, self.num_steps)
        if not rmf_frames:
            self.report({'ERROR'}, "Failed to calculate RMF frames.")
            return {'CANCELLED'}

        # --- Visualize the Calculated RMF Frames ---
        curve_world_matrix = curve_obj.matrix_world
        curve_rot_scale_matrix = curve_world_matrix.to_3x3()
        try:
            inv_matrix = curve_rot_scale_matrix.inverted_safe()
            inv_trans_matrix = inv_matrix.transposed()
            transform_normals_correctly = True
        except ValueError:
            inv_trans_matrix = Matrix.Identity(3)
            transform_normals_correctly = False

        for i, frame_data in enumerate(rmf_frames):
            pos_local, tan_local, norm_rmf_local, frame_t = frame_data

            # --- Apply Tilt for Visualization ---
            # Get interpolated tilt value at this frame's specific parameter t
            interpolated_tilt = rmf_utils.get_interpolated_tilt(spline, frame_t)
            tilt_rotation_matrix = Matrix.Rotation(interpolated_tilt, 3, tan_local)
            norm_final_local = tilt_rotation_matrix @ norm_rmf_local
            norm_final_local.normalize()
            bino_final_local = tan_local.cross(norm_final_local).normalized()

            # --- Transform to World Space ---
            pos_world = curve_world_matrix @ pos_local
            tan_world  = (curve_rot_scale_matrix @ tan_local).normalized() # Transform T

            if transform_normals_correctly:
                norm_world = (inv_trans_matrix @ norm_final_local).normalized() # Transform N correctly
            else: # Fallback
                norm_world = (curve_rot_scale_matrix @ norm_final_local).normalized()

            # --- Re-orthogonalize for Visualization ---
            # Recalculate Binormal based on world T and N
            bino_world = tan_world.cross(norm_world).normalized()
            # Optional: Recalculate Normal based on world T and B to ensure perfect orthogonality
            norm_world = bino_world.cross(tan_world).normalized()
            # --- End Re-orthogonalization ---

            # After calculating tan_world, norm_world, bino_world...
            dot_tn = tan_world.dot(norm_world)
            dot_tb = tan_world.dot(bino_world)
            dot_nb = norm_world.dot(bino_world)
            if abs(dot_tn) > 1e-4 or abs(dot_tb) > 1e-4 or abs(dot_nb) > 1e-4:
                 print(f"Step {i}: Non-orthogonality detected! T.N={dot_tn:.5f}, T.B={dot_tb:.5f}, N.B={dot_nb:.5f}")

            # Create Markers for the TILTED frame
            create_vector_marker(f"T_{i:03d}", pos_world, tan_world, self.scale, self.vector_radius, mat_tangent, viz_empty)
            create_vector_marker(f"N_{i:03d}", pos_world, norm_world, self.scale, self.vector_radius, mat_normal, viz_empty)
            create_vector_marker(f"B_{i:03d}", pos_world, bino_world, self.scale, self.vector_radius, mat_binormal, viz_empty)

        self.report({'INFO'}, f"Generated {self.num_steps} RMF frame visualizations.")
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
