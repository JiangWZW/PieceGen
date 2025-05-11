import bpy # Needed for bpy.types.Spline hint, potentially others
import math
from mathutils import Vector, Matrix

from . import common_vars as cvars

# --- Helper: Bezier Curve Derivatives ---

def get_bezier_segment_data(spline: bpy.types.Spline, segment_index: int):
    """Gets P0, P1, P2, P3 for a cubic Bezier segment index."""
    if not spline or not spline.bezier_points or spline.type != 'BEZIER':
        return None
    # Check segment index bounds carefully
    if segment_index < 0 or segment_index >= len(spline.bezier_points) - 1:
         # Allow index 0 even if only 1 point exists for specific fallbacks?
         if segment_index == 0 and len(spline.bezier_points) == 1:
              pt = spline.bezier_points[0]
              # Return segment where P0=P3=co, P1=handle_right, P2=handle_left?
              # Simplification: Return None for now if not a valid segment
              return None
         else:
             return None


    start_point = spline.bezier_points[segment_index]
    end_point = spline.bezier_points[segment_index + 1]
    p0 = start_point.co
    p1 = start_point.handle_right
    p2 = end_point.handle_left
    p3 = end_point.co
    return p0, p1, p2, p3

def evaluate_bezier_segment(p0: Vector, p1: Vector, p2: Vector, p3: Vector, t: float, order: int = 0):
    """
    Evaluates position (order=0), 1st derivative (order=1), or
    2nd derivative (order=2) of a cubic Bezier segment defined by P0,P1,P2,P3
    at local parameter t (0-1).
    """
    t = max(0.0, min(1.0, t)) # Clamp t
    omt = 1.0 - t

    if order == 0: # Position (Bernstein polynomial)
        omt2 = omt * omt
        omt3 = omt2 * omt
        t2 = t * t
        t3 = t2 * t
        pos = (omt3 * p0) + (3.0 * omt2 * t * p1) + (3.0 * omt * t2 * p2) + (t3 * p3)
        return pos
    elif order == 1: # First Derivative (Tangent vector direction)
        omt2 = omt * omt
        t2 = t * t
        tangent = (3.0 * omt2 * (p1 - p0)) + \
                  (6.0 * omt * t * (p2 - p1)) + \
                  (3.0 * t2 * (p3 - p2))
        return tangent
    elif order == 2: # Second Derivative (Curvature vector direction)
        term1 = p2 - 2.0 * p1 + p0
        term2 = p3 - 2.0 * p2 + p1
        second_deriv = (6.0 * omt * term1) + (6.0 * t * term2)
        return second_deriv
    else:
        raise ValueError("Order must be 0, 1, or 2")

def get_spline_parameter_info(spline: bpy.types.Spline, global_t: float):
    """Maps global t (0-1) to segment index and local t (0-1)."""
    if not spline or not spline.bezier_points or len(spline.bezier_points) < 2:
        # If only one point, treat as single segment index 0, local_t irrelevant?
        if spline and spline.bezier_points and len(spline.bezier_points) == 1:
             return 0, 0.0
        return -1, 0.0 # Indicate invalid segment if less than 2 points

    num_points = len(spline.bezier_points)
    num_segments = num_points - 1

    t_clamped = max(0.0, min(1.0, global_t))

    if abs(t_clamped - 1.0) < 1e-9: # Handle t exactly at 1.0
        segment_index = num_segments - 1
        local_t = 1.0
    elif abs(t_clamped) < 1e-9: # Handle t exactly at 0.0
        segment_index = 0
        local_t = 0.0
    else:
        segment_float = t_clamped * num_segments
        segment_index = min(math.floor(segment_float), num_segments - 1) # Ensure valid index
        local_t = segment_float - segment_index
        # Handle precision near segment boundary? If local_t is extremely close to 1,
        # should it map to the *next* segment's start (local_t=0)?
        # Let's keep it simple: maps to current segment unless exactly 1.0

    # Final check if index is valid (can happen if num_segments is 0)
    if segment_index < 0:
        segment_index = 0
        local_t = 0.0

    return segment_index, local_t

def evaluate_spline(spline: bpy.types.Spline, global_t: float, order: int = 0):
    """Evaluates position or derivatives for the whole spline at global t."""
    segment_index, local_t = get_spline_parameter_info(spline, global_t)

    # Handle single-point spline case specifically
    if len(spline.bezier_points) == 1:
         if order == 0: return spline.bezier_points[0].co.copy()
         else: return Vector((0,0,0)) # No derivative info

    if segment_index < 0: # Should not happen if spline has points
        print(f"Warning: Invalid segment index {segment_index} for t={global_t}")
        if order == 0: return spline.bezier_points[0].co.copy() # Fallback
        else: return Vector((0,0,0))

    segment_points = get_bezier_segment_data(spline, segment_index)
    if segment_points is None:
        print(f"Warning: Could not get segment data for index {segment_index}")
        # Fallback if segment data fails for some reason
        if order == 0:
            # Try to return endpoint coordinate based on index
            pt_idx = min(segment_index + 1, len(spline.bezier_points)-1)
            return spline.bezier_points[pt_idx].co.copy()
        else: return Vector((0,0,0))
    return evaluate_bezier_segment(*segment_points, local_t, order)

def get_interpolated_tilt(spline: bpy.types.Spline, global_t: float) -> float:
    """Gets the linearly interpolated tilt value at global t."""
    segment_index, local_t = get_spline_parameter_info(spline, global_t)

    num_points = len(spline.bezier_points)
    if num_points == 0: return 0.0
    if num_points == 1: return spline.bezier_points[0].tilt

    # Clamp segment index for safety if mapping failed near ends
    segment_index = max(0, min(segment_index, num_points - 2))

    start_point = spline.bezier_points[segment_index]
    end_point = spline.bezier_points[segment_index + 1]
    tilt0 = start_point.tilt
    tilt1 = end_point.tilt
    interpolated_tilt = tilt0 * (1.0 - local_t) + tilt1 * local_t
    return interpolated_tilt

def get_interpolated_point_scale(spline: bpy.types.Spline, global_t: float, point_scales_list: list) -> Vector:
    """
    Gets the linearly interpolated (sx, sy, sz) scale vector at global t.
    point_scales_list is the custom property from curve_data, e.g., [[sx,sy,sz], [sx,sy,sz], ...].
    """
    if not point_scales_list:
        return Vector((1.0, 1.0, 1.0)) # Default scale if no data

    num_points = len(spline.bezier_points)
    if num_points == 0:
        return Vector((1.0, 1.0, 1.0))
    
    if len(point_scales_list) != num_points:
        # Data mismatch, return default. Should be ensured by the operator writing it.
        print(f"Warning: Mismatch between bezier points ({num_points}) and scale data ({len(point_scales_list)})")
        return Vector((1.0, 1.0, 1.0))

    if num_points == 1:
        return Vector(point_scales_list[0])

    segment_index, local_t = get_spline_parameter_info(spline, global_t) # Your existing helper

    # Clamp segment index for safety if mapping failed near ends
    segment_index = max(0, min(segment_index, num_points - 2))

    scale0_list = point_scales_list[segment_index]
    scale1_list = point_scales_list[segment_index + 1]

    s0 = Vector(scale0_list) # sx, sy, sz for start of segment
    s1 = Vector(scale1_list) # sx, sy, sz for end of segment

    # Linear interpolation for each component
    interpolated_scale_vec = s0.lerp(s1, local_t)
    
    return interpolated_scale_vec




# --- RMF Calculation Function ---
def calculate_rmf_frames(spline: bpy.types.Spline, num_steps: int):
    """
    Calculates a sequence of Rotation Minimizing Frames along a Bezier spline
    using the Double Reflection method.

    Args:
        spline: The bpy.types.Spline object (must be BEZIER).
        num_steps: Number of frames to calculate along the curve (>= 2).

    Returns:
        list: A list of tuples, where each tuple represents a frame:
              (position_local, tangent_local, normal_local, global_t)
              Returns empty list if calculation fails.
    """
    if not spline or not spline.bezier_points or spline.type != 'BEZIER' or num_steps < 2:
        print("Error: Invalid input for RMF calculation.")
        return []

    frames = []
    step_size = 1.0 / (num_steps - 1)

    # --- Calculate Data for All Steps ---
    curve_data = []
    for i in range(num_steps):
        t_glob = i * step_size
        # Ensure t_glob is exactly 1.0 at the end
        if i == num_steps - 1: t_glob = 1.0

        pos = evaluate_spline(spline, t_glob, order=0)
        tan = evaluate_spline(spline, t_glob, order=1)

        # Normalize tangent, handle zero tangent robustly
        tan_len_sq = tan.length_squared
        if tan_len_sq > 1e-12:
            tan.normalize()
        else: # Zero tangent case
            # If not the first point, use previous tangent?
            if i > 0:
                tan = curve_data[i-1]['tan'].copy()
            else: # First point has zero tangent
                # Try direction to next point if possible
                if num_steps > 1:
                    pos_next = evaluate_spline(spline, step_size, order=0)
                    tan = pos_next - pos
                    if tan.length_squared > 1e-12:
                        tan.normalize()
                    else: tan = Vector((0,0,1)) # Final fallback
                else: # Only one step requested?
                     tan = Vector((0,0,1))

        curve_data.append({'t': t_glob, 'pos': pos, 'tan': tan})


    # --- Initialize the First Frame ---
    t0 = curve_data[0]['tan']
    p0 = curve_data[0]['pos']

    # Choose an initial normal perpendicular to t0 using a stable method
    up_vector = Vector((0.0, 0.0, 1.0))
    if abs(t0.dot(up_vector)) > 0.999: # Tangent is aligned with Z
        up_vector = Vector((0.0, 1.0, 0.0)) # Use Y instead

    r0 = t0.cross(up_vector)
    if r0.length_squared < 1e-12: # If still parallel (e.g., T aligned with Y too)
        up_vector = Vector((1.0, 0.0, 0.0)) # Use X
        r0 = t0.cross(up_vector)
        # If still zero, tangent must be zero (handled above), but check again
        if r0.length_squared < 1e-12:
             # Find *any* orthogonal vector if T is non-zero
             if t0.length_squared > 1e-12:
                 r0 = t0.orthogonal()
             else: # Should not happen if fallback above worked
                 r0 = Vector((1.0, 0.0, 0.0)) # Absolute fallback

    r0.normalize() # This is the initial Binormal B0

    # Initial Normal N0 = T0 x B0
    s0 = t0.cross(r0).normalized()
    # Now r0 is the first *Normal* vector (consistent with paper/code)
    r0 = s0

    frames.append((p0, t0, r0, curve_data[0]['t'])) # Store (pos, tan, normal, t)


    # --- Iterate using Double Reflection Method ---
    for i in range(num_steps - 1):
        p_i = curve_data[i]['pos']
        t_i = curve_data[i]['tan']
        r_i = frames[i][2] # The reference vector (normal) from previous frame

        p_i1 = curve_data[i+1]['pos']
        t_i1 = curve_data[i+1]['tan']

        v1 = p_i1 - p_i
        c1_sq = v1.length_squared

        if c1_sq < 1e-15:
            r_i1 = r_i.copy() # Reuse previous normal if points are coincident
        else:
            # Reflect previous reference vector r_i across plane with normal v1
            ri_reflected_v1 = r_i - (2.0 / c1_sq) * v1.dot(r_i) * v1
            ti_reflected_v1 = t_i - (2.0 / c1_sq) * v1.dot(t_i) * v1
            
            v2 = t_i1 - ti_reflected_v1
            c2_sq = v2.length_squared

            r_i1 = ri_reflected_v1 - (2.0 / c2_sq) * v2.dot(ri_reflected_v1) * v2

        r_i1.normalize() # Ensure unit length

        frames.append((p_i1, t_i1, r_i1, curve_data[i+1]['t']))

    return frames
    

# --- Radius Array Initialization ---
def ensure_radius_array(curve_data: bpy.types.Curve) -> bool:
    """
    Ensures the custom radius array property exists on the curve data
    and matches the number of Bezier points. Initializes or corrects it
    with default values (1.0) if necessary.

    Args:
        curve_data: The bpy.types.Curve data block to check/modify.

    Returns:
        bool: True if the array exists and is valid after the check, False otherwise.
    """
    if not curve_data or not curve_data.splines or not curve_data.splines[0].bezier_points:
        print("Error: Cannot ensure radius array - Invalid curve data or no Bezier points.")
        return False

    prop_key = cvars.PROP_RADIUS_ARRAY
    spline = curve_data.splines[0]
    num_pts = len(spline.bezier_points)

    try:
        if prop_key in curve_data:
            # Property exists, check length
            current_array = curve_data[prop_key]
            if len(current_array) == num_pts:
                # print(f"Radius array already exists and length matches ({num_pts}).") # Debug
                return True # Array exists and is correct length
            else:
                # Length mismatch, recreate with default values
                print(f"Warning: Radius array length mismatch ({len(current_array)} vs {num_pts}). Recreating.")
                curve_data[prop_key] = [1.0] * num_pts
                return True
        else:
            # Property does not exist, create it
            print(f"Initializing radius array for '{curve_data.name}' with {num_pts} points.")
            curve_data[prop_key] = [1.0] * num_pts
            return True
    except Exception as e:
        print(f"Error ensuring radius array for '{curve_data.name}': {e}")
        # Attempt to remove potentially corrupted property
        if prop_key in curve_data:
            del curve_data[prop_key]
        return False