from mathutils import Vector, Matrix
from .common_imports import *
from .common_vars import *

# --- Helper Functions ---
def pack_verts_to_string(verts_coords_list: list[Vector]):
    """Converts list of Vector coords to compressed base64 string."""
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