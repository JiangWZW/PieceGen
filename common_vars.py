# <pep8 compliant>
# -*- coding: utf-8 -*-
# File: common_vars.py
# Contains shared constants and global state variables for the addon.

import bpy

# --- Persistent Mesh Properties ---
# Custom Property Keys (used on the mesh object being deformed)
PROP_ENABLED = "ccdg_enabled"             # (Boolean) Is realtime deform active?
PROP_CURVE_NAME = "ccdg_curve_name"       # (String) Name of the guide curve object
PROP_ORIG_VERTS = "ccdg_original_verts_b64" # (String) Packed original vertex data
PROP_HEIGHT = "ccdg_original_height"      # (Float) Original cylinder height used for mapping

class PieceGenPointScaleValues(bpy.types.PropertyGroup):
    scale: bpy.props.FloatVectorProperty(
        name="Scale Factors",
        description="SX, SY, SZ scale factors for this point",
        size=3,
        default=(1.0, 1.0, 1.0),
        subtype='XYZ' # Or 'NONE' if you don't want XYZ labels in UI
    )  
PROP_RADIUS_ARRAY = "piecegen_radius_array" # DEPRECATED Deformation scale per curve point
   

# --- Global State ---
# Stores names of mesh objects being actively deformed by the handler.
MONITORED_MESH_OBJECTS = set()

# Cache for original vertex coordinates to avoid repeated unpacking in handler
# Key: mesh object name (str), Value: list[Vector]
original_coords_cache = {}

