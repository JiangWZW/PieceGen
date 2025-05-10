# <pep8 compliant>
# -*- coding: utf-8 -*-
# File: common_vars.py
# Contains shared constants and global state variables for the addon.

# --- Constants ---
# Custom Property Keys (used on the mesh object being deformed)
PROP_ENABLED = "ccdg_enabled"             # (Boolean) Is realtime deform active?
PROP_CURVE_NAME = "ccdg_curve_name"       # (String) Name of the guide curve object
PROP_ORIG_VERTS = "ccdg_original_verts_b64" # (String) Packed original vertex data
PROP_HEIGHT = "ccdg_original_height"      # (Float) Original cylinder height used for mapping
PROP_RADIUS_ARRAY = "piecegen_radius_array" # (IDP Array/List of Floats) Deformation scale per curve point

# --- Global State ---
# Stores names of mesh objects being actively deformed by the handler.
MONITORED_MESH_OBJECTS = set()

# Cache for original vertex coordinates to avoid repeated unpacking in handler
# Key: mesh object name (str), Value: list[Vector]
original_coords_cache = {}

