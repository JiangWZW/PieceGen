# <pep8 compliant>
# -*- coding: utf-8 -*-
# File: common_imports.py
# Contains common standard and optional imports for the addon.

# --- Standard Imports ---
import bpy
import bmesh
import math
import mathutils
import base64
import zlib
from mathutils import Vector, Matrix, Quaternion

# --- Optional Imports ---
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False
    # Warning about missing numpy can be printed during registration in __init__.py
