import bpy

def get_current_view_camera_object():
    """
    Retrieves the camera object that the current 3D View is looking through.

    Returns:
        bpy.types.Object or None: The camera object if in camera view, 
                                  otherwise None.
    """
    context = bpy.context

    # Check if the current area is a 3D View
    if context.area and context.area.type == 'VIEW_3D':
        space_view3d = context.space_data  # This is a bpy.types.SpaceView3D

        # Check if the view is in camera perspective mode
        if space_view3d.region_3d.view_perspective == 'PERSP':
            if space_view3d.lock_camera:
                # If the view is locked to the scene camera,
                # and there is a scene camera
                return context.scene.camera
            else:
                # If the view is not locked, it uses its own camera setting.
                # This space_view3d.camera can be None if no specific camera is assigned,
                # in which case it might implicitly use the scene camera or behave
                # as if no camera is active for the view despite being in 'CAMERA' perspective.
                return space_view3d.camera
        else:
            # Not in camera view (e.g., perspective or orthographic user view)
            return None
    else:
        # Current context is not a 3D View
        return None
    

def get_viewport_camera_world_position():
    """
    Calculates the world space position of the current 3D Viewport's viewpoint.
    This applies whether in camera view, perspective, or orthographic mode.

    Returns:
        mathutils.Vector or None: The world space position (X, Y, Z) of the viewpoint,
                                  or None if 3D view data cannot be accessed.
    """
    context = bpy.context
    
    # Ensure we are in a 3D View context and have the necessary data
    if context.area and context.area.type == 'VIEW_3D':
        space_data = context.space_data  # bpy.types.SpaceView3D
        if space_data and hasattr(space_data, 'region_3d'):
            region_3d = space_data.region_3d  # bpy.types.RegionView3D
            return region_3d.view_matrix.inverted().translation
    return None

def get_viewport_camera_matrices():
    """
    Returns:
        view, proj, view_proj matrices
    """
    context = bpy.context
    
    # Ensure we are in a 3D View context and have the necessary data
    if context.area and context.area.type == 'VIEW_3D':
        space_data = context.space_data  # bpy.types.SpaceView3D
        if space_data and hasattr(space_data, 'region_3d'):
            region_3d: bpy.types.RegionView3D = space_data.region_3d  # bpy.types.RegionView3D
            return region_3d.view_matrix, region_3d.window_matrix, region_3d.perspective_matrix # == window * view
    return None

def get_viewpoint_near_clip_distance():
    """
    Gets the near clipping distance for the current 3D viewport's viewpoint.

    Returns:
        float or None: The near clip distance, or None if it cannot be determined.
    """
    context = bpy.context

    if not (context.area and context.area.type == 'VIEW_3D'):
        print("Error: Current context is not a 3D View.")
        return None

    space_data = context.space_data  # This is a bpy.types.SpaceView3D
    if not hasattr(space_data, 'region_3d') or not space_data.region_3d:
        print("Error: Could not access 3D region data.")
        return None
        
    region_3d = space_data.region_3d

    if region_3d.view_perspective == 'CAMERA':
        # View is looking through a Blender camera object
        camera_object = None
        if space_data.lock_camera and context.scene.camera:
            camera_object = context.scene.camera
        elif not space_data.lock_camera and space_data.camera:
            # View is using a specific camera assigned to it (not locked to scene)
            camera_object = space_data.camera
        
        if camera_object and camera_object.type == 'CAMERA':
            return camera_object.data.clip_start
        else:
            # Fallback to viewport settings if camera object not found or not a camera
            # This case should ideally not happen if view_perspective is 'CAMERA'
            # and a camera is properly set.
            print("Warning: In CAMERA view, but couldn't find valid camera object. Using viewport clip settings.")
            return space_data.clip_start
    else:
        # View is 'PERSP' (User Perspective) or 'ORTHO' (User Orthographic)
        # These views use the viewport's own clip_start setting
        return space_data.clip_start

