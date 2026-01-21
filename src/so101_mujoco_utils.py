#!/usr/bin/env python3
"""
SO-101 MuJoCo Utilities
========================

Helper functions for simulating the SO-101 robot arm in MuJoCo.

This module provides:
    - Joint configuration management (degrees <-> MuJoCo units)
    - Position control functions
    - Motion interpolation (move_to_pose)
    - Visualization helpers

Author: SO-101 Robotics Course
License: MIT
"""

import time
import numpy as np
from typing import Dict, Optional

# MuJoCo will be imported when available
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not available. Install with: pip install mujoco")


# ==============================================================================
# JOINT NAME MAPPING
# ==============================================================================
# Maps our user-friendly joint names to MuJoCo actuator names

JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift', 
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper'
]

# MuJoCo actuator names (must match the XML model)
MUJOCO_ACTUATOR_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex', 
    'wrist_flex',
    'wrist_roll',
    'gripper'
]


# ==============================================================================
# UNIT CONVERSION
# ==============================================================================
# The SO-101 uses Feetech STS3215 servos
# MuJoCo typically uses radians, but our FK/IK use degrees
# The gripper uses a different scale (servo position units)

def degrees_to_mujoco(joint_angles: Dict[str, float]) -> Dict[str, float]:
    """
    Convert joint angles from degrees to MuJoCo units (radians for arm, servo units for gripper).
    
    Args:
        joint_angles: Dictionary with angles in degrees
        
    Returns:
        Dictionary with angles in MuJoCo units
    """
    mujoco_angles = {}
    
    for joint in JOINT_NAMES:
        if joint == 'gripper':
            # Gripper uses servo position units (0-100 typically)
            mujoco_angles[joint] = joint_angles.get(joint, 0.0)
        else:
            # Convert degrees to radians for arm joints
            mujoco_angles[joint] = np.deg2rad(joint_angles.get(joint, 0.0))
    
    return mujoco_angles


def mujoco_to_degrees(mujoco_angles: Dict[str, float]) -> Dict[str, float]:
    """
    Convert joint angles from MuJoCo units to degrees.
    
    Args:
        mujoco_angles: Dictionary with angles in MuJoCo units
        
    Returns:
        Dictionary with angles in degrees
    """
    degree_angles = {}
    
    for joint in JOINT_NAMES:
        if joint == 'gripper':
            degree_angles[joint] = mujoco_angles.get(joint, 0.0)
        else:
            degree_angles[joint] = np.rad2deg(mujoco_angles.get(joint, 0.0))
    
    return degree_angles


# ==============================================================================
# INITIAL POSE SETTING
# ==============================================================================

def set_initial_pose(data, joint_angles: Dict[str, float], use_degrees: bool = True):
    """
    Set the initial joint positions in MuJoCo data structure.
    
    This sets qpos directly (instantaneous, no simulation).
    Use this at the start of simulation before stepping.
    
    Args:
        data: MuJoCo data object (mjData)
        joint_angles: Dictionary of joint angles
        use_degrees: If True, input is in degrees; if False, already in MuJoCo units
    """
    if use_degrees:
        angles = degrees_to_mujoco(joint_angles)
    else:
        angles = joint_angles
    
    for i, joint in enumerate(JOINT_NAMES):
        if i < len(data.qpos):
            if joint == 'gripper':
                # FIXED SCALING
                data.qpos[i] = angles[joint] / 100.0 * 1.57 
            else:
                data.qpos[i] = angles[joint]


def get_current_pose(data, use_degrees: bool = True) -> Dict[str, float]:
    """
    Get the current joint positions from MuJoCo data.
    
    Args:
        data: MuJoCo data object (mjData)
        use_degrees: If True, return in degrees; if False, return MuJoCo units
        
    Returns:
        Dictionary of joint angles
    """
    angles = {}
    
    for i, joint in enumerate(JOINT_NAMES):
        if i < len(data.qpos):
            if joint == 'gripper':
                angles[joint] = data.qpos[i] / 0.04 * 100.0  # Scale from rad to 0-100
            else:
                angles[joint] = data.qpos[i]
    
    if use_degrees:
        return mujoco_to_degrees(angles)
    return angles


# ==============================================================================
# POSITION CONTROL
# ==============================================================================

def send_position_command(data, joint_angles: Dict[str, float], use_degrees: bool = True):
    """
    Send position commands to the MuJoCo actuators.
    
    This sets the control targets (ctrl) that the position controllers will track.
    
    Args:
        data: MuJoCo data object (mjData)
        joint_angles: Dictionary of target joint angles
        use_degrees: If True, input is in degrees
    """
    if use_degrees:
        angles = degrees_to_mujoco(joint_angles)
    else:
        angles = joint_angles
    
    for i, joint in enumerate(JOINT_NAMES):
        if i < len(data.ctrl):
            if joint == 'gripper':
                #data.ctrl[i] = angles[joint] / 100.0 * 0.04
                data.ctrl[i] = angles[joint] / 100.0 * 1.57 # adjusted for gripper
            else:
                data.ctrl[i] = angles[joint]


# ==============================================================================
# MOTION FUNCTIONS
# ==============================================================================

def hold_position(model, data, viewer, duration: float, joint_angles: Optional[Dict[str, float]] = None):
    """
    Hold the current (or specified) position for a given duration.
    
    Args:
        model: MuJoCo model object (mjModel)
        data: MuJoCo data object (mjData)
        viewer: MuJoCo viewer object
        duration: Time to hold in seconds
        joint_angles: If provided, hold this configuration; otherwise hold current
    """
    if joint_angles is not None:
        send_position_command(data, joint_angles)
    
    start_time = time.time()
    
    while time.time() - start_time < duration:
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Update viewer
        viewer.sync()
        
        # Small sleep to prevent CPU overload
        time.sleep(0.001)


def move_to_pose(
    model, 
    data, 
    viewer, 
    target_angles: Dict[str, float],
    duration: float,
    start_angles: Optional[Dict[str, float]] = None,
    use_degrees: bool = True
):
    """
    Smoothly interpolate from current/start position to target position.
    
    Uses cubic interpolation for smooth acceleration/deceleration.
    
    Args:
        model: MuJoCo model object (mjModel)
        data: MuJoCo data object (mjData)
        viewer: MuJoCo viewer object
        target_angles: Target joint configuration
        duration: Time for motion in seconds
        start_angles: Starting configuration (if None, use current)
        use_degrees: If True, angles are in degrees
    """
    # Get starting configuration
    if start_angles is None:
        start_angles = get_current_pose(data, use_degrees=use_degrees)
    
    start_time = time.time()
    
    while True:
        elapsed = time.time() - start_time
        
        if elapsed >= duration:
            # Ensure we reach exactly the target
            send_position_command(data, target_angles, use_degrees=use_degrees)
            mujoco.mj_step(model, data)
            viewer.sync()
            break
        
        # Normalized time [0, 1]
        t = elapsed / duration
        
        # Cubic interpolation: smooth start and end (ease-in-out)
        # s(t) = 3t² - 2t³
        s = 3 * t**2 - 2 * t**3
        
        # Interpolate each joint
        interp_angles = {}
        for joint in JOINT_NAMES:
            start_val = start_angles.get(joint, 0.0)
            target_val = target_angles.get(joint, 0.0)
            interp_angles[joint] = start_val + s * (target_val - start_val)
        
        # Send interpolated command
        send_position_command(data, interp_angles, use_degrees=use_degrees)
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Update viewer
        viewer.sync()
        
        # Small sleep
        time.sleep(0.001)


def move_trajectory(
    model,
    data, 
    viewer,
    waypoints: list,
    durations: list,
    use_degrees: bool = True
):
    """
    Execute a trajectory through multiple waypoints.
    
    Args:
        model: MuJoCo model object
        data: MuJoCo data object
        viewer: MuJoCo viewer object
        waypoints: List of joint angle dictionaries
        durations: List of durations between waypoints (len = len(waypoints) - 1)
        use_degrees: If True, angles are in degrees
    """
    if len(durations) != len(waypoints) - 1:
        raise ValueError("Number of durations must be one less than number of waypoints")
    
    for i in range(len(waypoints) - 1):
        move_to_pose(
            model, data, viewer,
            target_angles=waypoints[i + 1],
            duration=durations[i],
            start_angles=waypoints[i],
            use_degrees=use_degrees
        )


# ==============================================================================
# VISUALIZATION HELPERS
# ==============================================================================

def show_cylinder(
    viewer, 
    position: np.ndarray, 
    rotation: np.ndarray,
    geom_idx: int = 0,
    radius: float = 0.0245, 
    halfheight: float = 0.05, 
    rgba: list = [1, 0, 0, 1]
):
    """
    Add a cylinder visualization to the MuJoCo viewer.
    
    The cylinder is aligned with the Z-axis of the given orientation.
    
    Args:
        viewer: MuJoCo viewer object
        position: 3D position [x, y, z]
        rotation: 3x3 rotation matrix
        geom_idx: Index in user_scn.geoms array
        radius: Cylinder radius
        halfheight: Half of cylinder height
        rgba: Color [R, G, B, A]
    """
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_idx],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius, halfheight, 0],
        pos=np.array(position, dtype=np.float64),
        mat=np.array(rotation, dtype=np.float64).flatten(),
        rgba=np.array(rgba, dtype=np.float32)
    )
    viewer.user_scn.ngeom = max(viewer.user_scn.ngeom, geom_idx + 1)
    viewer.sync()


def show_cube(
    viewer,
    position: np.ndarray,
    rotation: np.ndarray,
    geom_idx: int = 0,
    halfwidth: float = 0.013,
    rgba: list = [1, 0, 0, 0.5]
):
    """
    Add a cube visualization to the MuJoCo viewer.
    
    Args:
        viewer: MuJoCo viewer object
        position: 3D position [x, y, z]
        rotation: 3x3 rotation matrix
        geom_idx: Index in user_scn.geoms array
        halfwidth: Half-width of the cube
        rgba: Color [R, G, B, A]
    """
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_idx],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[halfwidth, halfwidth, halfwidth],
        pos=np.array(position, dtype=np.float64),
        mat=np.array(rotation, dtype=np.float64).flatten(),
        rgba=np.array(rgba, dtype=np.float32)
    )
    viewer.user_scn.ngeom = max(viewer.user_scn.ngeom, geom_idx + 1)
    viewer.sync()


def show_sphere(
    viewer,
    position: np.ndarray,
    geom_idx: int = 0,
    radius: float = 0.01,
    rgba: list = [0, 1, 0, 1]
):
    """
    Add a sphere visualization to the MuJoCo viewer.
    
    Args:
        viewer: MuJoCo viewer object
        position: 3D position [x, y, z]
        geom_idx: Index in user_scn.geoms array
        radius: Sphere radius
        rgba: Color [R, G, B, A]
    """
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[geom_idx],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.array(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.array(rgba, dtype=np.float32)
    )
    viewer.user_scn.ngeom = max(viewer.user_scn.ngeom, geom_idx + 1)
    viewer.sync()


def show_frame(
    viewer,
    position: np.ndarray,
    rotation: np.ndarray,
    start_geom_idx: int = 0,
    axis_length: float = 0.05,
    axis_radius: float = 0.003
):
    """
    Show a coordinate frame (3 colored axes) at the given pose.
    
    X-axis: Red
    Y-axis: Green
    Z-axis: Blue
    
    Args:
        viewer: MuJoCo viewer object
        position: 3D position of frame origin
        rotation: 3x3 rotation matrix
        start_geom_idx: Starting index in user_scn.geoms
        axis_length: Length of each axis
        axis_radius: Radius of axis cylinders
    """
    colors = [
        [1, 0, 0, 1],  # X: Red
        [0, 1, 0, 1],  # Y: Green
        [0, 0, 1, 1],  # Z: Blue
    ]
    
    for i in range(3):
        # Get axis direction in world frame
        axis_dir = rotation[:, i]
        
        # Position is offset by half the length along the axis
        axis_pos = position + 0.5 * axis_length * axis_dir
        
        # Create rotation matrix that aligns Z with the axis direction
        z_axis = axis_dir
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross([0, 0, 1], z_axis)
        else:
            x_axis = np.cross([0, 1, 0], z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        axis_rot = np.column_stack([x_axis, y_axis, z_axis])
        
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[start_geom_idx + i],
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[axis_radius, axis_length / 2, 0],
            pos=np.array(axis_pos, dtype=np.float64),
            mat=np.array(axis_rot, dtype=np.float64).flatten(),
            rgba=np.array(colors[i], dtype=np.float32)
        )
    
    viewer.user_scn.ngeom = max(viewer.user_scn.ngeom, start_geom_idx + 3)
    viewer.sync()


def clear_visualizations(viewer):
    """Clear all user-added visualizations."""
    viewer.user_scn.ngeom = 0
    viewer.sync()


# ==============================================================================
# MAIN - Test utilities
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SO-101 MuJoCo Utilities Test")
    print("=" * 60)
    
    # Test unit conversions
    test_angles = {
        'shoulder_pan': 45.0,
        'shoulder_lift': -30.0,
        'elbow_flex': 60.0,
        'wrist_flex': 15.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    
    print("\n1. Unit Conversion Test:")
    print(f"   Input (degrees): {test_angles}")
    
    mujoco_units = degrees_to_mujoco(test_angles)
    print(f"   MuJoCo units: {mujoco_units}")
    
    back_to_degrees = mujoco_to_degrees(mujoco_units)
    print(f"   Back to degrees: {back_to_degrees}")
    
    print("\n" + "=" * 60)
    print("Utilities module loaded successfully!")
    print("=" * 60)
