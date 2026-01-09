#!/usr/bin/env python3
"""
SO-101 Forward Kinematics Implementation
=========================================

This module implements forward kinematics for the SO-101 robot arm from scratch,
using the product of exponentials (Lie groups) approach.

The SO-101 arm has 5 degrees of freedom plus a gripper:
    - Joint 1 (θ1): shoulder_pan  - rotates about Z-axis
    - Joint 2 (θ2): shoulder_lift - rotates about Y-axis
    - Joint 3 (θ3): elbow_flex    - rotates about Y-axis
    - Joint 4 (θ4): wrist_flex    - rotates about Y-axis
    - Joint 5 (θ5): wrist_roll    - rotates about X-axis
    - Joint 6: gripper (not used for FK of tool position)

Kinematic Parameters (from URDF):
    - Base to Joint 1: x = 0.0388353, z = 0.0624
    - Joint 1 offset: z = -0.0303992
    - Joint 1 to Joint 2: z = -0.0542
    - Joint 2 to Joint 3: z = 0.11257, then z = 0.028
    - Joint 3 to Joint 4: x = 0.1349
    - Joint 4 to Joint 5: x = 0.0611
    - Joint 5 to Tool: x = 0.1034

Author: SO-101 Robotics Course
License: MIT
"""

import numpy as np
from typing import Tuple, Dict

# ==============================================================================
# ROTATION MATRICES (Elementary Rotations)
# ==============================================================================
# These are the building blocks of 3D rotations.
# Each function returns a 3x3 rotation matrix for rotation about the specified axis.

def Rx(theta_deg: float) -> np.ndarray:
    """
    Rotation matrix about the X-axis.
    
    Args:
        theta_deg: Rotation angle in degrees
        
    Returns:
        3x3 rotation matrix
        
    Mathematical form:
        Rx(θ) = | 1    0       0    |
                | 0  cos(θ) -sin(θ) |
                | 0  sin(θ)  cos(θ) |
    """
    theta_rad = np.deg2rad(theta_deg)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])


def Ry(theta_deg: float) -> np.ndarray:
    """
    Rotation matrix about the Y-axis.
    
    Args:
        theta_deg: Rotation angle in degrees
        
    Returns:
        3x3 rotation matrix
        
    Mathematical form:
        Ry(θ) = |  cos(θ)  0  sin(θ) |
                |    0     1    0    |
                | -sin(θ)  0  cos(θ) |
    """
    theta_rad = np.deg2rad(theta_deg)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def Rz(theta_deg: float) -> np.ndarray:
    """
    Rotation matrix about the Z-axis.
    
    Args:
        theta_deg: Rotation angle in degrees
        
    Returns:
        3x3 rotation matrix
        
    Mathematical form:
        Rz(θ) = | cos(θ) -sin(θ)  0 |
                | sin(θ)  cos(θ)  0 |
                |   0       0     1 |
    """
    theta_rad = np.deg2rad(theta_deg)
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


# ==============================================================================
# HOMOGENEOUS TRANSFORMATION UTILITIES
# ==============================================================================

def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """
    Create a 4x4 homogeneous transformation matrix from rotation and translation.
    
    Args:
        rotation: 3x3 rotation matrix
        translation: 3-element translation vector (x, y, z)
        
    Returns:
        4x4 homogeneous transformation matrix
        
    The homogeneous transformation matrix has the form:
        T = | R   t |
            | 0   1 |
    where R is the 3x3 rotation matrix and t is the 3x1 translation vector.
    """
    T = np.eye(4)
    T[0:3, 0:3] = rotation
    T[0:3, 3] = translation
    return T


def translation_matrix(x: float, y: float, z: float) -> np.ndarray:
    """
    Create a pure translation transformation matrix.
    
    Args:
        x, y, z: Translation along each axis
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    return make_transform(np.eye(3), np.array([x, y, z]))


def rotation_matrix_x(theta_deg: float) -> np.ndarray:
    """Create a 4x4 transformation matrix for rotation about X-axis."""
    return make_transform(Rx(theta_deg), np.zeros(3))


def rotation_matrix_y(theta_deg: float) -> np.ndarray:
    """Create a 4x4 transformation matrix for rotation about Y-axis."""
    return make_transform(Ry(theta_deg), np.zeros(3))


def rotation_matrix_z(theta_deg: float) -> np.ndarray:
    """Create a 4x4 transformation matrix for rotation about Z-axis."""
    return make_transform(Rz(theta_deg), np.zeros(3))


# ==============================================================================
# SO-101 KINEMATIC PARAMETERS
# ==============================================================================
# These values are extracted from the SO-101 URDF file.
# See the kinematic diagram for visual reference.

class SO101Params:
    """SO-101 robot kinematic parameters."""
    
    # Base to Joint 1 frame
    BASE_TO_J1_X = 0.0388353
    BASE_TO_J1_Z = 0.0624
    
    # Joint 1 (shoulder_pan) offset
    J1_OFFSET_Z = -0.0303992
    
    # Joint 1 to Joint 2 (shoulder_lift)
    J1_TO_J2_Z = -0.0542
    
    # Joint 2 to Joint 3 (elbow_flex)
    J2_TO_J3_Z = 0.11257
    J3_OFFSET_Z = 0.028
    
    # Joint 3 to Joint 4 (wrist_flex)
    J3_TO_J4_X = 0.1349
    
    # Joint 4 to Joint 5 (wrist_roll)
    J4_TO_J5_X = 0.0611
    
    # Joint 5 to Tool frame
    J5_TO_TOOL_X = 0.1034


# ==============================================================================
# INDIVIDUAL TRANSFORMATION FUNCTIONS
# ==============================================================================
# Each function computes the transformation from one frame to the next.
# These are composed to get the full forward kinematics.

def get_gw1(theta1_deg: float) -> np.ndarray:
    """
    Compute transformation from World frame to Joint 1 frame.
    
    This transformation includes:
    1. Translation from world origin to joint 1 location
    2. Fixed rotation to align frames (180° about Z, then 180° about X)
    3. Joint 1 rotation (shoulder_pan) about Z-axis
    
    Args:
        theta1_deg: Joint 1 angle (shoulder_pan) in degrees
        
    Returns:
        4x4 transformation matrix T_w1
    """
    # Translation from world frame to joint 1
    displacement = np.array([SO101Params.BASE_TO_J1_X, 0.0, SO101Params.BASE_TO_J1_Z])
    
    # Fixed frame alignment (from URDF): rotate 180° about Z, then 180° about X
    # This aligns the world frame with the joint 1 frame convention
    fixed_rotation = Rz(180) @ Rx(180)
    
    # Joint 1 rotation about Z-axis
    joint_rotation = Rz(theta1_deg)
    
    # Combined rotation
    rotation = fixed_rotation @ joint_rotation
    
    return make_transform(rotation, displacement)


def get_g12(theta2_deg: float) -> np.ndarray:
    """
    Compute transformation from Joint 1 frame to Joint 2 frame.
    
    This transformation includes:
    1. Translation along Z from J1 to J2
    2. Joint 2 rotation (shoulder_lift) about Y-axis
    
    Args:
        theta2_deg: Joint 2 angle (shoulder_lift) in degrees
        
    Returns:
        4x4 transformation matrix T_12
    """
    # Translation from joint 1 to joint 2 (combined offsets)
    total_z = SO101Params.J1_OFFSET_Z + SO101Params.J1_TO_J2_Z
    displacement = np.array([0.0, 0.0, total_z])
    
    # Joint 2 rotation about Y-axis
    rotation = Ry(theta2_deg)
    
    return make_transform(rotation, displacement)


def get_g23(theta3_deg: float) -> np.ndarray:
    """
    Compute transformation from Joint 2 frame to Joint 3 frame.
    
    This transformation includes:
    1. Translation along Z from J2 to J3
    2. Joint 3 rotation (elbow_flex) about Y-axis
    
    Args:
        theta3_deg: Joint 3 angle (elbow_flex) in degrees
        
    Returns:
        4x4 transformation matrix T_23
    """
    # Translation from joint 2 to joint 3
    total_z = SO101Params.J2_TO_J3_Z + SO101Params.J3_OFFSET_Z
    displacement = np.array([0.0, 0.0, total_z])
    
    # Joint 3 rotation about Y-axis
    rotation = Ry(theta3_deg)
    
    return make_transform(rotation, displacement)


def get_g34(theta4_deg: float) -> np.ndarray:
    """
    Compute transformation from Joint 3 frame to Joint 4 frame.
    
    This transformation includes:
    1. Translation along X from J3 to J4
    2. Joint 4 rotation (wrist_flex) about Y-axis
    
    Args:
        theta4_deg: Joint 4 angle (wrist_flex) in degrees
        
    Returns:
        4x4 transformation matrix T_34
    """
    # Translation from joint 3 to joint 4 (along X in current frame)
    displacement = np.array([SO101Params.J3_TO_J4_X, 0.0, 0.0])
    
    # Joint 4 rotation about Y-axis
    rotation = Ry(theta4_deg)
    
    return make_transform(rotation, displacement)


def get_g45(theta5_deg: float) -> np.ndarray:
    """
    Compute transformation from Joint 4 frame to Joint 5 frame.
    
    This transformation includes:
    1. Translation along X from J4 to J5
    2. Joint 5 rotation (wrist_roll) about X-axis
    
    Args:
        theta5_deg: Joint 5 angle (wrist_roll) in degrees
        
    Returns:
        4x4 transformation matrix T_45
    """
    # Translation from joint 4 to joint 5
    displacement = np.array([SO101Params.J4_TO_J5_X, 0.0, 0.0])
    
    # Joint 5 rotation about X-axis (wrist roll)
    rotation = Rx(theta5_deg)
    
    return make_transform(rotation, displacement)


def get_g5t() -> np.ndarray:
    """
    Compute transformation from Joint 5 frame to Tool frame.
    
    This is a fixed transformation (no joint variable).
    The tool frame is located at the center of the gripper.
    
    Returns:
        4x4 transformation matrix T_5t
    """
    # Translation from joint 5 to tool frame
    displacement = np.array([SO101Params.J5_TO_TOOL_X, 0.0, 0.0])
    
    # No rotation (identity)
    rotation = np.eye(3)
    
    return make_transform(rotation, displacement)


# ==============================================================================
# FORWARD KINEMATICS FUNCTION
# ==============================================================================

def get_forward_kinematics(joint_angles: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the forward kinematics for the SO-101 robot arm.
    
    This function computes the position and orientation of the tool frame
    given the joint angles by multiplying the individual transformation matrices:
    
        T_wt = T_w1 @ T_12 @ T_23 @ T_34 @ T_45 @ T_5t
    
    Args:
        joint_angles: Dictionary with joint names as keys and angles (degrees) as values.
                     Required keys: 'shoulder_pan', 'shoulder_lift', 'elbow_flex',
                                   'wrist_flex', 'wrist_roll'
                     Optional: 'gripper' (not used for FK)
    
    Returns:
        Tuple of (position, rotation):
            - position: 3-element numpy array [x, y, z] of tool position
            - rotation: 3x3 rotation matrix of tool orientation
            
    Example:
        >>> angles = {
        ...     'shoulder_pan': 0.0,
        ...     'shoulder_lift': 45.0,
        ...     'elbow_flex': -30.0,
        ...     'wrist_flex': 15.0,
        ...     'wrist_roll': 0.0,
        ...     'gripper': 50.0
        ... }
        >>> position, rotation = get_forward_kinematics(angles)
        >>> print(f"Tool position: {position}")
    """
    # Extract joint angles
    theta1 = joint_angles['shoulder_pan']
    theta2 = joint_angles['shoulder_lift']
    theta3 = joint_angles['elbow_flex']
    theta4 = joint_angles['wrist_flex']
    theta5 = joint_angles['wrist_roll']
    
    # Compute individual transformations
    gw1 = get_gw1(theta1)
    g12 = get_g12(theta2)
    g23 = get_g23(theta3)
    g34 = get_g34(theta4)
    g45 = get_g45(theta5)
    g5t = get_g5t()
    
    # Chain multiply to get world-to-tool transformation
    gwt = gw1 @ g12 @ g23 @ g34 @ g45 @ g5t
    
    # Extract position and rotation from the transformation matrix
    position = gwt[0:3, 3]
    rotation = gwt[0:3, 0:3]
    
    return position, rotation


def get_full_transform(joint_angles: Dict[str, float]) -> np.ndarray:
    """
    Get the full 4x4 homogeneous transformation matrix for the tool frame.
    
    Args:
        joint_angles: Dictionary of joint angles in degrees
        
    Returns:
        4x4 homogeneous transformation matrix
    """
    theta1 = joint_angles['shoulder_pan']
    theta2 = joint_angles['shoulder_lift']
    theta3 = joint_angles['elbow_flex']
    theta4 = joint_angles['wrist_flex']
    theta5 = joint_angles['wrist_roll']
    
    gw1 = get_gw1(theta1)
    g12 = get_g12(theta2)
    g23 = get_g23(theta3)
    g34 = get_g34(theta4)
    g45 = get_g45(theta5)
    g5t = get_g5t()
    
    return gw1 @ g12 @ g23 @ g34 @ g45 @ g5t


def get_intermediate_transforms(joint_angles: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Get all intermediate transformation matrices for visualization/debugging.
    
    This is useful for visualizing the coordinate frames at each joint.
    
    Args:
        joint_angles: Dictionary of joint angles in degrees
        
    Returns:
        Dictionary with frame names as keys and 4x4 transforms as values
    """
    theta1 = joint_angles['shoulder_pan']
    theta2 = joint_angles['shoulder_lift']
    theta3 = joint_angles['elbow_flex']
    theta4 = joint_angles['wrist_flex']
    theta5 = joint_angles['wrist_roll']
    
    gw1 = get_gw1(theta1)
    g12 = get_g12(theta2)
    g23 = get_g23(theta3)
    g34 = get_g34(theta4)
    g45 = get_g45(theta5)
    g5t = get_g5t()
    
    transforms = {
        'world': np.eye(4),
        'joint1': gw1,
        'joint2': gw1 @ g12,
        'joint3': gw1 @ g12 @ g23,
        'joint4': gw1 @ g12 @ g23 @ g34,
        'joint5': gw1 @ g12 @ g23 @ g34 @ g45,
        'tool': gw1 @ g12 @ g23 @ g34 @ g45 @ g5t
    }
    
    return transforms


# ==============================================================================
# WORKSPACE ANALYSIS
# ==============================================================================

def compute_workspace_sample(n_samples: int = 1000) -> np.ndarray:
    """
    Sample the robot workspace by computing FK for random joint configurations.
    
    Args:
        n_samples: Number of random configurations to sample
        
    Returns:
        Array of shape (n_samples, 3) containing tool positions
    """
    # Joint limits (approximate, in degrees)
    joint_limits = {
        'shoulder_pan': (-180, 180),
        'shoulder_lift': (-90, 90),
        'elbow_flex': (-135, 135),
        'wrist_flex': (-90, 90),
        'wrist_roll': (-180, 180),
    }
    
    positions = []
    
    for _ in range(n_samples):
        # Generate random joint configuration
        joint_angles = {
            joint: np.random.uniform(low, high)
            for joint, (low, high) in joint_limits.items()
        }
        joint_angles['gripper'] = 0  # Not used
        
        # Compute FK
        position, _ = get_forward_kinematics(joint_angles)
        positions.append(position)
    
    return np.array(positions)


# ==============================================================================
# MAIN - Test the implementation
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SO-101 Forward Kinematics Test")
    print("=" * 60)
    
    # Test configuration
    test_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    print("\n1. Home Configuration (all zeros):")
    print(f"   Joint angles: {test_config}")
    position, rotation = get_forward_kinematics(test_config)
    print(f"   Tool position: x={position[0]:.4f}, y={position[1]:.4f}, z={position[2]:.4f}")
    
    # Test with non-zero configuration
    test_config2 = {
        'shoulder_pan': -45.0,
        'shoulder_lift': 45.0,
        'elbow_flex': -45.0,
        'wrist_flex': 90.0,
        'wrist_roll': 0.0,
        'gripper': 10.0
    }
    
    print("\n2. Test Configuration:")
    print(f"   Joint angles: {test_config2}")
    position2, rotation2 = get_forward_kinematics(test_config2)
    print(f"   Tool position: x={position2[0]:.4f}, y={position2[1]:.4f}, z={position2[2]:.4f}")
    
    # Print the full transformation matrix
    print("\n3. Full Transformation Matrix (Test Config 2):")
    T = get_full_transform(test_config2)
    print(T)
    
    # Print intermediate frames
    print("\n4. Intermediate Frame Positions:")
    transforms = get_intermediate_transforms(test_config2)
    for frame_name, transform in transforms.items():
        pos = transform[0:3, 3]
        print(f"   {frame_name:8s}: x={pos[0]:7.4f}, y={pos[1]:7.4f}, z={pos[2]:7.4f}")
    
    print("\n" + "=" * 60)
    print("Forward Kinematics Test Complete!")
    print("=" * 60)
