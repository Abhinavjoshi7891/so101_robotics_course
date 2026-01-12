#!/usr/bin/env python3
"""
SO-101 Forward Kinematics - Complete Implementation
====================================================

This module provides forward kinematics with three modes:

1. 'urdf_native' (RECOMMENDED): Uses actual URDF transforms with RPY values
   - Matches your actual so101.urdf file exactly
   - Use for RViz/ROS2

2. 'mujoco': Uses MuJoCo XML transforms with quaternion pre-rotations
   - For MuJoCo simulation compatibility
   
3. 'urdf': Simple URDF convention (legacy, may not match your URDF)
   - Kept for backward compatibility

Author: SO-101 Robotics Course
"""

import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class SO101Params:
    """SO-101 kinematic parameters (meters) - Reference values"""
    BASE_X = 0.0388353
    BASE_Z = 0.0624
    J1_Z = -0.0303992
    J1_TO_J2_Z = -0.0542
    J2_TO_J3_Z = 0.11257
    J3_OFFSET_Z = 0.028
    J3_TO_J4_X = 0.1349
    J4_TO_J5_X = 0.0611
    J5_TO_TOOL_X = 0.1034


# ==============================================================================
# ROTATION MATRICES
# ==============================================================================

def Rx(angle_deg: float) -> np.ndarray:
    """Rotation matrix around X-axis (degrees)"""
    th = np.deg2rad(angle_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(th), -np.sin(th)],
        [0, np.sin(th), np.cos(th)]
    ])


def Ry(angle_deg: float) -> np.ndarray:
    """Rotation matrix around Y-axis (degrees)"""
    th = np.deg2rad(angle_deg)
    return np.array([
        [np.cos(th), 0, np.sin(th)],
        [0, 1, 0],
        [-np.sin(th), 0, np.cos(th)]
    ])


def Rz(angle_deg: float) -> np.ndarray:
    """Rotation matrix around Z-axis (degrees)"""
    th = np.deg2rad(angle_deg)
    return np.array([
        [np.cos(th), -np.sin(th), 0],
        [np.sin(th), np.cos(th), 0],
        [0, 0, 1]
    ])


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert RPY (roll, pitch, yaw) in radians to rotation matrix.
    ROS/URDF convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    """
    Rx_mat = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    Ry_mat = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    Rz_mat = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    return Rz_mat @ Ry_mat @ Rx_mat


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Create 4x4 homogeneous transformation matrix"""
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T


# ==============================================================================
# MODE 1: URDF NATIVE (RECOMMENDED FOR RVIZ)
# ==============================================================================

def get_intermediate_transforms_urdf_native(joint_angles: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Compute FK using ACTUAL URDF transforms from so101.urdf.
    
    Each joint:
    - Has origin with XYZ + RPY that pre-rotates the frame
    - Rotates about local Z-axis
    
    This matches your actual URDF file exactly!
    """
    q1 = joint_angles.get('shoulder_pan', 0.0)
    q2 = joint_angles.get('shoulder_lift', 0.0)
    q3 = joint_angles.get('elbow_flex', 0.0)
    q4 = joint_angles.get('wrist_flex', 0.0)
    q5 = joint_angles.get('wrist_roll', 0.0)
    
    # World frame
    T_world = np.eye(4)
    
    # Joint 1: base → shoulder
    # URDF: <origin xyz="0.0207909 -0.0230745 0.0948817" rpy="-3.14159 6.03684e-16 1.5708" />
    T_base_shoulder = make_transform(
        rpy_to_matrix(-3.14159, 6.03684e-16, 1.5708),
        np.array([0.0207909, -0.0230745, 0.0948817])
    )
    T_j1_rot = make_transform(Rz(q1), np.array([0, 0, 0]))
    T_1 = T_world @ T_base_shoulder @ T_j1_rot
    
    # Joint 2: shoulder → upper_arm
    # URDF: <origin xyz="-0.0303992 -0.0182778 -0.0542" rpy="-1.5708 -1.5708 0" />
    T_shoulder_upper = make_transform(
        rpy_to_matrix(-1.5708, -1.5708, 0),
        np.array([-0.0303992, -0.0182778, -0.0542])
    )
    T_j2_rot = make_transform(Rz(q2), np.array([0, 0, 0]))
    T_2 = T_1 @ T_shoulder_upper @ T_j2_rot
    
    # Joint 3: upper_arm → lower_arm
    # URDF: <origin xyz="-0.11257 -0.028 2.46331e-16" rpy="-1.22818e-15 5.75928e-16 1.5708" />
    T_upper_lower = make_transform(
        rpy_to_matrix(-1.22818e-15, 5.75928e-16, 1.5708),
        np.array([-0.11257, -0.028, 0.0])
    )
    T_j3_rot = make_transform(Rz(q3), np.array([0, 0, 0]))
    T_3 = T_2 @ T_upper_lower @ T_j3_rot
    
    # Joint 4: lower_arm → wrist
    # URDF: <origin xyz="-0.1349 0.0052 1.65232e-16" rpy="3.2474e-15 2.86219e-15 -1.5708" />
    T_lower_wrist = make_transform(
        rpy_to_matrix(3.2474e-15, 2.86219e-15, -1.5708),
        np.array([-0.1349, 0.0052, 0.0])
    )
    T_j4_rot = make_transform(Rz(q4), np.array([0, 0, 0]))
    T_4 = T_3 @ T_lower_wrist @ T_j4_rot
    
    # Joint 5: wrist → gripper
    # URDF: <origin xyz="0 -0.0611 0.0181" rpy="1.5708 -9.38083e-08 3.14159" />
    T_wrist_gripper = make_transform(
        rpy_to_matrix(1.5708, -9.38083e-08, 3.14159),
        np.array([0, -0.0611, 0.0181])
    )
    T_j5_rot = make_transform(Rz(q5), np.array([0, 0, 0]))
    T_5 = T_4 @ T_wrist_gripper @ T_j5_rot
    
    # Note: T_5 represents the 'gripper' link frame in the URDF
    # Joint 6 (jaw) is attached to gripper, but we stop at gripper frame
    
    return {
        'joint1': T_1,
        'joint2': T_2,
        'joint3': T_3,
        'joint4': T_4,
        'joint5': T_5,  # This IS the 'gripper' frame!
        'tool': T_5     # Same as joint5
    }


# ==============================================================================
# MODE 2: SIMPLE URDF (LEGACY)
# ==============================================================================

def get_intermediate_transforms_urdf(joint_angles: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Simple URDF convention: offsets + explicit joint axes.
    Kept for backward compatibility but may not match your actual URDF.
    """
    q1 = joint_angles.get('shoulder_pan', 0.0)
    q2 = joint_angles.get('shoulder_lift', 0.0)
    q3 = joint_angles.get('elbow_flex', 0.0)
    q4 = joint_angles.get('wrist_flex', 0.0)
    q5 = joint_angles.get('wrist_roll', 0.0)
    
    T_world = np.eye(4)
    
    # Joint 1: Z-axis rotation
    T_base_to_j1 = make_transform(
        np.eye(3),
        np.array([SO101Params.BASE_X, 0, SO101Params.BASE_Z + SO101Params.J1_Z])
    )
    T_j1_rot = make_transform(Rz(q1), np.array([0, 0, 0]))
    T_1 = T_world @ T_base_to_j1 @ T_j1_rot
    
    # Joint 2: Y-axis rotation
    T_j1_to_j2 = make_transform(
        np.eye(3),
        np.array([0, 0, SO101Params.J1_TO_J2_Z])
    )
    T_j2_rot = make_transform(Ry(q2), np.array([0, 0, 0]))
    T_2 = T_1 @ T_j1_to_j2 @ T_j2_rot
    
    # Joint 3: Y-axis rotation
    T_j2_to_j3 = make_transform(
        np.eye(3),
        np.array([0, 0, SO101Params.J2_TO_J3_Z + SO101Params.J3_OFFSET_Z])
    )
    T_j3_rot = make_transform(Ry(q3), np.array([0, 0, 0]))
    T_3 = T_2 @ T_j2_to_j3 @ T_j3_rot
    
    # Joint 4: Y-axis rotation
    T_j3_to_j4 = make_transform(
        np.eye(3),
        np.array([SO101Params.J3_TO_J4_X, 0, 0])
    )
    T_j4_rot = make_transform(Ry(q4), np.array([0, 0, 0]))
    T_4 = T_3 @ T_j3_to_j4 @ T_j4_rot
    
    # Joint 5: X-axis rotation
    T_j4_to_j5 = make_transform(
        np.eye(3),
        np.array([SO101Params.J4_TO_J5_X, 0, 0])
    )
    T_j5_rot = make_transform(Rx(q5), np.array([0, 0, 0]))
    T_5 = T_4 @ T_j4_to_j5 @ T_j5_rot
    
    # Tool frame
    T_j5_to_tool = make_transform(
        np.eye(3),
        np.array([SO101Params.J5_TO_TOOL_X, 0, 0])
    )
    T_tool = T_5 @ T_j5_to_tool
    
    return {
        'joint1': T_1,
        'joint2': T_2,
        'joint3': T_3,
        'joint4': T_4,
        'joint5': T_5,
        'tool': T_tool
    }


# ==============================================================================
# MODE 3: MUJOCO
# ==============================================================================

def quat_to_rot(w, x, y, z):
    """Convert quaternion (w,x,y,z) to rotation matrix"""
    Nq = w*w + x*x + y*y + z*z
    if Nq < np.finfo(float).eps:
        return np.eye(3)
    s = 2.0 / Nq
    X, Y, Z = x*s, y*s, z*s
    wX, wY, wZ = w*X, w*Y, w*Z
    xX, xY, xZ = x*X, x*Y, x*Z
    yY, yZ = y*Y, y*Z
    zZ = z*Z
    
    return np.array([
        [1.0-(yY+zZ), xY-wZ, xZ+wY],
        [xY+wZ, 1.0-(xX+zZ), yZ-wX],
        [xZ-wY, yZ+wX, 1.0-(xX+yY)]
    ])


def make_tf_mujoco(pos, quat_wxyz):
    """Build transform from MuJoCo XML pos and quaternion"""
    T = np.eye(4)
    T[0:3, 3] = pos
    T[0:3, 0:3] = quat_to_rot(*quat_wxyz)
    return T


# MuJoCo XML transforms
TRANS_BASE_J1_MUJOCO = make_tf_mujoco(
    pos=[0.0388353, 0.0, 0.0624],
    quat_wxyz=[0.0, 0.0, -1.0, 0.0]
)

TRANS_J1_J2_MUJOCO = make_tf_mujoco(
    pos=[-0.0303992, -0.0182778, -0.0542],
    quat_wxyz=[0.5, -0.5, -0.5, -0.5]
)

TRANS_J2_J3_MUJOCO = make_tf_mujoco(
    pos=[-0.11257, -0.028, 0.0],
    quat_wxyz=[0.707107, 0.0, 0.0, 0.707107]
)

TRANS_J3_J4_MUJOCO = make_tf_mujoco(
    pos=[-0.1349, 0.0052, 0.0],
    quat_wxyz=[0.707107, 0.0, 0.0, -0.707107]
)

TRANS_J4_J5_MUJOCO = make_tf_mujoco(
    pos=[0.0, -0.0611, 0.0181],
    quat_wxyz=[0.0172091, -0.0172091, 0.706897, 0.706897]
)

TRANS_J5_TOOL_MUJOCO = make_tf_mujoco(
    pos=[-0.0079, -0.000218121, -0.0981274],
    quat_wxyz=[0.707107, 0.0, 0.707107, 0.0]
)


def get_intermediate_transforms_mujoco(joint_angles: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Compute FK using MuJoCo convention (pre-rotated frames + Z rotations).
    For MuJoCo XML compatibility.
    """
    q1 = joint_angles.get('shoulder_pan', 0.0)
    q2 = joint_angles.get('shoulder_lift', 0.0)
    q3 = joint_angles.get('elbow_flex', 0.0)
    q4 = joint_angles.get('wrist_flex', 0.0)
    q5 = joint_angles.get('wrist_roll', 0.0)
    
    T_world = np.eye(4)
    
    # All joints rotate about local Z-axis
    R_1 = make_transform(Rz(q1), np.array([0, 0, 0]))
    TF_1 = T_world @ TRANS_BASE_J1_MUJOCO @ R_1
    
    R_2 = make_transform(Rz(q2), np.array([0, 0, 0]))
    TF_2 = TF_1 @ TRANS_J1_J2_MUJOCO @ R_2
    
    R_3 = make_transform(Rz(q3), np.array([0, 0, 0]))
    TF_3 = TF_2 @ TRANS_J2_J3_MUJOCO @ R_3
    
    R_4 = make_transform(Rz(q4), np.array([0, 0, 0]))
    TF_4 = TF_3 @ TRANS_J3_J4_MUJOCO @ R_4
    
    R_5 = make_transform(Rz(q5), np.array([0, 0, 0]))
    TF_5 = TF_4 @ TRANS_J4_J5_MUJOCO @ R_5
    
    TF_Tool = TF_5 @ TRANS_J5_TOOL_MUJOCO
    
    return {
        'joint1': TF_1,
        'joint2': TF_2,
        'joint3': TF_3,
        'joint4': TF_4,
        'joint5': TF_5,
        'tool': TF_Tool
    }


# ==============================================================================
# UNIFIED INTERFACE
# ==============================================================================

def get_intermediate_transforms(
    joint_angles: Dict[str, float],
    mode: str = 'urdf_native'
) -> Dict[str, np.ndarray]:
    """
    Compute forward kinematics transforms for all joints.
    
    Args:
        joint_angles: Dictionary with keys 'shoulder_pan', 'shoulder_lift', 
                     'elbow_flex', 'wrist_flex', 'wrist_roll' (degrees)
        mode: FK computation mode
            - 'urdf_native' (default): Uses actual URDF RPY values → FOR RVIZ
            - 'mujoco': Uses MuJoCo XML quaternions → FOR MUJOCO
            - 'urdf': Simple URDF convention → LEGACY
    
    Returns:
        Dictionary of 4x4 transformation matrices
    """
    if mode == 'urdf_native':
        return get_intermediate_transforms_urdf_native(joint_angles)
    elif mode == 'mujoco':
        return get_intermediate_transforms_mujoco(joint_angles)
    elif mode == 'urdf':
        return get_intermediate_transforms_urdf(joint_angles)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'urdf_native', 'mujoco', or 'urdf'")


def get_forward_kinematics(
    joint_angles: Dict[str, float],
    mode: str = 'urdf_native'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute tool position and orientation from joint angles.
    
    Args:
        joint_angles: Dictionary of joint angles in degrees
        mode: 'urdf_native' (default), 'mujoco', or 'urdf'
        
    Returns:
        Tuple of (position, rotation_matrix)
    """
    transforms = get_intermediate_transforms(joint_angles, mode=mode)
    T_tool = transforms['tool']
    return T_tool[0:3, 3], T_tool[0:3, 0:3]


def get_full_transform(
    joint_angles: Dict[str, float],
    mode: str = 'urdf_native'
) -> np.ndarray:
    """Get full 4x4 transformation matrix to tool frame"""
    transforms = get_intermediate_transforms(joint_angles, mode=mode)
    return transforms['tool']


# ==============================================================================
# TESTING & COMPARISON
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SO-101 Forward Kinematics - Mode Comparison")
    print("=" * 70)
    
    test_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0
    }
    
    print("\nHome configuration (all zeros):")
    print("-" * 70)
    
    # Test URDF Native mode
    pos_native, _ = get_forward_kinematics(test_config, mode='urdf_native')
    print(f"\n[URDF_NATIVE] joint5/gripper: [{pos_native[0]:.4f}, {pos_native[1]:.4f}, {pos_native[2]:.4f}]")
    
    # Test simple URDF mode
    pos_urdf, _ = get_forward_kinematics(test_config, mode='urdf')
    transforms_urdf = get_intermediate_transforms(test_config, mode='urdf')
    pos_urdf_j5 = transforms_urdf['joint5'][0:3, 3]
    print(f"[URDF]        joint5:         [{pos_urdf_j5[0]:.4f}, {pos_urdf_j5[1]:.4f}, {pos_urdf_j5[2]:.4f}]")
    
    # Test MuJoCo mode
    pos_mujoco, _ = get_forward_kinematics(test_config, mode='mujoco')
    transforms_mujoco = get_intermediate_transforms(test_config, mode='mujoco')
    pos_mujoco_j5 = transforms_mujoco['joint5'][0:3, 3]
    print(f"[MUJOCO]      joint5:         [{pos_mujoco_j5[0]:.4f}, {pos_mujoco_j5[1]:.4f}, {pos_mujoco_j5[2]:.4f}]")
    
    # Compare with actual URDF output
    print("\n" + "=" * 70)
    print("COMPARISON WITH ACTUAL URDF")
    print("=" * 70)
    urdf_actual = np.array([0.0206, -0.2775, 0.2669])
    
    err_native = np.linalg.norm(pos_native - urdf_actual) * 1000
    err_urdf = np.linalg.norm(pos_urdf_j5 - urdf_actual) * 1000
    err_mujoco = np.linalg.norm(pos_mujoco_j5 - urdf_actual) * 1000
    
    print(f"\nActual URDF gripper position: [{urdf_actual[0]:.4f}, {urdf_actual[1]:.4f}, {urdf_actual[2]:.4f}]")
    print(f"\nError comparison:")
    print(f"  urdf_native: {err_native:6.2f} mm {'✓ BEST MATCH' if err_native < min(err_urdf, err_mujoco) else ''}")
    print(f"  urdf:        {err_urdf:6.2f} mm {'✓ BEST MATCH' if err_urdf < min(err_native, err_mujoco) else ''}")
    print(f"  mujoco:      {err_mujoco:6.2f} mm {'✓ BEST MATCH' if err_mujoco < min(err_native, err_urdf) else ''}")
    
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    if err_native < 10:
        print("✓ Use mode='urdf_native' for RViz (error < 10mm)")
    elif err_mujoco < 10:
        print("✓ Use mode='mujoco' for RViz (error < 10mm)")
    else:
        print("⚠ Large errors detected. May need URDF debugging.")
    
    print("\nFor MuJoCo labs: Try both 'urdf_native' and 'mujoco' to see which works better")