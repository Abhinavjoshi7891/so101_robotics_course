#!/usr/bin/env python3
"""
SO-101 Forward Kinematics (XML-Exact Version)
=============================================

This module implements Forward Kinematics by exactly replicating the
MuJoCo/URDF kinematic chain defined in 'so101_new_calib.xml'.

Kinematic Chain Logic:
    T_world_to_link = T_parent @ T_offset @ T_joint
    
    Where:
      - T_offset is the fixed transform (pos + quat) from the XML body definition.
      - T_joint is the rotation about Z (0,0,1) for the hinge joint.

Author: SO-101 Robotics Course
"""

import numpy as np
from typing import Tuple, Dict

# ==============================================================================
# QUATERNION UTILITIES
# ==============================================================================

def quat_to_rot(w, x, y, z):
    """
    Convert (w, x, y, z) quaternion to 3x3 rotation matrix.
    Using the standard conversion formula.
    """
    Nq = w*w + x*x + y*y + z*z
    if Nq < np.finfo(float).eps:
        return np.eye(3)
    s = 2.0 / Nq
    X = x*s; Y = y*s; Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    
    return np.array([
        [1.0-(yY+zZ), xY-wZ, xZ+wY],
        [xY+wZ, 1.0-(xX+zZ), yZ-wX],
        [xZ-wY, yZ+wX, 1.0-(xX+yY)]
    ])

def Rz(theta_deg):
    """Active Rotation about Z axis."""
    rad = np.deg2rad(theta_deg)
    c = np.cos(rad)
    s = np.sin(rad)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def make_tf(pos, quat_wxyz):
    """Build homogeneous transform from XML pos and quat."""
    T = np.eye(4)
    T[0:3, 3] = pos
    T[0:3, 0:3] = quat_to_rot(*quat_wxyz)
    return T

# ==============================================================================
# ROBOT CONSTANTS (Exact copy from so101_new_calib.xml)
# ==============================================================================

# 1. Base -> Shoulder (body "shoulder")
# <body name="shoulder" pos="0.0388353 ~0 0.0624" quat="3.56e-16 1.22e-15 -1 -4.14e-16">
# Note: The XML quat "0 0 -1 0" is roughly 180 deg Y rotation.
TRANS_BASE_J1 = make_tf(
    pos=[0.0388353, 0.0, 0.0624],
    quat_wxyz=[0.0, 0.0, -1.0, 0.0] 
)

# 2. Shoulder -> Upper Arm (body "upper_arm")
# <body name="upper_arm" pos="-0.0303992 -0.0182778 -0.0542" quat="0.5 -0.5 -0.5 -0.5">
TRANS_J1_J2 = make_tf(
    pos=[-0.0303992, -0.0182778, -0.0542],
    quat_wxyz=[0.5, -0.5, -0.5, -0.5]
)

# 3. Upper Arm -> Lower Arm (body "lower_arm")
# <body name="lower_arm" pos="-0.11257 -0.028 0" quat="0.707107 0 0 0.707107">
# Note: XML says -5.9e-17 which is effectively 0
TRANS_J2_J3 = make_tf(
    pos=[-0.11257, -0.028, 0.0],
    quat_wxyz=[0.707107, 0.0, 0.0, 0.707107]
)

# 4. Lower Arm -> Wrist (body "wrist")
# <body name="wrist" pos="-0.1349 0.0052 0" quat="0.707107 0 0 -0.707107">
TRANS_J3_J4 = make_tf(
    pos=[-0.1349, 0.0052, 0.0],
    quat_wxyz=[0.707107, 0.0, 0.0, -0.707107]
)

# 5. Wrist -> Gripper Base (body "gripper")
# <body name="gripper" pos="0 -0.0611 0.0181" quat="0.0172 -0.0172 0.7069 0.7069">
# Note: Using exact values from XML trace
TRANS_J4_J5 = make_tf(
    pos=[0.0, -0.0611, 0.0181],
    quat_wxyz=[0.0172091, -0.0172091, 0.706897, 0.706897]
)

# 6. Gripper Base -> Tool Tip (site "gripperframe")
# <site name="gripperframe" pos="-0.0079 -0.0002 -0.098" quat="0.7071 0 0.7071 0"/>
TRANS_J5_TOOL = make_tf(
    pos=[-0.0079, -0.000218121, -0.0981274],
    quat_wxyz=[0.707107, 0.0, 0.707107, 0.0]
)


# ==============================================================================
# FORWARD KINEMATICS
# ==============================================================================

def get_intermediate_transforms(joint_angles: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Computes transforms for all joints.
    All joints in XML are type="hinge" axis="0 0 1", so they rotate Z.
    """
    # 1. World Frame
    T_world = np.eye(4)
    
    # 2. Joint 1 (Shoulder Pan)
    # T_w_j1 = T_base_offset @ RotZ(theta1)
    q1 = joint_angles.get('shoulder_pan', 0.0)
    T_j1 = TRANS_BASE_J1 @ make_tf([0,0,0], [1,0,0,0]) # Offset
    # Apply joint rotation in the NEW local frame (post-offset)
    # But wait, T_offset includes rotation.
    # The chain is: T_parent * T_offset * T_joint_rotation
    T_1 = T_world @ TRANS_BASE_J1 @ make_tf([0,0,0], [1,0,0,0]) # Base pos
    T_1_rotated = T_1 @ make_tf([0,0,0], [1,0,0,0]) # Apply rotation?
    
    # Simplified Chain:
    # Frame N = Frame (N-1) @ Offset_N @ RotZ(theta_N)
    
    # Joint 1
    T_1 = TRANS_BASE_J1 @ make_tf([0,0,0], [1,0,0,0]) # Base->J1
    R_1 = make_tf([0,0,0], [1,0,0,0])
    R_1[0:3, 0:3] = Rz(q1)
    TF_1 = T_1 @ R_1
    
    # Joint 2
    q2 = joint_angles.get('shoulder_lift', 0.0)
    R_2 = make_tf([0,0,0], [1,0,0,0]); R_2[0:3, 0:3] = Rz(q2)
    TF_2 = TF_1 @ TRANS_J1_J2 @ R_2
    
    # Joint 3
    q3 = joint_angles.get('elbow_flex', 0.0)
    R_3 = make_tf([0,0,0], [1,0,0,0]); R_3[0:3, 0:3] = Rz(q3)
    TF_3 = TF_2 @ TRANS_J2_J3 @ R_3
    
    # Joint 4
    q4 = joint_angles.get('wrist_flex', 0.0)
    R_4 = make_tf([0,0,0], [1,0,0,0]); R_4[0:3, 0:3] = Rz(q4)
    TF_4 = TF_3 @ TRANS_J3_J4 @ R_4
    
    # Joint 5
    q5 = joint_angles.get('wrist_roll', 0.0)
    R_5 = make_tf([0,0,0], [1,0,0,0]); R_5[0:3, 0:3] = Rz(q5)
    TF_5 = TF_4 @ TRANS_J4_J5 @ R_5
    
    # Tool
    TF_Tool = TF_5 @ TRANS_J5_TOOL
    
    return {
        'joint1': TF_1,
        'joint2': TF_2,
        'joint3': TF_3,
        'joint4': TF_4,
        'joint5': TF_5, # This is the gripper base
        'tool': TF_Tool
    }

def get_forward_kinematics(joint_angles: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    transforms = get_intermediate_transforms(joint_angles)
    T_tool = transforms['tool']
    return T_tool[0:3, 3], T_tool[0:3, 0:3]

if __name__ == "__main__":
    print("Testing XML-Exact FK...")
    test_config = {
        'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 
        'elbow_flex': 0.0, 'wrist_flex': 0.0, 'wrist_roll': 0.0
    }
    pos, rot = get_forward_kinematics(test_config)
    print(f"Home Pos: {pos}")