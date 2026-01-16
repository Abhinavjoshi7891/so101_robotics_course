#!/usr/bin/env python3
"""
SO-101 Inverse Kinematics (ECE4560 – Assignment Model)
=====================================================

Analytic geometric IK following the GaTech ECE4560 assignment.

Model assumptions (INTENTIONAL):
- Reduced planar model for θ₂–θ₄
- Vertical grasp
- θ₄ chosen geometrically, not via FK
- θ₅ compensates yaw

This matches all validated test scripts.
"""

import numpy as np
from math import atan2, sqrt, acos, sin, cos, pi, degrees

from so101_forward_kinematics import (
    SO101Params,
    TRANS_J4_J5_MUJOCO,
    TRANS_J5_TOOL_MUJOCO
)

# ============================================================
# Main IK function
# ============================================================

def get_inverse_kinematics(target_position, target_yaw=0.0):
    """
    Compute joint angles for SO-101 using assignment analytic IK.

    Args:
        target_position: [x, y, z] tool position in world frame
        target_yaw: desired yaw about world z-axis (radians)

    Returns:
        joint_config dictionary (degrees)
    """

    x, y, z = target_position

    # --------------------------------------------------------
    # θ₁ — Shoulder pan (verified)
    # --------------------------------------------------------
    theta1 = atan2(-y, x)

    # --------------------------------------------------------
    # Wrist position (assignment step 1c)
    # --------------------------------------------------------
    g_wrist_to_tool = TRANS_J4_J5_MUJOCO @ TRANS_J5_TOOL_MUJOCO
    tool_offset = g_wrist_to_tool[0:3, 3]

    # For vertical grasp, tool points straight down
    #z_tool = np.array([0.0, 0.0, -1.0])
    #wrist_pos = np.array([x, y, z]) - tool_offset[2] * z_tool

    # Build target transform (tool pointing down)
    R_tool = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ])

    g_target = np.eye(4)
    g_target[0:3, 0:3] = R_tool
    g_target[0:3, 3] = np.array([x, y, z])

    # Proper wrist position
    g_wrist = g_target @ np.linalg.inv(
        TRANS_J4_J5_MUJOCO @ TRANS_J5_TOOL_MUJOCO
    )

    wrist_pos = g_wrist[0:3, 3]


    # --------------------------------------------------------
    # Planar reduction for θ₂, θ₃
    # --------------------------------------------------------
    #rho = sqrt(wrist_pos[0]**2 + wrist_pos[1]**2)
    #x_p = rho
    #z_p = wrist_pos[2]
    x_p =  cos(theta1) * wrist_pos[0] - sin(theta1) * wrist_pos[1]
    z_p =  wrist_pos[2]


    # Link lengths (instructional model)
    L2 = SO101Params.J2_TO_J3_Z + SO101Params.J3_OFFSET_Z
    L3 = SO101Params.J3_TO_J4_X

    r = sqrt(x_p**2 + z_p**2)

    # --------------------------------------------------------
    # θ₃ — Elbow flex (law of cosines)
    # --------------------------------------------------------
    cos_t3 = (r**2 - L2**2 - L3**2) / (2 * L2 * L3)
    cos_t3 = np.clip(cos_t3, -1.0, 1.0)
    theta3 = acos(cos_t3)

    # --------------------------------------------------------
    # θ₂ — Shoulder lift
    # --------------------------------------------------------
    phi = atan2(z_p, x_p)
    psi = atan2(L3 * sin(theta3), L2 + L3 * cos(theta3))
    theta2 = phi - psi

    # --------------------------------------------------------
    # θ₄ — Wrist flex (pure planar geometry)
    # --------------------------------------------------------
    theta4 = +pi/2 + (theta2 + theta3)

    # --------------------------------------------------------
    # θ₅ — Wrist yaw compensation
    # --------------------------------------------------------
    theta5 = target_yaw - theta1

    # --------------------------------------------------------
    # Return joint configuration (degrees)
    # --------------------------------------------------------
    return {
        'shoulder_pan': degrees(theta1),
        'shoulder_lift': degrees(theta2),
        'elbow_flex': degrees(theta3),
        'wrist_flex': degrees(theta4),
        'wrist_roll': degrees(theta5),
        'gripper': 50.0
    }
