#!/usr/bin/env python3
"""
MuJoCo Geometric IK - Following Assignment EXACTLY
===================================================

Assignment Step 1c:
"g_wrist = g_target @ inv(g_wrist_to_tool)"

where g_wrist_to_tool (g₄ₜ) is computed using FK helper functions.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from so101_forward_kinematics import (
    get_forward_kinematics, 
    get_intermediate_transforms,
    TRANS_J4_J5_MUJOCO,
    TRANS_J5_TOOL_MUJOCO
)

print("="*80)
print("STEP 2: Compute Wrist Position (Assignment Method)")
print("="*80)

target_position = np.array([0.2, 0.2, 0.014])
print(f"\n📍 Target position: {target_position}")

# ============================================================================
# Step 1: Solve θ₁ (CORRECTED)
# ============================================================================
print("\n" + "="*80)
print("Step 1: Solve θ₁")
print("="*80)

theta1_rad = np.arctan2(-target_position[1], target_position[0])
theta1_deg = np.rad2deg(theta1_rad)
print(f"  θ₁ = atan2(-y, x) = {theta1_deg:.2f}°")

# ============================================================================
# Step 2: Compute g₄ₜ (wrist to tool) using FK transforms
# ============================================================================
print("\n" + "="*80)
print("Step 2: Compute g₄ₜ (wrist-to-tool) from FK")
print("="*80)

# From FK code: g₄ₜ = TRANS_J4_J5 @ TRANS_J5_TOOL
# This is the FIXED transformation from wrist (joint4) to tool
g_wrist_to_tool = TRANS_J4_J5_MUJOCO @ TRANS_J5_TOOL_MUJOCO

print("  g₄ₜ (wrist to tool):")
print(f"    Translation: {g_wrist_to_tool[0:3, 3]}")
print(f"    This is FIXED - doesn't depend on joint angles")

# ============================================================================
# Step 3: Build gₜ (target pose)
# ============================================================================
print("\n" + "="*80)
print("Step 3: Build gₜ (tool target pose)")
print("="*80)

# Assignment says: "grasp from above" means tool points DOWN
# We need to determine what orientation represents "tool pointing down"

# From earlier analysis: tool X-axis points down when oriented correctly
# Let's test different target orientations

print("\n  Testing different target orientations...")

# Get numerical IK solution for comparison
from so101_inverse_kinematics import inverse_kinematics_numerical
num_config, _, _ = inverse_kinematics_numerical(target_position, mode='mujoco', verbose=False)
transforms_num = get_intermediate_transforms(num_config, mode='mujoco')
wrist_pos_numerical = transforms_num['joint4'][0:3, 3]
tool_pose_numerical = transforms_num['tool']
R_target_numerical = tool_pose_numerical[0:3, 0:3]

print(f"\n  Numerical solution's tool orientation:")
print(R_target_numerical)
print(f"  Tool X-axis: {R_target_numerical[:, 0]} (points down if Z < 0)")
print(f"  Tool Y-axis: {R_target_numerical[:, 1]}")
print(f"  Tool Z-axis: {R_target_numerical[:, 2]}")

# Use this as our target orientation
R_target = R_target_numerical

g_target = np.eye(4)
g_target[0:3, 0:3] = R_target
g_target[0:3, 3] = target_position

print(f"\n  Using target orientation from numerical solution")

# ============================================================================
# Step 4: Compute g_wrist using assignment equation
# ============================================================================
print("\n" + "="*80)
print("Step 4: g_wrist = g_target @ inv(g_wrist_to_tool)")
print("="*80)

g_wrist_desired = g_target @ np.linalg.inv(g_wrist_to_tool)
wrist_position = g_wrist_desired[0:3, 3]
wrist_orientation = g_wrist_desired[0:3, 0:3]

print(f"  Computed wrist position: {wrist_position}")
print(f"  Numerical wrist position: {wrist_pos_numerical}")
print(f"  Difference: {wrist_position - wrist_pos_numerical}")

diff_norm = np.linalg.norm(wrist_position - wrist_pos_numerical)
print(f"  Distance: {diff_norm*1000:.2f}mm")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

if diff_norm < 0.001:
    print(f"  ✅ PERFECT! Wrist positions match (< 1mm)")
    print(f"\n  The assignment equation works when we use the")
    print(f"  CORRECT target orientation (from numerical solution)")
elif diff_norm < 0.01:
    print(f"  ✅ EXCELLENT! Very close (< 10mm)")
else:
    print(f"  ❌ Still mismatch: {diff_norm*1000:.2f}mm")
    print(f"\n  Problem: We need to determine target orientation")
    print(f"  WITHOUT knowing the numerical solution first!")

print("\n" + "="*80)
print("KEY INSIGHT")
print("="*80)
print("  The assignment assumes we KNOW the target orientation")
print("  for 'vertical grasp'. But what IS that orientation?")
print("  ")
print("  For a simplified teaching model, it might be identity matrix.")
print("  For the full MuJoCo model, it's more complex.")
print("\n  Next step: Determine target orientation from first principles!")