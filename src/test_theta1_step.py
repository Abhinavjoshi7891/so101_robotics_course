#!/usr/bin/env python3
"""
Debug θ₁ Convention for MuJoCo
================================
Test different θ₁ formulas to find the correct one
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from so101_forward_kinematics import get_forward_kinematics, get_intermediate_transforms

target_position = np.array([0.2, 0.2, 0.014])
target_x, target_y = target_position[0], target_position[1]

print("="*80)
print("Testing Different θ₁ Formulas for MuJoCo")
print("="*80)
print(f"\nTarget: [{target_x:.3f}, {target_y:.3f}, 0.014]")
print(f"Target direction (XY): [0.707, 0.707] (45° from +X axis)")

# Test different formulas
theta1_formulas = [
    ("Standard: atan2(y, x)", np.arctan2(target_y, target_x)),
    ("Negated Y: atan2(-y, x)", np.arctan2(-target_y, target_x)),
    ("Swapped: atan2(x, y)", np.arctan2(target_x, target_y)),
    ("Both neg: atan2(-y, -x)", np.arctan2(-target_y, -target_x)),
    ("Negated X: atan2(y, -x)", np.arctan2(target_y, -target_x)),
    ("Add 90°: atan2(y, x) + 90°", np.arctan2(target_y, target_x) + np.pi/2),
]

results = []

for name, theta1_rad in theta1_formulas:
    theta1_deg = np.rad2deg(theta1_rad)
    
    config = {
        'shoulder_pan': theta1_deg,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    
    # Get joint positions
    transforms = get_intermediate_transforms(config, mode='mujoco')
    j2_pos = transforms['joint2'][0:3, 3]
    j3_pos = transforms['joint3'][0:3, 3]
    
    # Get tool position
    tool_pos, _ = get_forward_kinematics(config, mode='mujoco')
    
    # Arm direction (J2 to J3)
    arm_dir = (j3_pos - j2_pos)[0:2]
    arm_dir_norm = arm_dir / np.linalg.norm(arm_dir)
    
    # Target direction
    target_dir = np.array([target_x, target_y])
    target_dir_norm = target_dir / np.linalg.norm(target_dir)
    
    # Alignment
    alignment = np.dot(arm_dir_norm, target_dir_norm)
    
    results.append((name, theta1_deg, alignment, arm_dir_norm, tool_pos))

print("\n" + "="*80)
print("Results:")
print("="*80)

for name, theta1_deg, alignment, arm_dir, tool_pos in results:
    status = "✅" if alignment > 0.95 else ("✓" if alignment > 0.8 else "❌")
    print(f"\n{status} {name}")
    print(f"   θ₁ = {theta1_deg:7.2f}°")
    print(f"   Arm direction: [{arm_dir[0]:6.3f}, {arm_dir[1]:6.3f}]")
    print(f"   Alignment: {alignment:6.4f}")
    print(f"   Tool XY: [{tool_pos[0]:6.3f}, {tool_pos[1]:6.3f}]")

print("\n" + "="*80)
print("Finding the BEST formula:")
print("="*80)

best_idx = max(range(len(results)), key=lambda i: results[i][2])
best_name, best_theta1, best_alignment, _, _ = results[best_idx]

print(f"\n🎯 WINNER: {best_name}")
print(f"   θ₁ = {best_theta1:.2f}°")
print(f"   Alignment = {best_alignment:.4f}")

if best_alignment > 0.95:
    print(f"\n✅ SUCCESS! This is the correct θ₁ formula for MuJoCo!")
else:
    print(f"\n⚠️  Best we found, but still not perfect. May need more investigation.")