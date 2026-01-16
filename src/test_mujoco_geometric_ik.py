#!/usr/bin/env python3
"""
SO-101 Geometric IK for MuJoCo - Using MuJoCo Coordinate System
================================================================

Key insight: MuJoCo and URDF use DIFFERENT coordinate conventions.
This solver works in MuJoCo's coordinate system using FK for verification.
"""

import numpy as np
from typing import Tuple, Dict
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from so101_forward_kinematics import get_forward_kinematics, get_intermediate_transforms


class SO101GeometricIK_MuJoCo_Fixed:
    """Geometric IK that solves in MuJoCo's coordinate system."""
    
    def __init__(self):
        self.d_base_x = 0.0388353
        self.d_base_z = 0.0624
        
        # For straight-arm approximation
        self.L_arm_approx = 0.25  # J2 to J4 when straight
        self.L_tool_eff = 0.16    # Effective tool length when bent
    
    def get_inverse_kinematics(self, target_position, elbow_up=True, 
                              validate_workspace=True, verbose=False):
        """
        Solve geometric IK for MuJoCo using straight-arm approximation.
        
        For low tabletop targets, the optimal solution uses:
        - θ₃ ≈ 0 (straight arm)
        - θ₄ ≈ 90° (wrist bent down)
        
        This dramatically simplifies the geometric problem!
        """
        target_x, target_y, target_z = target_position
        
        # Step 1: θ₁ from XY projection (standard)
        theta1 = np.arctan2(target_y, target_x)
        
        if verbose:
            print(f"Step 1 - θ₁: {np.rad2deg(theta1):.2f}°")
        
        # Step 2: Get J2 position using FK
        config_j1 = {
            'shoulder_pan': np.rad2deg(theta1),
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0
        }
        
        transforms = get_intermediate_transforms(config_j1, mode='mujoco')
        j2_pos = transforms['joint2'][0:3, 3]
        
        if verbose:
            print(f"Step 2 - J2 position: {j2_pos}")
        
        # Step 3: Project target relative to J2 into arm's plane
        target_rel = target_position - j2_pos
        r_planar = np.linalg.norm(target_rel[0:2])
        z_planar = target_rel[2]
        
        if verbose:
            print(f"Step 3 - Planar coords: r={r_planar:.4f}, z={z_planar:.4f}")
        
        # Step 4: For straight arm + bent wrist:
        # The arm extends at angle θ₂, then wrist bends ~90°
        # Geometry (see numerical solution):
        # - Arm length ≈ 0.25m
        # - Tool hangs ≈ 0.16m when wrist bent
        
        # Using iterative approach to find θ₂
        theta2 = np.arctan2(z_planar, r_planar)  # Initial guess
        
        for iteration in range(10):
            # Wrist position if arm extends at theta2
            r_wrist = self.L_arm_approx * np.cos(theta2)
            z_wrist = self.L_arm_approx * np.sin(theta2)
            
            # Tool position when wrist bends (approximately perpendicular)
            # Tool hangs at an angle dependent on θ₂
            tool_angle = theta2 - np.pi/2  # Perpendicular to arm
            r_tool = r_wrist + self.L_tool_eff * np.cos(tool_angle)
            z_tool = z_wrist + self.L_tool_eff * np.sin(tool_angle)
            
            # Error from target
            err_r = r_tool - r_planar
            err_z = z_tool - z_planar
            total_err = np.sqrt(err_r**2 + err_z**2)
            
            if verbose and iteration % 2 == 0:
                print(f"  Iteration {iteration}: θ₂={np.rad2deg(theta2):.2f}°, err={total_err*1000:.2f}mm")
            
            if total_err < 0.001:  # 1mm tolerance
                break
            
            # Gradient descent update
            # Jacobian approximation
            dtheta = 0.01  # Small angle for numerical derivative
            
            r_wrist_plus = self.L_arm_approx * np.cos(theta2 + dtheta)
            z_wrist_plus = self.L_arm_approx * np.sin(theta2 + dtheta)
            tool_angle_plus = (theta2 + dtheta) - np.pi/2
            r_tool_plus = r_wrist_plus + self.L_tool_eff * np.cos(tool_angle_plus)
            z_tool_plus = z_wrist_plus + self.L_tool_eff * np.sin(tool_angle_plus)
            
            dr_dtheta = (r_tool_plus - r_tool) / dtheta
            dz_dtheta = (z_tool_plus - z_tool) / dtheta
            
            # Update using gradient
            gradient = err_r * dr_dtheta + err_z * dz_dtheta
            theta2 -= 0.3 * gradient  # Learning rate
        
        if verbose:
            print(f"Step 4 - θ₂ converged: {np.rad2deg(theta2):.2f}°")
        
        # Step 5: Straight arm
        theta3 = 0.0
        
        # Step 6: Wrist angle to make tool point down
        # θ₄ should make tool perpendicular to arm
        theta4 = 90.0 - np.rad2deg(theta2)
        
        # Step 7: No roll
        theta5 = 0.0
        
        joint_config = {
            'shoulder_pan': np.rad2deg(theta1),
            'shoulder_lift': np.rad2deg(theta2),
            'elbow_flex': theta3,
            'wrist_flex': theta4,
            'wrist_roll': theta5,
            'gripper': 50.0
        }
        
        # Verify with MuJoCo FK
        fk_pos, _ = get_forward_kinematics(joint_config, mode='mujoco')
        error = np.linalg.norm(fk_pos - target_position)
        
        if verbose:
            print(f"\nVerification:")
            print(f"  Target:   {target_position}")
            print(f"  Achieved: {fk_pos}")
            print(f"  Error:    {error*1000:.2f}mm")
        
        success = error < 0.05
        message = f"{'Success' if success else 'Failed'} (error: {error*1000:.1f}mm)"
        
        return joint_config, success, message


if __name__ == "__main__":
    print("="*70)
    print("Testing MuJoCo Geometric IK (Straight-Arm Method)")
    print("="*70)
    
    solver = SO101GeometricIK_MuJoCo_Fixed()
    target = np.array([0.2, 0.0, 0.02])
    
    print(f"\nTarget: {target}")
    
    config, success, msg = solver.get_inverse_kinematics(target, verbose=True)
    
    print(f"\n{msg}")
    print("\nJoint angles:")
    for joint, angle in config.items():
        if joint != 'gripper':
            print(f"  {joint:15s}: {angle:7.2f}°")
    
    # Compare with numerical
    print("\n" + "="*70)
    print("Comparison with Numerical IK:")
    print("="*70)
    print("Numerical:  θ₁=-0.01°, θ₂=21.65°, θ₃=0.00°, θ₄=89.22°")
    print(f"Geometric:  θ₁={config['shoulder_pan']:.2f}°, θ₂={config['shoulder_lift']:.2f}°, θ₃={config['elbow_flex']:.2f}°, θ₄={config['wrist_flex']:.2f}°")