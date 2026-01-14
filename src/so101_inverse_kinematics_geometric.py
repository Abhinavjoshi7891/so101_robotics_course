#!/usr/bin/env python3
"""
SO-101 Geometric Inverse Kinematics - FIXED
================================================

Analytical IK using EXACT dimensions from SO-101 FK diagram.
Fixed for Base Rotation alignment and Radial Reaching.
"""

import numpy as np
from typing import Tuple, Dict, Optional

class SO101GeometricIK:
    """Geometric IK solver with exact SO-101 dimensions."""
    
    def __init__(self, mode='mujoco'):
        self.mode = mode
        
        # EXACT dimensions from SO-101 FK diagram
        self.d_base_x = 0.0388353      
        self.d_base_z = 0.0624         
        
        self.d1_x = -0.0303992         
        self.d1_z = -0.0542            
        
        self.L2 = 0.11257              
        self.d3_x = 0.028              
        
        self.L3 = 0.1349               
        
        self.d4 = 0.0611               
        self.d5 = 0.1034               
        
        # Workspace limits
        self.workspace = self._compute_workspace_limits()
    
    def _compute_workspace_limits(self):
        max_reach = self.L2 + self.d3_x + self.L3 + self.d4 + self.d5
        return {
            'r_min': 0.05,
            'r_max': max_reach * 1.05, # Allow slight tolerance
            'z_min': -0.10,
            'z_max': max_reach + self.d_base_z
        }
    
    def is_in_workspace(self, target_position: np.ndarray) -> Tuple[bool, str]:
        x, y, z = target_position
        r = np.sqrt(x**2 + y**2 + z**2) # Simple spherical check
        if r > self.workspace['r_max']:
            return False, f"Target too far: r={r:.3f}m"
        return True, "Position in workspace"
    
    def solve_theta1(self, target_x: float, target_y: float) -> float:
        """
        Solve θ1 (shoulder_pan).
        Corrected for SO-101 where 0 degrees aligns with -Y axis.
        """
        # Calculate angle assuming standard X-forward
        # But SO-101 Home (0 deg) is along -Y.
        # So we want x~0, y<0 to result in theta1=0.
        
        # Using atan2(x, -y) gives 0 when x=0, y=-1
        theta1 = np.arctan2(target_x - self.d_base_x, -target_y)
        
        return theta1
    
    def get_wrist_position(self, target_position: np.ndarray, theta1: float) -> np.ndarray:
        """
        Compute wrist position.
        
        STRATEGY CHANGE: 
        Instead of forcing 'Vertical Down', we assume 'Radial Reach' 
        (Tool points from shoulder towards target) because no orientation 
        was provided in Lab 2.2.
        """
        tool_length = self.d4 + self.d5
        
        # Vector from shoulder (approx) to target
        # Shoulder is roughly at (d_base_x, 0, d_base_z)
        shoulder_pos = np.array([self.d_base_x, 0, self.d_base_z])
        
        reach_vector = target_position - shoulder_pos
        reach_dist = np.linalg.norm(reach_vector)
        
        if reach_dist < 0.001:
            direction = np.array([0, 0, -1])
        else:
            direction = reach_vector / reach_dist
            
        # Retract back from target along the reaching direction
        wrist_position = target_position - (direction * tool_length)
        
        return wrist_position
    
    def solve_theta2_theta3(self, wrist_x, wrist_y, wrist_z, theta1, elbow_up=True):
        """Solve θ2 and θ3 using Law of Cosines."""
        
        # 1. Transform Wrist to Joint 2 Frame
        # Translate base
        dx = wrist_x - self.d_base_x
        dy = wrist_y
        dz = wrist_z - self.d_base_z
        
        # Rotate by Theta1 (Using the inverse of our new Theta1 logic)
        # If Theta1 represents rotation from -Y, we need to handle that carefully.
        # Standard rotation matrix Rz(theta):
        # x' = x cos - y sin
        # y' = x sin + y cos
        # But our theta1 definition was specific. Let's use magnitude for r.
        
        # Horizontal distance from J1 axis
        r_horiz = np.sqrt(dx**2 + dy**2)
        
        # Correct for J1 X-offset (d1_x)
        # If d1_x pushes the arm "forward/backward", it changes the reach radius
        # Triangle in the plane:
        # horizontal_reach = sqrt(r_horiz**2 - d1_x**2)  (if d1_x is a side offset)
        # But here d1_x seems to be an inline offset.
        
        # Simplified planar projection:
        # Coordinate in the arm's plane:
        x_arm = r_horiz - abs(self.d1_x) # Subtract the backward offset
        z_arm = dz - self.d1_z           # Subtract downward offset (add height)
        
        # Distance from J2 to Wrist
        r_j2_wrist = np.sqrt(x_arm**2 + z_arm**2)
        
        # Effective L3 (including offset)
        L3_eff = np.sqrt(self.L3**2 + self.d3_x**2)
        
        # Law of Cosines for Theta3 (Elbow)
        # c^2 = a^2 + b^2 - 2ab cos(C)
        # r^2 = L2^2 + L3_eff^2 - 2*L2*L3*cos(180-theta3)
        numerator = r_j2_wrist**2 - self.L2**2 - L3_eff**2
        denominator = 2 * self.L2 * L3_eff
        
        cos_theta3 = numerator / denominator
        
        if abs(cos_theta3) > 1.0:
            return np.nan, np.nan
            
        # Elbow Angle
        if elbow_up:
            theta3 = -np.arccos(cos_theta3)
        else:
            theta3 = np.arccos(cos_theta3)
            
        # Shoulder Angle (Theta2)
        # Angle to wrist vector
        alpha = np.arctan2(z_arm, x_arm)
        
        # Angle of triangle at shoulder
        cos_beta = (self.L2**2 + r_j2_wrist**2 - L3_eff**2) / (2 * self.L2 * r_j2_wrist)
        cos_beta = np.clip(cos_beta, -1, 1)
        beta = np.arccos(cos_beta)
        
        # Offset angle for forearm (gamma)
        gamma = np.arctan2(self.d3_x, self.L3)
        
        if elbow_up:
            theta2 = alpha - beta + gamma
        else:
            theta2 = alpha + beta - gamma # Flip logic for elbow down
            
        return theta2, theta3

    def solve_theta4(self, theta2, theta3):
        """
        Solve θ4.
        For Radial Reach (pointing at target), wrist_flex should be 0 
        relative to the line of the arm, minus the L3 offset gamma.
        """
        gamma = np.arctan2(self.d3_x, self.L3)
        return -gamma

    def solve_theta5(self, theta1, target_yaw=0.0):
        return 0.0

    def get_inverse_kinematics(
        self, target_position, target_orientation=None, 
        target_yaw=0.0, elbow_up=True, validate_workspace=True, verbose=False
    ):
        
        if validate_workspace:
            valid, msg = self.is_in_workspace(target_position)
            if not valid:
                return self._fail(msg)

        try:
            # 1. Theta 1
            theta1 = self.solve_theta1(target_position[0], target_position[1])
            
            # 2. Wrist Position (Radial Strategy)
            wrist_pos = self.get_wrist_position(target_position, theta1)
            
            # 3. Theta 2 & 3
            theta2, theta3 = self.solve_theta2_theta3(
                wrist_pos[0], wrist_pos[1], wrist_pos[2], theta1, elbow_up
            )
            
            if np.isnan(theta2):
                return self._fail("Unreachable (Triangle inequality)")

            # 4. Theta 4 & 5
            theta4 = self.solve_theta4(theta2, theta3)
            theta5 = self.solve_theta5(theta1)
            
            return {
                'shoulder_pan': np.rad2deg(theta1),
                'shoulder_lift': np.rad2deg(theta2),
                'elbow_flex': np.rad2deg(theta3),
                'wrist_flex': np.rad2deg(theta4),
                'wrist_roll': np.rad2deg(theta5),
                'gripper': 50.0
            }, True, "Success"
            
        except Exception as e:
            return self._fail(str(e))

    def _fail(self, msg):
        return {
            'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 'elbow_flex': 0.0,
            'wrist_flex': 0.0, 'wrist_roll': 0.0, 'gripper': 50.0
        }, False, msg

# Convenience wrapper
def get_inverse_kinematics(target_position, **kwargs):
    solver = SO101GeometricIK(mode=kwargs.get('mode', 'urdf_native'))
    return solver.get_inverse_kinematics(target_position, **kwargs)