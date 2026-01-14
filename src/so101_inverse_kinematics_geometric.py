#!/usr/bin/env python3
"""
SO-101 Geometric Inverse Kinematics
====================================

Analytical/geometric approach for SO-101 robot arm.
Fast, reliable, closed-form solution.

Based on the geometric decomposition approach:
1. Solve θ1 (shoulder_pan) from x-y projection
2. Compute wrist position from tool position
3. Solve θ2, θ3 (shoulder_lift, elbow_flex) using law of cosines
4. Solve θ4 (wrist_flex) to keep tool pointing down
5. Solve θ5 (wrist_roll) to match target orientation

Author: SO-101 Robotics Course
"""

import numpy as np
from typing import Tuple, Dict, Optional


class SO101GeometricIK:
    """Geometric IK solver for SO-101 robot arm."""
    
    def __init__(self, mode='mujoco'):
        """
        Initialize geometric IK solver.
        
        Args:
            mode: 'mujoco' or 'urdf_native' for different coordinate systems
        """
        self.mode = mode
        
        # Link lengths (meters) - from URDF/MuJoCo
        if mode == 'mujoco':
            # MuJoCo coordinate system
            self.base_offset_x = 0.0388353  # Base offset in X
            self.base_offset_z = 0.0624     # Base height
            self.L1 = 0.0948817             # Joint 1 to Joint 2 (Z direction)
            self.L2 = 0.11257               # Upper arm length (Joint 2 to Joint 3)
            self.L3 = 0.1349                # Forearm length (Joint 3 to Joint 4)
            self.L4 = 0.0611                # Wrist to tool frame
            self.L5 = 0.1034                # Tool offset (if needed)
        else:  # urdf_native
            # URDF coordinate system
            self.base_offset_x = 0.0388353
            self.base_offset_z = 0.0624
            self.L1 = 0.0948817
            self.L2 = 0.11257
            self.L3 = 0.1349
            self.L4 = 0.0611
            self.L5 = 0.1034
        
        # Workspace limits (for validation)
        self.workspace = self._compute_workspace_limits()
    
    def _compute_workspace_limits(self):
        """Compute approximate workspace limits."""
        # Maximum reach (all joints extended)
        max_reach = self.L2 + self.L3 + self.L4
        min_reach = abs(self.L2 - self.L3) - self.L4
        
        return {
            'r_min': max(0.05, min_reach),  # Minimum radial distance
            'r_max': max_reach,              # Maximum radial distance
            'z_min': 0.01,                   # Minimum height (above ground)
            'z_max': self.L1 + self.L2 + self.L3 + self.L4  # Maximum height
        }
    
    def is_in_workspace(self, target_position: np.ndarray) -> Tuple[bool, str]:
        """
        Check if target position is within robot workspace.
        
        Args:
            target_position: [x, y, z] target position
        
        Returns:
            (is_valid, reason) tuple
        """
        x, y, z = target_position
        
        # Check Z limits
        if z < self.workspace['z_min']:
            return False, f"Target too low: z={z:.3f}m < min={self.workspace['z_min']:.3f}m"
        if z > self.workspace['z_max']:
            return False, f"Target too high: z={z:.3f}m > max={self.workspace['z_max']:.3f}m"
        
        # Check radial distance
        r = np.sqrt(x**2 + y**2)
        if r < self.workspace['r_min']:
            return False, f"Target too close: r={r:.3f}m < min={self.workspace['r_min']:.3f}m"
        if r > self.workspace['r_max']:
            return False, f"Target too far: r={r:.3f}m > max={self.workspace['r_max']:.3f}m"
        
        return True, "Position in workspace"
    
    def get_wrist_position(self, target_position: np.ndarray, target_orientation: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute wrist position from target tool position.
        
        The wrist frame (joint 4) is offset from the tool frame.
        We need to find where the wrist should be to place the tool at target.
        
        Args:
            target_position: [x, y, z] target tool position
            target_orientation: 3x3 rotation matrix (optional, assumes pointing down)
        
        Returns:
            wrist_position: [x, y, z] wrist frame position
        """
        if target_orientation is None:
            # Default: tool pointing straight down
            # Z-axis points down, X-axis points forward
            target_orientation = np.array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ])
        
        # Tool frame offset from wrist (in tool frame coordinates)
        # The tool extends along the local X-axis
        tool_offset_local = np.array([self.L4, 0, 0])
        
        # Transform to world frame
        tool_offset_world = target_orientation @ tool_offset_local
        
        # Wrist position = Tool position - offset
        wrist_position = target_position - tool_offset_world
        
        return wrist_position
    
    def solve_theta1(self, target_position: np.ndarray) -> float:
        """
        Solve θ1 (shoulder_pan) from x-y projection.
        
        Projects target onto ground plane and computes angle.
        
        Args:
            target_position: [x, y, z] target position
        
        Returns:
            θ1 in radians
        """
        x, y, z = target_position
        
        # Account for base offset
        x_adjusted = x - self.base_offset_x
        
        # Compute angle from x-y projection
        theta1 = np.arctan2(y, x_adjusted)
        
        return theta1
    
    def solve_theta2_theta3(self, wrist_position: np.ndarray, theta1: float, elbow_up: bool = True) -> Tuple[float, float]:
        """
        Solve θ2 (shoulder_lift) and θ3 (elbow_flex) using law of cosines.
        
        Forms a triangle between joint 1, 2, and wrist frame.
        Uses law of cosines to solve for joint angles.
        
        Args:
            wrist_position: [x, y, z] wrist frame position
            theta1: Previously computed shoulder_pan angle
            elbow_up: True for elbow-up configuration, False for elbow-down
        
        Returns:
            (θ2, θ3) in radians
        """
        x, y, z = wrist_position
        
        # Transform to plane perpendicular to θ1
        # Distance in x-y plane
        r_xy = np.sqrt(x**2 + y**2)
        r_adjusted = r_xy - self.base_offset_x
        
        # Height from joint 2
        z_from_j2 = z - self.base_offset_z - self.L1
        
        # Distance from joint 2 to wrist
        d = np.sqrt(r_adjusted**2 + z_from_j2**2)
        
        # Check reachability
        if d > (self.L2 + self.L3) or d < abs(self.L2 - self.L3):
            # Return NaN if unreachable
            return np.nan, np.nan
        
        # Law of cosines for θ3 (elbow angle)
        cos_theta3 = (d**2 - self.L2**2 - self.L3**2) / (2 * self.L2 * self.L3)
        cos_theta3 = np.clip(cos_theta3, -1, 1)  # Numerical stability
        
        if elbow_up:
            theta3 = -np.arccos(cos_theta3)  # Negative for elbow up
        else:
            theta3 = np.arccos(cos_theta3)   # Positive for elbow down
        
        # Solve for θ2 (shoulder lift)
        alpha = np.arctan2(z_from_j2, r_adjusted)
        beta = np.arccos((self.L2**2 + d**2 - self.L3**2) / (2 * self.L2 * d))
        
        if elbow_up:
            theta2 = alpha - beta
        else:
            theta2 = alpha + beta
        
        return theta2, theta3
    
    def solve_theta4(self, theta2: float, theta3: float) -> float:
        """
        Solve θ4 (wrist_flex) to keep tool pointing down.
        
        For a vertical grasp (tool pointing down), the sum of joint angles
        should align the tool with the vertical axis.
        
        Args:
            theta2: shoulder_lift angle
            theta3: elbow_flex angle
        
        Returns:
            θ4 in radians
        """
        # To keep tool pointing down (parallel to Z-axis)
        # The sum of pitch angles should be zero
        theta4 = -(theta2 + theta3)
        
        return theta4
    
    def solve_theta5(self, theta1: float, target_yaw: float = 0.0) -> float:
        """
        Solve θ5 (wrist_roll) to match target orientation around Z-axis.
        
        Since θ1 and θ5 both rotate around Z-axis, we need:
        θ5 = target_yaw - θ1
        
        But θ5 has opposite rotation direction, so:
        θ5 = -(target_yaw - θ1) = θ1 - target_yaw
        
        Args:
            theta1: shoulder_pan angle
            target_yaw: desired rotation around world Z-axis (default 0)
        
        Returns:
            θ5 in radians
        """
        # Compensate for θ1 rotation, apply target yaw
        theta5 = -theta1 + target_yaw
        
        return theta5
    
    def get_inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        target_yaw: float = 0.0,
        elbow_up: bool = True,
        validate_workspace: bool = True
    ) -> Tuple[Dict[str, float], bool, str]:
        """
        Compute inverse kinematics using geometric approach.
        
        Args:
            target_position: [x, y, z] target tool position
            target_orientation: 3x3 rotation matrix (optional)
            target_yaw: rotation around world Z-axis in radians
            elbow_up: True for elbow-up configuration
            validate_workspace: Check if position is reachable
        
        Returns:
            (joint_config, success, message) tuple
        """
        # Validate workspace
        if validate_workspace:
            is_valid, reason = self.is_in_workspace(target_position)
            if not is_valid:
                return {
                    'shoulder_pan': 0.0,
                    'shoulder_lift': 0.0,
                    'elbow_flex': 0.0,
                    'wrist_flex': 0.0,
                    'wrist_roll': 0.0,
                    'gripper': 50.0
                }, False, f"Workspace violation: {reason}"
        
        try:
            # Step 1b: Solve θ1 (shoulder_pan)
            theta1 = self.solve_theta1(target_position)
            
            # Step 1c: Compute wrist position
            wrist_position = self.get_wrist_position(target_position, target_orientation)
            
            # Step 1d: Solve θ2 and θ3 (shoulder_lift, elbow_flex)
            theta2, theta3 = self.solve_theta2_theta3(wrist_position, theta1, elbow_up)
            
            # Check for NaN (unreachable)
            if np.isnan(theta2) or np.isnan(theta3):
                return {
                    'shoulder_pan': 0.0,
                    'shoulder_lift': 0.0,
                    'elbow_flex': 0.0,
                    'wrist_flex': 0.0,
                    'wrist_roll': 0.0,
                    'gripper': 50.0
                }, False, "Target unreachable (IK returned NaN)"
            
            # Step 1e: Solve θ4 (wrist_flex)
            theta4 = self.solve_theta4(theta2, theta3)
            
            # Step 1f: Solve θ5 (wrist_roll)
            theta5 = self.solve_theta5(theta1, target_yaw)
            
            # Build joint configuration (convert to degrees)
            joint_config = {
                'shoulder_pan': np.rad2deg(theta1),
                'shoulder_lift': np.rad2deg(theta2),
                'elbow_flex': np.rad2deg(theta3),
                'wrist_flex': np.rad2deg(theta4),
                'wrist_roll': np.rad2deg(theta5),
                'gripper': 50.0
            }
            
            return joint_config, True, "IK solution found"
            
        except Exception as e:
            return {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 50.0
            }, False, f"IK failed: {str(e)}"


# Convenience function matching the assignment signature
def get_inverse_kinematics(
    target_position: np.ndarray,
    target_orientation: Optional[np.ndarray] = None,
    mode: str = 'mujoco',
    **kwargs
) -> Tuple[Dict[str, float], bool, str]:
    """
    Convenience function for geometric IK.
    
    Args:
        target_position: [x, y, z] target position
        target_orientation: 3x3 rotation matrix (optional)
        mode: 'mujoco' or 'urdf_native'
        **kwargs: Additional arguments (elbow_up, validate_workspace, etc.)
    
    Returns:
        (joint_config, success, message) tuple
    """
    solver = SO101GeometricIK(mode=mode)
    return solver.get_inverse_kinematics(target_position, target_orientation, **kwargs)


# Test the geometric IK
if __name__ == "__main__":
    print("=" * 70)
    print("SO-101 Geometric IK Test")
    print("=" * 70)
    
    # Test positions
    test_cases = [
        {
            'name': 'Forward reach',
            'position': np.array([0.25, 0.0, 0.20]),
            'mode': 'mujoco'
        },
        {
            'name': 'Right side',
            'position': np.array([0.20, 0.15, 0.15]),
            'mode': 'mujoco'
        },
        {
            'name': 'Left side',
            'position': np.array([0.20, -0.15, 0.15]),
            'mode': 'mujoco'
        },
        {
            'name': 'Too far (should fail)',
            'position': np.array([0.50, 0.0, 0.20]),
            'mode': 'mujoco'
        },
    ]
    
    for test in test_cases:
        print(f"\n{'-'*70}")
        print(f"Test: {test['name']}")
        print(f"Target: {test['position']}")
        
        config, success, message = get_inverse_kinematics(
            test['position'],
            mode=test['mode'],
            validate_workspace=True
        )
        
        print(f"Success: {success}")
        print(f"Message: {message}")
        
        if success:
            print(f"Joint angles:")
            for joint, angle in config.items():
                if joint != 'gripper':
                    print(f"  {joint:15s}: {angle:7.2f}°")
            
            # Verify with FK
            from so101_forward_kinematics import get_forward_kinematics
            verify_pos, _ = get_forward_kinematics(config, mode=test['mode'])
            error = np.linalg.norm(verify_pos - test['position'])
            print(f"\nFK Verification:")
            print(f"  Target:   {test['position']}")
            print(f"  Achieved: {verify_pos}")
            print(f"  Error:    {error*1000:.2f} mm")
    
    print("\n" + "=" * 70)
    print("Geometric IK Test Complete!")
    print("=" * 70)