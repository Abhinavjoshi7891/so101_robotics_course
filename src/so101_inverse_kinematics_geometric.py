#!/usr/bin/env python3
"""
SO-101 Geometric Inverse Kinematics - CORRECTED
================================================

Analytical IK using EXACT dimensions from SO-101 FK diagram.
Accounts for all joint offsets and transformations.

Key dimensions (from diagram):
- Base offset X: 0.0388353 m
- Base height: 0.0624 m
- Joint 1 offset X: -0.0303992 m (backward!)
- Joint 1 to Joint 2: -0.0542 m (downward!)
- Upper arm (J2-J3): 0.11257 m
- Joint 3 offset: 0.028 m
- Forearm (J3-J4): 0.1349 m
- Wrist offset: 0.0611 m
- Tool extension: 0.1034 m

Author: SO-101 Robotics Course
"""

import numpy as np
from typing import Tuple, Dict, Optional


class SO101GeometricIK:
    """Geometric IK solver with exact SO-101 dimensions."""
    
    def __init__(self, mode='mujoco'):
        """
        Initialize geometric IK solver.
        
        Args:
            mode: 'mujoco' or 'urdf_native' for different coordinate systems
        """
        self.mode = mode
        
        # EXACT dimensions from SO-101 FK diagram
        self.d_base_x = 0.0388353      # World frame to base
        self.d_base_z = 0.0624         # Base height to joint 1
        
        self.d1_x = -0.0303992         # Joint 1 offset (BACKWARD!)
        self.d1_z = -0.0542            # Joint 1 to joint 2 (DOWNWARD!)
        
        self.L2 = 0.11257              # Upper arm (joint 2 to joint 3)
        self.d3_x = 0.028              # Joint 3 offset
        
        self.L3 = 0.1349               # Forearm (joint 3 to joint 4)
        
        self.d4 = 0.0611               # Wrist offset (joint 4 to joint 5)
        self.d5 = 0.1034               # Tool extension (joint 5 to tool)
        
        # For workspace validation
        self.workspace = self._compute_workspace_limits()
        
        print(f"[GeometricIK] Initialized with mode={mode}")
        print(f"  Link dimensions from diagram:")
        print(f"    Base: ({self.d_base_x:.4f}, 0, {self.d_base_z:.4f})")
        print(f"    J1 offset: ({self.d1_x:.4f}, 0, {self.d1_z:.4f})")
        print(f"    Upper arm: {self.L2:.4f}m")
        print(f"    Forearm: {self.L3:.4f}m")
        print(f"    Tool: {self.d4 + self.d5:.4f}m")
    
    def _compute_workspace_limits(self):
        """Compute approximate workspace limits."""
        # Maximum reach when fully extended
        max_reach = self.L2 + self.d3_x + self.L3 + self.d4 + self.d5
        
        # Minimum reach (arm folded back)
        min_reach = abs(self.d_base_x + self.d1_x)
        
        # Height limits
        z_max = self.d_base_z + abs(self.d1_z) + self.L2 + self.L3 + self.d4 + self.d5
        z_min = 0.01  # Above ground
        
        return {
            'r_min': min_reach,
            'r_max': max_reach,
            'z_min': z_min,
            'z_max': z_max
        }
    
    def is_in_workspace(self, target_position: np.ndarray) -> Tuple[bool, str]:
        """Check if target position is within robot workspace."""
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
    
    def solve_theta1(self, target_x: float, target_y: float) -> float:
        """
        Solve θ1 (shoulder_pan) from x-y projection.
        
        Args:
            target_x, target_y: Target position in world frame
        
        Returns:
            θ1 in radians
        """
        # Account for base offset
        x_adj = target_x - self.d_base_x
        
        # Solve for angle
        theta1 = np.arctan2(target_y, x_adj)
        
        return theta1
    
    def get_wrist_position(self, target_position: np.ndarray, theta1: float) -> np.ndarray:
        """
        Compute wrist position (joint 4) from tool position.
        
        The tool extends from wrist by (d4 + d5) along the local X axis
        of the wrist frame (after all rotations).
        
        For vertical grasp (tool pointing down), the tool extends
        primarily in the -Z direction in world frame.
        
        Args:
            target_position: [x, y, z] tool position in world frame
            theta1: Shoulder pan angle
        
        Returns:
            wrist_position: [x, y, z] wrist frame position
        """
        # Total tool extension from wrist
        tool_length = self.d4 + self.d5
        
        # For vertical grasp, tool points down (-Z direction)
        # So wrist is ABOVE the tool
        wrist_position = target_position.copy()
        wrist_position[2] += tool_length  # Add tool length in Z
        
        return wrist_position
    
    def solve_theta2_theta3(
        self,
        wrist_x: float,
        wrist_y: float,
        wrist_z: float,
        theta1: float,
        elbow_up: bool = True
    ) -> Tuple[float, float]:
        """
        Solve θ2 (shoulder_lift) and θ3 (elbow_flex) using law of cosines.
        
        This accounts for ALL the offsets in the SO-101 arm.
        
        Args:
            wrist_x, wrist_y, wrist_z: Wrist position in world frame
            theta1: Already computed shoulder pan angle
            elbow_up: True for elbow-up configuration
        
        Returns:
            (θ2, θ3) in radians
        """
        # Transform wrist position to joint 2 frame
        # Account for base offset
        x_from_base = wrist_x - self.d_base_x
        y_from_base = wrist_y
        z_from_base = wrist_z - self.d_base_z
        
        # Rotate by -theta1 to get into joint 1's frame
        c1 = np.cos(theta1)
        s1 = np.sin(theta1)
        x_j1_frame = c1 * x_from_base + s1 * y_from_base
        y_j1_frame = -s1 * x_from_base + c1 * y_from_base
        
        # Now account for joint 1's offsets
        x_from_j1 = x_j1_frame - self.d1_x
        z_from_j1 = z_from_base - self.d1_z
        
        # This is now the position of the wrist relative to joint 2
        # in the plane of the arm (the y component should be ~0)
        
        # Distance from joint 2 to wrist in the arm plane
        r = np.sqrt(x_from_j1**2 + z_from_j1**2)
        
        # Check reachability with law of cosines
        # The triangle is: joint 2 → joint 3 → wrist
        # Sides: L2 (upper arm), L3+d3_x (forearm+offset), r (distance)
        
        # Effective forearm length (includes joint 3 offset)
        L3_eff = np.sqrt(self.L3**2 + self.d3_x**2)
        
        # Law of cosines for elbow angle
        cos_theta3 = (r**2 - self.L2**2 - L3_eff**2) / (2 * self.L2 * L3_eff)
        
        # Check if reachable
        if cos_theta3 < -1 or cos_theta3 > 1:
            return np.nan, np.nan
        
        # Elbow angle (negative for elbow up in SO-101)
        if elbow_up:
            theta3 = -np.arccos(cos_theta3)
        else:
            theta3 = np.arccos(cos_theta3)
        
        # Solve for shoulder lift angle
        # Angle to wrist from joint 2
        alpha = np.arctan2(z_from_j1, x_from_j1)
        
        # Angle within the triangle (law of cosines again)
        cos_beta = (self.L2**2 + r**2 - L3_eff**2) / (2 * self.L2 * r)
        cos_beta = np.clip(cos_beta, -1, 1)
        beta = np.arccos(cos_beta)
        
        # Adjustment for joint 3 offset
        gamma = np.arctan2(self.d3_x, self.L3)
        
        if elbow_up:
            theta2 = alpha - beta + gamma
        else:
            theta2 = alpha + beta + gamma
        
        return theta2, theta3
    
    def solve_theta4(self, theta2: float, theta3: float) -> float:
        """
        Solve θ4 (wrist_flex) to keep tool pointing down.
        
        For vertical grasp, sum of pitch angles should make tool vertical.
        
        Args:
            theta2: shoulder_lift angle
            theta3: elbow_flex angle
        
        Returns:
            θ4 in radians
        """
        # Account for joint 3 offset effect
        gamma = np.arctan2(self.d3_x, self.L3)
        
        # To keep tool pointing down (parallel to world Z)
        theta4 = -(theta2 + theta3 + gamma)
        
        return theta4
    
    def solve_theta5(self, theta1: float, target_yaw: float = 0.0) -> float:
        """
        Solve θ5 (wrist_roll) to match target orientation.
        
        Args:
            theta1: shoulder_pan angle
            target_yaw: desired rotation around world Z-axis
        
        Returns:
            θ5 in radians
        """
        # θ5 compensates for θ1 and applies target yaw
        theta5 = -theta1 + target_yaw
        
        return theta5
    
    def get_inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        target_yaw: float = 0.0,
        elbow_up: bool = True,
        validate_workspace: bool = True,
        verbose: bool = False
    ) -> Tuple[Dict[str, float], bool, str]:
        """
        Compute inverse kinematics using geometric approach with exact dimensions.
        
        Args:
            target_position: [x, y, z] target tool position
            target_orientation: 3x3 rotation matrix (optional)
            target_yaw: rotation around world Z-axis in radians
            elbow_up: True for elbow-up configuration
            validate_workspace: Check if position is reachable
            verbose: Print debug information
        
        Returns:
            (joint_config, success, message) tuple
        """
        if verbose:
            print(f"\n[GeometricIK] Solving IK for target: {target_position}")
        
        # Validate workspace
        if validate_workspace:
            is_valid, reason = self.is_in_workspace(target_position)
            if not is_valid:
                if verbose:
                    print(f"  ✗ Workspace validation failed: {reason}")
                return {
                    'shoulder_pan': 0.0,
                    'shoulder_lift': 0.0,
                    'elbow_flex': 0.0,
                    'wrist_flex': 0.0,
                    'wrist_roll': 0.0,
                    'gripper': 50.0
                }, False, f"Workspace violation: {reason}"
        
        try:
            # Step 1: Solve θ1 (shoulder_pan)
            theta1 = self.solve_theta1(target_position[0], target_position[1])
            if verbose:
                print(f"  θ1 (shoulder_pan): {np.rad2deg(theta1):.2f}°")
            
            # Step 2: Get wrist position
            wrist_pos = self.get_wrist_position(target_position, theta1)
            if verbose:
                print(f"  Wrist position: [{wrist_pos[0]:.4f}, {wrist_pos[1]:.4f}, {wrist_pos[2]:.4f}]")
            
            # Step 3: Solve θ2 and θ3
            theta2, theta3 = self.solve_theta2_theta3(
                wrist_pos[0], wrist_pos[1], wrist_pos[2],
                theta1, elbow_up
            )
            
            if np.isnan(theta2) or np.isnan(theta3):
                if verbose:
                    print(f"  ✗ θ2, θ3 returned NaN (unreachable)")
                return {
                    'shoulder_pan': 0.0,
                    'shoulder_lift': 0.0,
                    'elbow_flex': 0.0,
                    'wrist_flex': 0.0,
                    'wrist_roll': 0.0,
                    'gripper': 50.0
                }, False, "Target unreachable (IK returned NaN)"
            
            if verbose:
                print(f"  θ2 (shoulder_lift): {np.rad2deg(theta2):.2f}°")
                print(f"  θ3 (elbow_flex): {np.rad2deg(theta3):.2f}°")
            
            # Step 4: Solve θ4 (wrist_flex)
            theta4 = self.solve_theta4(theta2, theta3)
            if verbose:
                print(f"  θ4 (wrist_flex): {np.rad2deg(theta4):.2f}°")
            
            # Step 5: Solve θ5 (wrist_roll)
            theta5 = self.solve_theta5(theta1, target_yaw)
            if verbose:
                print(f"  θ5 (wrist_roll): {np.rad2deg(theta5):.2f}°")
            
            # Build joint configuration (convert to degrees)
            joint_config = {
                'shoulder_pan': np.rad2deg(theta1),
                'shoulder_lift': np.rad2deg(theta2),
                'elbow_flex': np.rad2deg(theta3),
                'wrist_flex': np.rad2deg(theta4),
                'wrist_roll': np.rad2deg(theta5),
                'gripper': 50.0
            }
            
            if verbose:
                print(f"  ✓ IK solution found")
            
            return joint_config, True, "IK solution found"
            
        except Exception as e:
            if verbose:
                print(f"  ✗ Exception: {str(e)}")
            return {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 50.0
            }, False, f"IK failed: {str(e)}"


# Convenience function
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
        **kwargs: Additional arguments
    
    Returns:
        (joint_config, success, message) tuple
    """
    solver = SO101GeometricIK(mode=mode)
    return solver.get_inverse_kinematics(target_position, target_orientation, **kwargs)


# Test the corrected geometric IK
if __name__ == "__main__":
    print("=" * 70)
    print("SO-101 Geometric IK Test (CORRECTED)")
    print("=" * 70)
    
    # Import FK for verification
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    
    try:
        from so101_forward_kinematics import get_forward_kinematics
        FK_AVAILABLE = True
    except ImportError:
        print("Warning: FK not available for verification")
        FK_AVAILABLE = False
    
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
            'name': 'Low grasp',
            'position': np.array([0.20, 0.10, 0.014]),
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
            validate_workspace=True,
            verbose=True
        )
        
        print(f"\nResult:")
        print(f"  Success: {success}")
        print(f"  Message: {message}")
        
        if success and FK_AVAILABLE:
            # Verify with FK
            verify_pos, _ = get_forward_kinematics(config, mode=test['mode'])
            error = np.linalg.norm(verify_pos - test['position'])
            print(f"\n  FK Verification:")
            print(f"    Target:   {test['position']}")
            print(f"    Achieved: {verify_pos}")
            print(f"    Error:    {error*1000:.2f} mm")
            
            if error < 0.001:
                print(f"    ✓ EXCELLENT (< 1mm)")
            elif error < 0.010:
                print(f"    ✓ GOOD (< 10mm)")
            elif error < 0.050:
                print(f"    ⚠ ACCEPTABLE (< 50mm)")
            else:
                print(f"    ✗ LARGE ERROR (> 50mm)")
    
    print("\n" + "=" * 70)
    print("Geometric IK Test Complete!")
    print("=" * 70)