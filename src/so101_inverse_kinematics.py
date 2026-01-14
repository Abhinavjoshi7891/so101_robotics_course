#!/usr/bin/env python3
"""
SO-101 Inverse Kinematics Module
=================================

This module provides inverse kinematics solutions for the SO-101 robot arm.
Supports multiple modes: 'urdf_native' (RViz) and 'mujoco' (MuJoCo sim).

Two IK methods:
1. Numerical IK - Iterative optimization (scipy)
2. Geometric IK - Closed-form analytical solution

Author: SO-101 Robotics Course
"""

import numpy as np
from scipy.optimize import minimize
from so101_forward_kinematics import get_forward_kinematics, get_intermediate_transforms


def inverse_kinematics_numerical(
    target_position, 
    target_orientation=None,
    initial_guess=None,
    max_iterations=100,
    tolerance=1e-6,
    verbose=False,
    mode='urdf_native'
):
    """
    Numerical inverse kinematics using iterative optimization.
    
    This method minimizes the distance between the desired target position
    and the forward kinematics result by optimizing joint angles.
    
    Args:
        target_position: numpy array [x, y, z] target position in meters
        target_orientation: (optional) target rotation matrix (3x3) - not used yet
        initial_guess: (optional) dict of initial joint angles in degrees
        max_iterations: maximum number of optimization iterations
        tolerance: convergence tolerance for position error
        verbose: print optimization progress
        mode: 'urdf_native' for RViz, 'mujoco' for MuJoCo simulation
    
    Returns:
        joint_config: dict of joint angles in degrees
        success: bool, True if IK converged
        error: float, final position error in meters
    """
    
    # Default initial guess (home position)
    if initial_guess is None:
        initial_guess = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 50.0
        }
    
    # Extract joint names (exclude gripper)
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    
    # Initial joint angles as array
    x0 = np.array([initial_guess[name] for name in joint_names])
    
    # Joint limits (degrees)
    bounds = [
        (-180, 180),  # shoulder_pan
        (-90, 90),    # shoulder_lift
        (-120, 0),    # elbow_flex (limited range)
        (-90, 90),    # wrist_flex
        (-180, 180),  # wrist_roll
    ]
    
    # Iteration counter for verbose output
    iteration = [0]
    
    def objective_function(x):
        """Objective: minimize position error."""
        iteration[0] += 1
        
        # Create joint configuration
        config = {
            'shoulder_pan': x[0],
            'shoulder_lift': x[1],
            'elbow_flex': x[2],
            'wrist_flex': x[3],
            'wrist_roll': x[4],
            'gripper': initial_guess['gripper']
        }
        
        # Compute forward kinematics with specified mode
        fk_position, _ = get_forward_kinematics(config, mode=mode)
        
        # Position error
        error = np.linalg.norm(fk_position - target_position)
        
        if verbose and iteration[0] % 10 == 0:
            print(f"  Iteration {iteration[0]}: error = {error:.6f} m")
        
        return error
    
    # Run optimization
    if verbose:
        print(f"  Starting numerical IK (mode={mode})...")
    
    result = minimize(
        objective_function,
        x0,
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': max_iterations, 'ftol': tolerance}
    )
    
    # Extract solution
    joint_angles = result.x
    joint_config = {
        'shoulder_pan': joint_angles[0],
        'shoulder_lift': joint_angles[1],
        'elbow_flex': joint_angles[2],
        'wrist_flex': joint_angles[3],
        'wrist_roll': joint_angles[4],
        'gripper': initial_guess['gripper']
    }
    
    # Verify solution
    final_position, _ = get_forward_kinematics(joint_config, mode=mode)
    final_error = np.linalg.norm(final_position - target_position)
    
    success = final_error < 0.01  # Success if error < 10mm
    
    if verbose:
        if success:
            print(f"  Converged! Final error: {final_error:.6f} m")
        else:
            print(f"  Failed to converge. Final error: {final_error:.6f} m")
    
    return joint_config, success, final_error


def inverse_kinematics_geometric(
    target_position,
    target_orientation=None,
    elbow_up=True,
    mode='urdf_native'
):
    """
    Geometric (analytical) inverse kinematics.
    
    This method uses closed-form solutions based on robot geometry.
    For a 5-DoF arm, we solve for the first 3 joints (position) and
    last 2 joints (orientation) separately.
    
    Args:
        target_position: numpy array [x, y, z] target position in meters
        target_orientation: (optional) target rotation matrix (3x3)
        elbow_up: bool, choose elbow-up (True) or elbow-down (False) configuration
        mode: 'urdf_native' for RViz, 'mujoco' for MuJoCo simulation
    
    Returns:
        joint_config: dict of joint angles in degrees
        success: bool, True if solution is valid
    """
    
    # Target position
    x, y, z = target_position
    
    # Robot link lengths (approximate, adjust based on URDF/MuJoCo)
    # These should match your robot's actual dimensions
    if mode == 'mujoco':
        # MuJoCo coordinate system
        L1 = 0.0948817  # Base to shoulder_lift
        L2 = 0.11257    # Upper arm length
        L3 = 0.1349     # Forearm length
        L4 = 0.0611     # Wrist to tool
    else:  # urdf_native
        # URDF coordinate system
        L1 = 0.0948817
        L2 = 0.11257
        L3 = 0.1349
        L4 = 0.0611
    
    # Joint 1: shoulder_pan (rotation about Z)
    theta1 = np.arctan2(y, x)
    
    # Project into XZ plane
    r = np.sqrt(x**2 + y**2)
    
    # Adjust z for base offset
    z_adjusted = z - L1
    
    # Distance to wrist center (subtract tool length)
    r_wrist = r
    z_wrist = z_adjusted - L4
    
    # Distance from shoulder to wrist
    d = np.sqrt(r_wrist**2 + z_wrist**2)
    
    # Check reachability
    if d > (L2 + L3) or d < abs(L2 - L3):
        # Target unreachable
        # Return best-effort solution
        joint_config = {
            'shoulder_pan': np.rad2deg(theta1),
            'shoulder_lift': 0.0,
            'elbow_flex': -45.0,
            'wrist_flex': 45.0,
            'wrist_roll': 0.0,
            'gripper': 50.0
        }
        return joint_config, False
    
    # Joint 3: elbow_flex (using law of cosines)
    cos_theta3 = (d**2 - L2**2 - L3**2) / (2 * L2 * L3)
    cos_theta3 = np.clip(cos_theta3, -1, 1)  # Numerical stability
    
    if elbow_up:
        theta3 = -np.arccos(cos_theta3)  # Negative for elbow up
    else:
        theta3 = np.arccos(cos_theta3)   # Positive for elbow down
    
    # Joint 2: shoulder_lift
    alpha = np.arctan2(z_wrist, r_wrist)
    beta = np.arccos((L2**2 + d**2 - L3**2) / (2 * L2 * d))
    
    if elbow_up:
        theta2 = alpha - beta
    else:
        theta2 = alpha + beta
    
    # Joint 4: wrist_flex (keep tool pointing down)
    theta4 = -(theta2 + theta3)
    
    # Joint 5: wrist_roll (arbitrary, set to 0)
    theta5 = 0.0
    
    # Create configuration
    joint_config = {
        'shoulder_pan': np.rad2deg(theta1),
        'shoulder_lift': np.rad2deg(theta2),
        'elbow_flex': np.rad2deg(theta3),
        'wrist_flex': np.rad2deg(theta4),
        'wrist_roll': np.rad2deg(theta5),
        'gripper': 50.0
    }
    
    # Verify solution
    verify_position, _ = get_forward_kinematics(joint_config, mode=mode)
    error = np.linalg.norm(verify_position - target_position)
    
    success = error < 0.05  # Success if error < 50mm
    
    return joint_config, success


def get_inverse_kinematics(
    target_position,
    target_orientation=None,
    method='numerical',
    mode='urdf_native',
    **kwargs
):
    """
    Main IK function that dispatches to specific methods.
    
    Args:
        target_position: numpy array [x, y, z] target position
        target_orientation: (optional) target rotation matrix
        method: 'numerical' or 'geometric'
        mode: 'urdf_native' for RViz, 'mujoco' for MuJoCo
        **kwargs: additional arguments passed to specific method
    
    Returns:
        If method='numerical': (joint_config, success, error)
        If method='geometric': (joint_config, success)
    """
    if method == 'numerical':
        return inverse_kinematics_numerical(
            target_position,
            target_orientation=target_orientation,
            mode=mode,
            **kwargs
        )
    elif method == 'geometric':
        return inverse_kinematics_geometric(
            target_position,
            target_orientation=target_orientation,
            mode=mode,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown IK method: {method}")


# Test the IK functions
if __name__ == "__main__":
    print("=" * 70)
    print("SO-101 Inverse Kinematics Test")
    print("=" * 70)
    
    # Test target
    target = np.array([0.25, 0.10, 0.20])
    
    print(f"\nTarget position: [{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}]")
    
    # Test both modes
    for mode in ['urdf_native', 'mujoco']:
        print(f"\n{'='*70}")
        print(f"Testing mode: {mode}")
        print('='*70)
        
        # Numerical IK
        print("\n1. Numerical IK:")
        config_num, success_num, error_num = inverse_kinematics_numerical(
            target,
            verbose=True,
            mode=mode
        )
        print(f"   Success: {success_num}")
        print(f"   Error: {error_num*1000:.2f} mm")
        print(f"   Joint angles:")
        for joint, angle in config_num.items():
            if joint != 'gripper':
                print(f"     {joint:15s}: {angle:7.2f}°")
        
        # Verify with FK
        verify_pos, _ = get_forward_kinematics(config_num, mode=mode)
        print(f"   Verification: [{verify_pos[0]:.3f}, {verify_pos[1]:.3f}, {verify_pos[2]:.3f}]")
        
        # Geometric IK
        print("\n2. Geometric IK:")
        config_geo, success_geo = inverse_kinematics_geometric(
            target,
            elbow_up=True,
            mode=mode
        )
        print(f"   Success: {success_geo}")
        if success_geo:
            verify_pos_geo, _ = get_forward_kinematics(config_geo, mode=mode)
            error_geo = np.linalg.norm(verify_pos_geo - target)
            print(f"   Error: {error_geo*1000:.2f} mm")
            print(f"   Joint angles:")
            for joint, angle in config_geo.items():
                if joint != 'gripper':
                    print(f"     {joint:15s}: {angle:7.2f}°")
            print(f"   Verification: [{verify_pos_geo[0]:.3f}, {verify_pos_geo[1]:.3f}, {verify_pos_geo[2]:.3f}]")
    
    print("\n" + "="*70)
    print("IK Test Complete!")
    print("="*70)