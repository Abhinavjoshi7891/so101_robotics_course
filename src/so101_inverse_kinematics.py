#!/usr/bin/env python3
"""
SO-101 Inverse Kinematics Implementation
=========================================

This module implements inverse kinematics for the SO-101 robot arm from scratch,
using both geometric and numerical (Jacobian-based) approaches.

The inverse kinematics problem:
    Given: Desired tool position (and optionally orientation)
    Find: Joint angles that achieve this position/orientation

Methods implemented:
    1. Geometric IK: Analytical solution specific to SO-101 geometry
    2. Numerical IK: Jacobian pseudoinverse method (general purpose)
    3. Damped Least Squares: More robust numerical method

Author: SO-101 Robotics Course
License: MIT
"""

import numpy as np
from typing import Tuple, Dict, Optional
from so101_forward_kinematics import (
    get_forward_kinematics, 
    get_full_transform,
    get_intermediate_transforms,
    SO101Params,
    Rx, Ry, Rz,
    make_transform
)


# ==============================================================================
# JACOBIAN COMPUTATION
# ==============================================================================

def compute_jacobian(joint_angles: Dict[str, float], delta: float = 1e-4) -> np.ndarray:
    """
    Compute the Jacobian matrix numerically using finite differences.
    
    The Jacobian relates joint velocities to end-effector velocities:
        ẋ = J(q) * q̇
    
    where:
        - ẋ is the 6x1 end-effector velocity (3 linear + 3 angular)
        - q̇ is the 5x1 joint velocity vector
        - J(q) is the 6x5 Jacobian matrix
    
    We compute this numerically:
        J[i,j] = (f(q + δe_j) - f(q - δe_j)) / (2δ)
    
    Args:
        joint_angles: Current joint configuration (degrees)
        delta: Perturbation size for finite differences
        
    Returns:
        6x5 Jacobian matrix (3 position rows + 3 orientation rows)
    """
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    n_joints = len(joint_names)
    
    # Initialize Jacobian (6 rows: 3 position + 3 orientation)
    J = np.zeros((6, n_joints))
    
    for j, joint_name in enumerate(joint_names):
        # Create perturbed configurations
        angles_plus = joint_angles.copy()
        angles_minus = joint_angles.copy()
        
        angles_plus[joint_name] += delta
        angles_minus[joint_name] -= delta
        
        # Compute forward kinematics for perturbed configurations
        T_plus = get_full_transform(angles_plus)
        T_minus = get_full_transform(angles_minus)
        
        # Position Jacobian (finite difference)
        pos_plus = T_plus[0:3, 3]
        pos_minus = T_minus[0:3, 3]
        J[0:3, j] = (pos_plus - pos_minus) / (2 * np.deg2rad(delta))
        
        # Orientation Jacobian (using rotation matrix difference)
        # This is a simplified version; for full SO(3) treatment, use axis-angle
        R_plus = T_plus[0:3, 0:3]
        R_minus = T_minus[0:3, 0:3]
        
        # Approximate angular velocity from rotation matrices
        # ω ≈ vee(R_plus @ R_minus.T - I) / dt
        dR = R_plus @ R_minus.T
        # Extract approximate rotation vector (small angle approximation)
        omega = np.array([
            (dR[2, 1] - dR[1, 2]) / 2,
            (dR[0, 2] - dR[2, 0]) / 2,
            (dR[1, 0] - dR[0, 1]) / 2
        ])
        J[3:6, j] = omega / (2 * np.deg2rad(delta))
    
    return J


def compute_position_jacobian(joint_angles: Dict[str, float], delta: float = 1e-4) -> np.ndarray:
    """
    Compute only the position part of the Jacobian (3x5 matrix).
    
    This is often sufficient when we only care about position, not orientation.
    
    Args:
        joint_angles: Current joint configuration (degrees)
        delta: Perturbation size
        
    Returns:
        3x5 position Jacobian matrix
    """
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    n_joints = len(joint_names)
    
    J = np.zeros((3, n_joints))
    
    for j, joint_name in enumerate(joint_names):
        angles_plus = joint_angles.copy()
        angles_minus = joint_angles.copy()
        
        angles_plus[joint_name] += delta
        angles_minus[joint_name] -= delta
        
        pos_plus, _ = get_forward_kinematics(angles_plus)
        pos_minus, _ = get_forward_kinematics(angles_minus)
        
        J[:, j] = (pos_plus - pos_minus) / (2 * np.deg2rad(delta))
    
    return J


# ==============================================================================
# NUMERICAL INVERSE KINEMATICS
# ==============================================================================

def inverse_kinematics_numerical(
    target_position: np.ndarray,
    initial_guess: Optional[Dict[str, float]] = None,
    max_iterations: int = 100,
    tolerance: float = 1e-4,
    damping: float = 0.01,
    position_only: bool = True,
    target_orientation: Optional[np.ndarray] = None,
    verbose: bool = False
) -> Tuple[Dict[str, float], bool, float]:
    """
    Compute inverse kinematics using damped least squares (Levenberg-Marquardt).
    
    The algorithm iteratively updates joint angles:
        q_new = q + J^T (J J^T + λ²I)^{-1} e
    
    where:
        - q is the current joint configuration
        - J is the Jacobian matrix
        - e is the position/orientation error
        - λ is the damping factor
    
    Args:
        target_position: Desired [x, y, z] position of tool frame
        initial_guess: Starting joint configuration (defaults to zeros)
        max_iterations: Maximum number of iterations
        tolerance: Convergence threshold for position error
        damping: Damping factor for singularity robustness
        position_only: If True, only consider position (3-DoF task)
        target_orientation: Desired 3x3 rotation matrix (if not position_only)
        verbose: Print iteration info
        
    Returns:
        Tuple of:
            - joint_config: Dictionary of joint angles in degrees
            - success: Whether convergence was achieved
            - final_error: Final position error norm
    """
    # Default initial guess (home position)
    if initial_guess is None:
        joint_config = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 0.0
        }
    else:
        joint_config = initial_guess.copy()
    
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    
    for iteration in range(max_iterations):
        # Compute current tool position
        current_position, current_rotation = get_forward_kinematics(joint_config)
        
        # Compute position error
        position_error = target_position - current_position
        error_norm = np.linalg.norm(position_error)
        
        if verbose:
            print(f"Iteration {iteration}: error = {error_norm:.6f}")
        
        # Check convergence
        if error_norm < tolerance:
            return joint_config, True, error_norm
        
        # Compute Jacobian
        if position_only:
            J = compute_position_jacobian(joint_config)
            error = position_error
        else:
            J = compute_jacobian(joint_config)
            # Combine position and orientation error
            if target_orientation is not None:
                # Compute orientation error using rotation matrix
                R_error = target_orientation @ current_rotation.T
                # Extract axis-angle representation
                angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))
                if angle < 1e-6:
                    orientation_error = np.zeros(3)
                else:
                    axis = np.array([
                        R_error[2, 1] - R_error[1, 2],
                        R_error[0, 2] - R_error[2, 0],
                        R_error[1, 0] - R_error[0, 1]
                    ]) / (2 * np.sin(angle))
                    orientation_error = angle * axis
                error = np.concatenate([position_error, orientation_error])
            else:
                error = np.concatenate([position_error, np.zeros(3)])
        
        # Damped least squares: Δq = J^T (J J^T + λ²I)^{-1} e
        JJT = J @ J.T
        damped = JJT + (damping ** 2) * np.eye(JJT.shape[0])
        delta_q = J.T @ np.linalg.solve(damped, error)
        
        # Convert delta_q from radians to degrees
        delta_q_deg = np.rad2deg(delta_q)
        
        # Update joint angles
        for i, joint_name in enumerate(joint_names):
            joint_config[joint_name] += delta_q_deg[i]
            
            # Apply joint limits (rough limits for SO-101)
            joint_config[joint_name] = np.clip(joint_config[joint_name], -180, 180)
    
    # Did not converge
    final_position, _ = get_forward_kinematics(joint_config)
    final_error = np.linalg.norm(target_position - final_position)
    return joint_config, False, final_error


# ==============================================================================
# GEOMETRIC INVERSE KINEMATICS
# ==============================================================================

def inverse_kinematics_geometric(
    target_position: np.ndarray,
    target_orientation: Optional[np.ndarray] = None,
    elbow_up: bool = True
) -> Tuple[Dict[str, float], bool]:
    """
    Compute inverse kinematics using geometric/analytical approach.
    
    This method exploits the specific geometry of the SO-101 arm.
    
    For a 5-DoF arm, we typically can only control either:
        - Position (3 DoF) + partial orientation (2 DoF)
        - Or position with some orientation constraint
    
    Geometric approach for position-only IK:
    1. Compute wrist center position (back-project from tool)
    2. Solve for shoulder_pan (θ1) using atan2
    3. Solve for shoulder_lift (θ2) and elbow_flex (θ3) using 2-link planar IK
    4. Solve for wrist angles to achieve desired orientation (if specified)
    
    Args:
        target_position: Desired [x, y, z] position
        target_orientation: Desired 3x3 rotation matrix (optional)
        elbow_up: Choose elbow-up solution (True) or elbow-down (False)
        
    Returns:
        Tuple of (joint_config, success)
    """
    # Initialize output
    joint_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    # Target in world frame
    x_t, y_t, z_t = target_position
    
    # ==========================================================================
    # Step 1: Solve for shoulder_pan (θ1)
    # ==========================================================================
    # The shoulder pan rotates about Z-axis
    # θ1 = atan2(y_t, x_t) adjusted for frame orientation
    
    # Account for base offset
    x_adj = x_t - SO101Params.BASE_TO_J1_X
    
    # The frame is rotated 180° about Z and X from world
    # So we need to account for this in our calculation
    theta1 = np.rad2deg(np.arctan2(-y_t, -x_adj))
    joint_config['shoulder_pan'] = theta1
    
    # ==========================================================================
    # Step 2: Solve for shoulder_lift (θ2) and elbow_flex (θ3)
    # ==========================================================================
    # Project into the plane of the arm (2-link planar IK)
    
    # Distance from shoulder to target in XY plane (after rotation by θ1)
    r_xy = np.sqrt(x_adj**2 + y_t**2)
    
    # Height from shoulder to target
    z_shoulder = SO101Params.BASE_TO_J1_Z + SO101Params.J1_OFFSET_Z + SO101Params.J1_TO_J2_Z
    z_rel = z_t - z_shoulder
    
    # Effective link lengths for 2-link IK
    # Link 1: shoulder to elbow (vertical portion)
    L1 = SO101Params.J2_TO_J3_Z + SO101Params.J3_OFFSET_Z  # ~0.14057
    
    # Link 2: elbow to wrist (horizontal portion)
    L2 = SO101Params.J3_TO_J4_X + SO101Params.J4_TO_J5_X + SO101Params.J5_TO_TOOL_X  # ~0.2994
    
    # Distance from shoulder to target in the arm plane
    D = np.sqrt(r_xy**2 + z_rel**2)
    
    # Check if target is reachable
    if D > L1 + L2:
        print(f"Warning: Target is outside workspace (D={D:.4f} > L1+L2={L1+L2:.4f})")
        # Scale down to maximum reach
        scale = (L1 + L2 - 0.01) / D
        r_xy *= scale
        z_rel *= scale
        D = L1 + L2 - 0.01
    
    if D < abs(L1 - L2):
        print(f"Warning: Target is inside workspace hole (D={D:.4f} < |L1-L2|={abs(L1-L2):.4f})")
        return joint_config, False
    
    # Law of cosines to find elbow angle
    cos_elbow = (L1**2 + L2**2 - D**2) / (2 * L1 * L2)
    cos_elbow = np.clip(cos_elbow, -1, 1)  # Numerical safety
    
    if elbow_up:
        theta3 = np.rad2deg(np.arccos(cos_elbow)) - 180  # Elbow flex
    else:
        theta3 = 180 - np.rad2deg(np.arccos(cos_elbow))
    
    # Shoulder angle (using law of cosines and geometry)
    cos_shoulder = (L1**2 + D**2 - L2**2) / (2 * L1 * D)
    cos_shoulder = np.clip(cos_shoulder, -1, 1)
    
    alpha = np.arctan2(z_rel, r_xy)  # Angle to target
    beta = np.arccos(cos_shoulder)   # Angle from link 1 to line to target
    
    if elbow_up:
        theta2 = np.rad2deg(alpha + beta) - 90
    else:
        theta2 = np.rad2deg(alpha - beta) - 90
    
    joint_config['shoulder_lift'] = theta2
    joint_config['elbow_flex'] = theta3
    
    # ==========================================================================
    # Step 3: Solve for wrist angles (θ4, θ5)
    # ==========================================================================
    # For position-only IK, we can set these to achieve a specific tool orientation
    # Default: point the tool perpendicular to the arm plane
    
    # The wrist flex should compensate for arm angles to keep tool level
    theta4 = -(theta2 + theta3)  # Keep tool horizontal
    joint_config['wrist_flex'] = theta4
    
    # Wrist roll can be set to desired value (default 0)
    joint_config['wrist_roll'] = 0.0
    
    # ==========================================================================
    # Verify solution
    # ==========================================================================
    result_pos, _ = get_forward_kinematics(joint_config)
    error = np.linalg.norm(result_pos - target_position)
    success = error < 0.01  # 1cm tolerance
    
    return joint_config, success


# ==============================================================================
# HIGH-LEVEL IK FUNCTION
# ==============================================================================

def get_inverse_kinematics(
    target_position: np.ndarray,
    target_orientation: Optional[np.ndarray] = None,
    method: str = 'numerical',
    initial_guess: Optional[Dict[str, float]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Main inverse kinematics function.
    
    Args:
        target_position: Desired [x, y, z] tool position
        target_orientation: Desired 3x3 rotation matrix (optional)
        method: 'numerical' or 'geometric'
        initial_guess: Starting configuration for numerical method
        **kwargs: Additional arguments passed to the specific method
        
    Returns:
        Dictionary of joint angles in degrees
    """
    if method == 'geometric':
        joint_config, success = inverse_kinematics_geometric(
            target_position,
            target_orientation,
            elbow_up=kwargs.get('elbow_up', True)
        )
        if not success:
            print("Geometric IK failed, falling back to numerical...")
            method = 'numerical'
    
    if method == 'numerical':
        joint_config, success, error = inverse_kinematics_numerical(
            target_position,
            initial_guess=initial_guess,
            target_orientation=target_orientation,
            **kwargs
        )
        if not success:
            print(f"Warning: IK did not converge. Final error: {error:.6f}")
    
    return joint_config


# ==============================================================================
# MAIN - Test the implementation
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SO-101 Inverse Kinematics Test")
    print("=" * 60)
    
    # Test 1: Compute FK for a known configuration
    test_angles = {
        'shoulder_pan': -45.0,
        'shoulder_lift': 45.0,
        'elbow_flex': -45.0,
        'wrist_flex': 90.0,
        'wrist_roll': 0.0,
        'gripper': 0.0
    }
    
    print("\n1. Forward Kinematics (known configuration):")
    print(f"   Input angles: {test_angles}")
    position, rotation = get_forward_kinematics(test_angles)
    print(f"   FK result: x={position[0]:.4f}, y={position[1]:.4f}, z={position[2]:.4f}")
    
    # Test 2: Numerical IK to recover the configuration
    print("\n2. Numerical IK (recover configuration):")
    recovered_angles, success, error = inverse_kinematics_numerical(
        position,
        initial_guess=None,  # Start from zeros
        verbose=False
    )
    print(f"   Target position: {position}")
    print(f"   Success: {success}, Final error: {error:.6f}")
    print(f"   Recovered angles: {recovered_angles}")
    
    # Verify the recovered solution
    verify_pos, _ = get_forward_kinematics(recovered_angles)
    print(f"   Verification FK: x={verify_pos[0]:.4f}, y={verify_pos[1]:.4f}, z={verify_pos[2]:.4f}")
    
    # Test 3: Geometric IK
    print("\n3. Geometric IK:")
    target = np.array([0.2, 0.15, 0.15])
    print(f"   Target position: {target}")
    
    geo_angles, success = inverse_kinematics_geometric(target, elbow_up=True)
    print(f"   Success: {success}")
    print(f"   Joint angles: {geo_angles}")
    
    verify_pos2, _ = get_forward_kinematics(geo_angles)
    geo_error = np.linalg.norm(verify_pos2 - target)
    print(f"   Verification FK: x={verify_pos2[0]:.4f}, y={verify_pos2[1]:.4f}, z={verify_pos2[2]:.4f}")
    print(f"   Error: {geo_error:.6f}")
    
    # Test 4: Jacobian computation
    print("\n4. Jacobian Test:")
    J = compute_position_jacobian(test_angles)
    print(f"   Position Jacobian shape: {J.shape}")
    print(f"   Jacobian:\n{J}")
    
    print("\n" + "=" * 60)
    print("Inverse Kinematics Test Complete!")
    print("=" * 60)
