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

PEDAGOGICAL ENHANCEMENTS:
    - Explicit wrist position calculation using transforms (links FK to IK)
    - θ5 = -θ1 to compensate shoulder rotation 
    - Shows how IK reverses FK transformations

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
        ẋ = J(q) · q̇
    
    For a 5-DoF arm, J is a 6×5 matrix:
        - First 3 rows: position Jacobian (dx/dq)
        - Last 3 rows: orientation Jacobian (dω/dq)
    
    Args:
        joint_angles: Current joint configuration (degrees)
        delta: Perturbation size for finite differences
        
    Returns:
        6x5 Jacobian matrix
    """
    joint_names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']
    n_joints = len(joint_names)
    
    J = np.zeros((6, n_joints))
    
    for j, joint_name in enumerate(joint_names):
        angles_plus = joint_angles.copy()
        angles_minus = joint_angles.copy()
        
        angles_plus[joint_name] += delta
        angles_minus[joint_name] -= delta
        
        pos_plus, rot_plus = get_forward_kinematics(angles_plus)
        pos_minus, rot_minus = get_forward_kinematics(angles_minus)
        
        # Position derivative
        J[0:3, j] = (pos_plus - pos_minus) / (2 * np.deg2rad(delta))
        
        # Orientation derivative (simplified: using rotation matrix difference)
        R_diff = rot_plus @ rot_minus.T
        angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1))
        if angle > 1e-6:
            axis = np.array([
                R_diff[2, 1] - R_diff[1, 2],
                R_diff[0, 2] - R_diff[2, 0],
                R_diff[1, 0] - R_diff[0, 1]
            ]) / (2 * np.sin(angle))
            J[3:6, j] = (angle * axis) / (2 * np.deg2rad(delta))
    
    return J


def compute_position_jacobian(joint_angles: Dict[str, float], delta: float = 1e-4) -> np.ndarray:
    """
    Compute position-only Jacobian (3×5 matrix).
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
# WRIST POSITION CALCULATION (Pedagogical: Links FK to IK)
# ==============================================================================

def get_wrist_position(target_position: np.ndarray, 
                       target_orientation: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute desired wrist frame position from desired tool position.
    
    PEDAGOGICAL PURPOSE: This function demonstrates how IK uses the INVERSE 
    of FK transforms. Students can see the direct connection:
    
    In FK (Lab 1): We built up transforms
        g_wt = g_w1 @ g_12 @ g_23 @ g_34 @ g_45 @ g_5t
        
    In IK (Lab 2): We reverse the process
        g_w4 = g_wt @ inv(g_4t)
        where g_4t = g_45 @ g_5t
    
    This is the KEY INSIGHT: IK "undoes" the last transformation to find where
    the wrist needs to be to place the tool at the target position.
    
    Args:
        target_position: Desired [x, y, z] tool position in world frame
        target_orientation: Desired 3x3 rotation matrix (if None, assumes vertical grasp)
        
    Returns:
        Tuple of (wrist_position, wrist_orientation)
        - wrist_position: [x, y, z] position of wrist frame (joint 4)
        - wrist_orientation: 3x3 rotation matrix of wrist frame
        
    Example:
        If tool should be at [0.2, 0.1, 0.15], this computes where joint 4
        (wrist) needs to be, accounting for the fixed wrist-to-tool distance.
    """
    # Step 1: Construct g_wt (world to tool transform)
    if target_orientation is None:
        # Default: vertical grasp (tool points down, aligned with world frame)
        target_orientation = np.eye(3)
    
    g_wt = make_transform(target_orientation, target_position)
    
    # Step 2: Construct g_4t (joint 4 to tool transform)
    # This is the FIXED transform from wrist to tool (same as in FK!)
    # Path: wrist (J4) → J5 → tool
    
    # Joint 4 to Joint 5 (translation along X-axis)
    g_45 = make_transform(np.eye(3), np.array([SO101Params.J4_TO_J5_X, 0, 0]))
    
    # Joint 5 to Tool (translation along X-axis)
    g_5t = make_transform(np.eye(3), np.array([SO101Params.J5_TO_TOOL_X, 0, 0]))
    
    # Combined: Joint 4 to Tool
    g_4t = g_45 @ g_5t
    
    # Step 3: Back-project to find wrist frame
    # Mathematical relationship: g_wt = g_w4 @ g_4t
    # Solve for g_w4: g_w4 = g_wt @ inv(g_4t)
    g_w4 = g_wt @ np.linalg.inv(g_4t)
    
    # Extract position and orientation of wrist frame
    wrist_position = g_w4[0:3, 3]
    wrist_orientation = g_w4[0:3, 0:3]
    
    return wrist_position, wrist_orientation


# ==============================================================================
# GEOMETRIC INVERSE KINEMATICS (Following Assignment Pedagogy)
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
    
    Geometric IK Pipeline (following assignment pedagogy):
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Step 0: Compute wrist position using g_w4 = g_wt @ inv(g_4t)
    Step 1: Solve for shoulder_pan (θ1) using atan2 on wrist x-y projection  
    Step 2: Solve for shoulder_lift (θ2) and elbow_flex (θ3) using 2-link planar IK
            with law of cosines
    Step 3: Solve for wrist_flex (θ4) to keep tool oriented correctly
    Step 4: Solve for wrist_roll (θ5) = -θ1 to compensate shoulder rotation
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    Args:
        target_position: Desired [x, y, z] position of tool frame
        target_orientation: Desired 3x3 rotation matrix (optional)
        elbow_up: Choose elbow-up solution (True) or elbow-down (False)
        
    Returns:
        Tuple of (joint_config, success)
        
    Example:
        >>> target = np.array([0.2, 0.1, 0.15])
        >>> config, success = inverse_kinematics_geometric(target)
        >>> if success:
        ...     print(f"Joint angles: {config}")
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
    
    # ==========================================================================
    # Step 0: Compute wrist position (back-projection from tool)
    # ==========================================================================
    # PEDAGOGICAL LINK: We use FK transforms in reverse!
    # This explicitly shows students how IK undoes FK transformations
    wrist_position, wrist_orientation = get_wrist_position(target_position, target_orientation)
    
    # Use wrist position for IK (this is our actual target for the 3-link arm)
    x_t, y_t, z_t = wrist_position
    
    # ==========================================================================
    # Step 1: Solve for shoulder_pan (θ1)
    # ==========================================================================
    # The shoulder pan rotates about Z-axis to align with the target
    # We project the wrist position onto the x-y plane
    
    # Account for base offset (frame is not at origin)
    x_adj = x_t - SO101Params.BASE_TO_J1_X
    
    # The SO-101 frame is rotated 180° about Z and X from world frame
    # So we need to account for this in our calculation
    theta1 = np.rad2deg(np.arctan2(-y_t, -x_adj))
    joint_config['shoulder_pan'] = theta1
    
    # ==========================================================================
    # Step 2: Solve for shoulder_lift (θ2) and elbow_flex (θ3)
    # ==========================================================================
    # Project into the plane of the arm for 2-link planar IK
    # This becomes a classic 2-link IK problem solved with law of cosines
    
    # Distance from shoulder to target in XY plane (after rotation by θ1)
    r_xy = np.sqrt(x_adj**2 + y_t**2)
    
    # Height from shoulder to target
    z_shoulder = SO101Params.BASE_TO_J1_Z + SO101Params.J1_OFFSET_Z + SO101Params.J1_TO_J2_Z
    z_rel = z_t - z_shoulder
    
    # Effective link lengths for 2-link IK
    # Link 1: shoulder to elbow (vertical portion)
    L1 = SO101Params.J2_TO_J3_Z + SO101Params.J3_OFFSET_Z  # ~0.14057 m
    
    # Link 2: elbow to wrist (horizontal portion)  
    L2 = SO101Params.J3_TO_J4_X  # ~0.1349 m (to wrist, not tool!)
    
    # Distance from shoulder to wrist in the arm plane
    D = np.sqrt(r_xy**2 + z_rel**2)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # WORKSPACE CHECKING (Improvement #1: Safety)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    max_reach = L1 + L2
    min_reach = abs(L1 - L2)
    
    if D > max_reach:
        print(f"⚠ Warning: Target is outside workspace")
        print(f"   Distance to wrist: {D:.4f} m > Max reach: {max_reach:.4f} m")
        # Scale down to maximum reach (with small safety margin)
        scale = (max_reach - 0.01) / D
        r_xy *= scale
        z_rel *= scale
        D = max_reach - 0.01
        print(f"   Scaled target to: D = {D:.4f} m")
    
    if D < min_reach:
        print(f"⚠ Warning: Target is inside workspace hole")
        print(f"   Distance: {D:.4f} m < Min reach: {min_reach:.4f} m")
        return joint_config, False
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # LAW OF COSINES (Classic 2-Link IK)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # For triangle with sides L1, L2, D:
    # cos(elbow_angle) = (L1² + L2² - D²) / (2·L1·L2)
    
    cos_elbow = (L1**2 + L2**2 - D**2) / (2 * L1 * L2)
    cos_elbow = np.clip(cos_elbow, -1, 1)  # Improvement #2: Numerical safety
    
    # Elbow angle (θ3)
    if elbow_up:
        # Elbow-up configuration (elbow points upward)
        theta3 = np.rad2deg(np.arccos(cos_elbow)) - 180
    else:
        # Elbow-down configuration (elbow points downward)
        theta3 = 180 - np.rad2deg(np.arccos(cos_elbow))
    
    # Shoulder angle (θ2) using law of cosines and geometry
    cos_shoulder = (L1**2 + D**2 - L2**2) / (2 * L1 * D)
    cos_shoulder = np.clip(cos_shoulder, -1, 1)  # Numerical safety
    
    alpha = np.arctan2(z_rel, r_xy)  # Angle to target in arm plane
    beta = np.arccos(cos_shoulder)    # Angle from link 1 to line to target
    
    if elbow_up:
        theta2 = np.rad2deg(alpha + beta) - 90
    else:
        theta2 = np.rad2deg(alpha - beta) - 90
    
    joint_config['shoulder_lift'] = theta2
    joint_config['elbow_flex'] = theta3
    
    # ==========================================================================
    # Step 3: Solve for wrist_flex (θ4)
    # ==========================================================================
    # The wrist flex should compensate for arm angles to keep tool oriented
    # For vertical grasp (tool pointing down), we want: θ2 + θ3 + θ4 = 0
    # Therefore: θ4 = -(θ2 + θ3)
    
    theta4 = -(theta2 + theta3)  # Keep tool horizontal/vertical
    joint_config['wrist_flex'] = theta4
    
    # ==========================================================================
    # Step 4: Solve for wrist_roll (θ5)
    # ==========================================================================
    # ASSIGNMENT APPROACH: θ5 = -θ1
    # 
    # Why? The wrist roll should compensate for shoulder pan rotation to keep
    # the gripper aligned with the object. Since shoulder_pan rotates the entire
    # arm about the Z-axis, we need to counter-rotate the wrist by the same amount.
    #
    # For a cube/object with symmetry, we can use θ5 = -θ1 to maintain alignment
    
    theta5 = -theta1  # Compensate shoulder pan rotation
    joint_config['wrist_roll'] = theta5
    
    # ==========================================================================
    # Verify solution (Improvement #3: Validation)
    # ==========================================================================
    result_pos, _ = get_forward_kinematics(joint_config)
    error = np.linalg.norm(result_pos - target_position)
    success = error < 0.01  # 1 cm tolerance
    
    if not success:
        print(f"⚠ Warning: FK verification failed")
        print(f"   Target: {target_position}")
        print(f"   Achieved: {result_pos}")
        print(f"   Error: {error:.6f} m")
    
    return joint_config, success


# ==============================================================================
# NUMERICAL INVERSE KINEMATICS (Improvement #4: Robust Fallback)
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
    
    This method is MORE ROBUST than geometric IK because:
        - Works even when geometric solution doesn't exist
        - Handles singularities gracefully (via damping)
        - Can incorporate orientation constraints
        - Generalizes to any manipulator structure
    
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
# HIGH-LEVEL IK FUNCTION (Improvement #5: Automatic Method Selection)
# ==============================================================================

def get_inverse_kinematics(
    target_position: np.ndarray,
    target_orientation: Optional[np.ndarray] = None,
    method: str = 'numerical',
    initial_guess: Optional[Dict[str, float]] = None,
    **kwargs
) -> Dict[str, float]:
    """
    Main inverse kinematics function with automatic fallback.
    
    This function provides a unified interface to both geometric and numerical IK,
    with automatic fallback if the first method fails.
    
    Args:
        target_position: Desired [x, y, z] tool position
        target_orientation: Desired 3x3 rotation matrix (optional)
        method: 'numerical' or 'geometric' (default: 'numerical')
        initial_guess: Starting configuration for numerical method
        **kwargs: Additional arguments passed to the specific method
        
    Returns:
        Dictionary of joint angles in degrees
        
    Example:
        >>> # Try geometric first, fallback to numerical
        >>> config = get_inverse_kinematics(
        ...     target_position=np.array([0.2, 0.1, 0.15]),
        ...     method='geometric'
        ... )
    """
    if method == 'geometric':
        joint_config, success = inverse_kinematics_geometric(
            target_position,
            target_orientation,
            elbow_up=kwargs.get('elbow_up', True)
        )
        if not success:
            print("⚠ Geometric IK failed, falling back to numerical...")
            method = 'numerical'
    
    if method == 'numerical':
        joint_config, success, error = inverse_kinematics_numerical(
            target_position,
            initial_guess=initial_guess,
            target_orientation=target_orientation,
            **kwargs
        )
        if not success:
            print(f"⚠ Warning: IK did not converge. Final error: {error:.6f}")
    
    return joint_config


# ==============================================================================
# MAIN - Test the implementation
# ==============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SO-101 Inverse Kinematics Test")
    print("=" * 70)
    
    # Test 1: Wrist position calculation (pedagogical demo)
    print("\n" + "-" * 70)
    print("Test 1: Wrist Position Calculation (FK → IK Link)")
    print("-" * 70)
    
    tool_target = np.array([0.25, 0.10, 0.15])
    print(f"Desired tool position: {tool_target}")
    
    wrist_pos, wrist_rot = get_wrist_position(tool_target)
    print(f"Computed wrist position: {wrist_pos}")
    print(f"This shows how IK 'backs up' from tool to wrist frame")
    
    # Test 2: Compute FK for a known configuration
    print("\n" + "-" * 70)
    print("Test 2: Forward-then-Inverse (Roundtrip Test)")
    print("-" * 70)
    
    test_angles = {
        'shoulder_pan': -45.0,
        'shoulder_lift': 45.0,
        'elbow_flex': -45.0,
        'wrist_flex': 90.0,
        'wrist_roll': 45.0,  # Note: will be recovered as -shoulder_pan
        'gripper': 0.0
    }
    
    print(f"Original angles: {test_angles}")
    position, rotation = get_forward_kinematics(test_angles)
    print(f"FK result: [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
    
    # Test 3: Geometric IK
    print("\n" + "-" * 70)
    print("Test 3: Geometric IK (Assignment Method)")
    print("-" * 70)
    
    target = np.array([0.20, 0.10, 0.15])
    print(f"Target position: {target}")
    
    geo_config, geo_success = inverse_kinematics_geometric(target, elbow_up=True)
    print(f"Success: {geo_success}")
    if geo_success:
        print(f"Joint angles:")
        for joint, angle in geo_config.items():
            if joint != 'gripper':
                print(f"  {joint}: {angle:.2f}°")
        
        # Verify
        verify_pos, _ = get_forward_kinematics(geo_config)
        error = np.linalg.norm(verify_pos - target)
        print(f"\nFK Verification:")
        print(f"  Achieved: [{verify_pos[0]:.4f}, {verify_pos[1]:.4f}, {verify_pos[2]:.4f}]")
        print(f"  Error: {error:.6f} m")
        print(f"  Note: θ5 (wrist_roll) = {geo_config['wrist_roll']:.2f}° = -θ1")
    
    # Test 4: Numerical IK
    print("\n" + "-" * 70)
    print("Test 4: Numerical IK (Jacobian Method)")
    print("-" * 70)
    
    num_config, num_success, num_error = inverse_kinematics_numerical(
        target,
        verbose=False
    )
    print(f"Success: {num_success}")
    print(f"Final error: {num_error:.6f} m")
    print(f"Joint angles:")
    for joint, angle in num_config.items():
        if joint != 'gripper':
            print(f"  {joint}: {angle:.2f}°")
    
    # Test 5: Jacobian computation
    print("\n" + "-" * 70)
    print("Test 5: Jacobian Computation")
    print("-" * 70)
    
    J_pos = compute_position_jacobian(test_angles)
    print(f"Position Jacobian shape: {J_pos.shape}")
    print(f"Jacobian matrix:\n{J_pos}")
    
    print("\n" + "=" * 70)
    print("All Tests Complete!")
    print("=" * 70)
    print("\nKey Pedagogical Points:")
    print("  1. Wrist position is computed using: g_w4 = g_wt @ inv(g_4t)")
    print("  2. This directly links FK transforms to IK back-projection")
    print("  3. θ5 (wrist_roll) = -θ1 compensates shoulder rotation")
    print("  4. Geometric IK is fast but limited to reachable workspace")
    print("  5. Numerical IK is robust and handles edge cases")