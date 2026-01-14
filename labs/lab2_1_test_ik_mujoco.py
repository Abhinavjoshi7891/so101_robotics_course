#!/usr/bin/env python3
"""
Lab 2.1: Test Inverse Kinematics in MuJoCo
==========================================

This lab tests the inverse kinematics implementation by:
1. Specifying a target position for the tool frame
2. Computing the joint angles using IK
3. Visualizing the robot reaching the target in MuJoCo

Learning Objectives:
    1. Understand the IK problem and its solutions
    2. Compare geometric vs numerical IK approaches
    3. Visualize IK solutions and understand workspace limitations

Usage:
    python labs/lab2_1_test_ik_mujoco.py
    python labs/lab2_1_test_ik_mujoco.py --multi

Author: SO-101 Robotics Course
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("Error: MuJoCo not installed. Install with: pip install mujoco")
    sys.exit(1)

from so101_forward_kinematics import get_forward_kinematics
from so101_inverse_kinematics import (
    get_inverse_kinematics,
    inverse_kinematics_numerical,
    inverse_kinematics_geometric
)
from so101_mujoco_utils import (
    set_initial_pose,
    send_position_command,
    move_to_pose,
    hold_position,
    show_cube,
    show_sphere,
    degrees_to_mujoco
)


def get_mujoco_gripper_position(model, data):
    """Get the actual gripper position from MuJoCo simulation."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
    if site_id >= 0:
        return data.site_xpos[site_id].copy()
    return None


def show_target_cube(viewer, position, halfwidth=0.015, rgba=[1, 0, 0, 0.5]):
    """Show a semi-transparent red cube at the target position."""
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[halfwidth, halfwidth, halfwidth],
        pos=np.array(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.array(rgba, dtype=np.float32)
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()


def show_sphere_at_fk(viewer, position, rgba=[0, 1, 0, 0.9], radius=0.02):
    """Show a green sphere at the FK-computed position."""
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.array(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.array(rgba, dtype=np.float32)
    )
    viewer.user_scn.ngeom = 2
    viewer.sync()


def interpolate_configs(config_start, config_target, alpha):
    """Linearly interpolate between two joint configurations."""
    result = {}
    for key in config_start.keys():
        start_val = config_start[key]
        target_val = config_target.get(key, 0)
        result[key] = start_val + alpha * (target_val - start_val)
    return result


def run_ik_test():
    """Main IK test function."""
    
    print("=" * 80)
    print("Lab 2.1: Inverse Kinematics Test in MuJoCo")
    print("=" * 80)
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # IMPORTANT: Use mode='mujoco' for MuJoCo simulations
    FK_MODE = 'mujoco'
    IK_MODE = 'mujoco'
    
    print(f"\nUsing FK/IK mode: {FK_MODE}")
    print("(MuJoCo and URDF use different coordinate frame conventions)")
    
    # ==========================================================================
    # Test 1: Forward-then-Inverse
    # ==========================================================================
    print("\n" + "-" * 80)
    print("Test 1: Forward Kinematics → Inverse Kinematics")
    print("-" * 80)
    print("Starting from a known configuration, we compute FK,")
    print("then use IK to recover the joint angles.")
    
    # Known configuration
    original_config = {
        'shoulder_pan': -30.0,
        'shoulder_lift': 45.0,
        'elbow_flex': -30.0,
        'wrist_flex': 60.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    
    print(f"\nOriginal joint angles:")
    for joint, angle in original_config.items():
        if joint != 'gripper':
            print(f"  {joint:15s}: {angle:7.2f}°")
    
    # Compute FK
    target_position, target_rotation = get_forward_kinematics(original_config, mode=FK_MODE)
    print(f"\nFK result: [{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}]")
    
    # Compute IK (numerical method)
    print("\nRunning numerical IK...")
    recovered_config, success, error = inverse_kinematics_numerical(
        target_position,
        initial_guess=None,  # Start from home
        verbose=True,
        max_iterations=50,
        mode=IK_MODE
    )
    
    print(f"\nIK Result:")
    print(f"  Success: {success}")
    print(f"  Final error: {error:.6f} m ({error*1000:.2f} mm)")
    print(f"  Recovered angles:")
    for joint, angle in recovered_config.items():
        if joint != 'gripper':
            print(f"    {joint:15s}: {angle:7.2f}°")
    
    # Verify with FK
    verify_pos, _ = get_forward_kinematics(recovered_config, mode=FK_MODE)
    fk_error = np.linalg.norm(verify_pos - target_position)
    print(f"\n  FK Verification:")
    print(f"    Target:   [{target_position[0]:.4f}, {target_position[1]:.4f}, {target_position[2]:.4f}]")
    print(f"    Achieved: [{verify_pos[0]:.4f}, {verify_pos[1]:.4f}, {verify_pos[2]:.4f}]")
    print(f"    Error: {fk_error:.6f} m ({fk_error*1000:.2f} mm)")
    
    # ==========================================================================
    # Test 2: Reach to arbitrary target
    # ==========================================================================
    print("\n" + "-" * 80)
    print("Test 2: Reach to Arbitrary Target")
    print("-" * 80)
    
    # Target position (should be within workspace for MuJoCo)
    target_position_2 = np.array([0.25, 0.10, 0.20])
    print(f"\nTarget position: [{target_position_2[0]:.4f}, {target_position_2[1]:.4f}, {target_position_2[2]:.4f}]")
    
    # Try geometric IK
    print("\n1. Geometric IK:")
    geo_config, geo_success = inverse_kinematics_geometric(target_position_2, mode=IK_MODE)
    print(f"   Success: {geo_success}")
    if geo_success:
        geo_verify, _ = get_forward_kinematics(geo_config, mode=FK_MODE)
        geo_error = np.linalg.norm(geo_verify - target_position_2)
        print(f"   Error: {geo_error:.6f} m ({geo_error*1000:.2f} mm)")
        print(f"   Joint angles:")
        for joint, angle in geo_config.items():
            if joint != 'gripper':
                print(f"     {joint:15s}: {angle:7.2f}°")
    
    # Try numerical IK
    print("\n2. Numerical IK:")
    num_config, num_success, num_error = inverse_kinematics_numerical(
        target_position_2,
        verbose=False,
        mode=IK_MODE
    )
    print(f"   Success: {num_success}")
    print(f"   Error: {num_error:.6f} m ({num_error*1000:.2f} mm)")
    print(f"   Joint angles:")
    for joint, angle in num_config.items():
        if joint != 'gripper':
            print(f"     {joint:15s}: {angle:7.2f}°")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    print("\n" + "=" * 80)
    print("Starting MuJoCo Visualization")
    print("=" * 80)
    print("\nThe robot will move to reach the target.")
    print("  RED cube   = IK target position")
    print("  GREEN sphere = FK-computed position (should overlap red cube)")
    print("\nPress Ctrl+C or close the window to exit.\n")
    
    # Start from home position
    initial_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    
    set_initial_pose(data, initial_config, use_degrees=True)
    send_position_command(data, initial_config, use_degrees=True)
    
    # Step to settle
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    # Use the better IK result
    if num_success:
        ik_result = num_config
    elif geo_success:
        ik_result = geo_config
    else:
        print("Warning: Neither IK method succeeded. Using best effort.")
        ik_result = num_config
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("Phase 1: Moving to target position...")
        
        # Move to IK solution
        start_time = time.time()
        move_duration = 3.0
        
        while viewer.is_running() and (time.time() - start_time) < move_duration:
            elapsed = time.time() - start_time
            t = elapsed / move_duration
            s = 3 * t**2 - 2 * t**3  # Smooth interpolation
            
            # Interpolate joint angles
            interp_config = interpolate_configs(initial_config, ik_result, s)
            
            send_position_command(data, interp_config, use_degrees=True)
            mujoco.mj_step(model, data)
            
            # Compute FK for current config
            fk_pos, _ = get_forward_kinematics(interp_config, mode=FK_MODE)
            
            # Show target cube and FK sphere
            show_target_cube(viewer, target_position_2)
            show_sphere_at_fk(viewer, fk_pos)
            
            # Get MuJoCo position
            mujoco_pos = get_mujoco_gripper_position(model, data)
            if mujoco_pos is not None:
                mujoco_error = np.linalg.norm(mujoco_pos - target_position_2) * 1000.0
                fk_error = np.linalg.norm(fk_pos - target_position_2) * 1000.0
                print(f"\r[MOVING {int(s*100):3d}%] "
                      f"FK_to_Target: {fk_error:6.2f}mm | "
                      f"MuJoCo_to_Target: {mujoco_error:6.2f}mm", end="")
            
            viewer.sync()
            time.sleep(0.01)
        
        print("\n\nPhase 2: Holding at target...")
        
        # Hold at target
        hold_start = time.time()
        while viewer.is_running() and (time.time() - hold_start) < 10.0:
            send_position_command(data, ik_result, use_degrees=True)
            mujoco.mj_step(model, data)
            
            # Compute FK
            fk_pos, _ = get_forward_kinematics(ik_result, mode=FK_MODE)
            
            # Show markers
            show_target_cube(viewer, target_position_2)
            show_sphere_at_fk(viewer, fk_pos)
            
            # Get MuJoCo position
            mujoco_pos = get_mujoco_gripper_position(model, data)
            if mujoco_pos is not None:
                mujoco_error = np.linalg.norm(mujoco_pos - target_position_2) * 1000.0
                fk_error = np.linalg.norm(fk_pos - target_position_2) * 1000.0
                print(f"\r[HOLDING] "
                      f"FK_to_Target: {fk_error:6.2f}mm | "
                      f"MuJoCo_to_Target: {mujoco_error:6.2f}mm", end="")
            
            viewer.sync()
            time.sleep(0.01)
    
    print("\n\n" + "=" * 80)
    print("IK Test Complete!")
    print("=" * 80)


def run_multiple_targets_test():
    """Test IK with multiple target positions."""
    
    print("\n" + "=" * 80)
    print("Multiple Targets IK Test")
    print("=" * 80)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    FK_MODE = 'mujoco'
    IK_MODE = 'mujoco'
    
    # Define multiple target positions (within MuJoCo workspace)
    targets = [
        np.array([0.30, 0.00, 0.25]),   # Front
        np.array([0.25, 0.15, 0.20]),   # Front-left
        np.array([0.25, -0.15, 0.20]),  # Front-right
        np.array([0.20, 0.20, 0.28]),   # Left-high
        np.array([0.20, -0.20, 0.28]),  # Right-high
    ]
    
    target_colors = [
        [1, 0, 0, 0.5],   # Red
        [0, 1, 0, 0.5],   # Green
        [0, 0, 1, 0.5],   # Blue
        [1, 1, 0, 0.5],   # Yellow
        [1, 0, 1, 0.5],   # Magenta
    ]
    
    # Compute IK for each target
    ik_solutions = []
    print("\nComputing IK solutions:")
    for i, target in enumerate(targets):
        config, success, error = inverse_kinematics_numerical(target, verbose=False, mode=IK_MODE)
        ik_solutions.append(config)
        print(f"  Target {i+1}: pos=[{target[0]:.3f}, {target[1]:.3f}, {target[2]:.3f}], "
              f"success={success}, error={error*1000:.2f}mm")
    
    # Initialize
    initial_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    
    set_initial_pose(data, initial_config, use_degrees=True)
    
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    print("\nStarting visualization - robot will visit each target...")
    print("Each target is marked with a colored cube")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        current_config = initial_config.copy()
        
        # Visit each target
        for target_idx, (target, target_config) in enumerate(zip(targets, ik_solutions)):
            if not viewer.is_running():
                break
                
            print(f"\n→ Moving to target {target_idx + 1}/{len(targets)}...")
            
            # Smooth move to target
            start_time = time.time()
            move_duration = 2.5
            
            while viewer.is_running() and (time.time() - start_time) < move_duration:
                elapsed = time.time() - start_time
                t = elapsed / move_duration
                s = 3 * t**2 - 2 * t**3
                
                interp_config = interpolate_configs(current_config, target_config, s)
                
                send_position_command(data, interp_config, use_degrees=True)
                mujoco.mj_step(model, data)
                
                # Show all target cubes
                for i, (tgt, color) in enumerate(zip(targets, target_colors)):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.015, 0.015, 0.015],
                        pos=np.array(tgt, dtype=np.float64),
                        mat=np.eye(3, dtype=np.float64).flatten(),
                        rgba=np.array(color, dtype=np.float32)
                    )
                viewer.user_scn.ngeom = len(targets)
                
                viewer.sync()
                time.sleep(0.01)
            
            # Brief pause at target
            hold_start = time.time()
            while viewer.is_running() and (time.time() - hold_start) < 1.0:
                send_position_command(data, target_config, use_degrees=True)
                mujoco.mj_step(model, data)
                
                # Show all target cubes
                for i, (tgt, color) in enumerate(zip(targets, target_colors)):
                    mujoco.mjv_initGeom(
                        viewer.user_scn.geoms[i],
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.015, 0.015, 0.015],
                        pos=np.array(tgt, dtype=np.float64),
                        mat=np.eye(3, dtype=np.float64).flatten(),
                        rgba=np.array(color, dtype=np.float32)
                    )
                viewer.user_scn.ngeom = len(targets)
                
                viewer.sync()
                time.sleep(0.01)
            
            current_config = target_config.copy()
        
        # Return to home
        print("\n→ Returning to home...")
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < 2.0:
            elapsed = time.time() - start_time
            t = elapsed / 2.0
            s = 3 * t**2 - 2 * t**3
            
            interp_config = interpolate_configs(current_config, initial_config, s)
            
            send_position_command(data, interp_config, use_degrees=True)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
    
    print("\n" + "=" * 80)
    print("Multiple targets test complete!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--multi':
        run_multiple_targets_test()
    else:
        run_ik_test()