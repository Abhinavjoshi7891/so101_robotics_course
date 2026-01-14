#!/usr/bin/env python3
"""
Lab 1.2: Test Forward Kinematics in MuJoCo
==========================================

This lab verifies your forward kinematics implementation by visualizing
the computed tool frame position and orientation in MuJoCo.

If your FK is correct, a red sphere will appear at the gripper location.

Learning Objectives:
    1. Validate FK implementation against simulation ground truth
    2. Understand the relationship between joint angles and end-effector pose
    3. Visualize coordinate frames in 3D

Usage:
    python labs/lab1_2_test_fk_mujoco.py

Author: SO-101 Robotics Course
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check for MuJoCo
try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("Error: MuJoCo not installed. Install with: pip install mujoco")
    sys.exit(1)

from so101_forward_kinematics import get_forward_kinematics, get_intermediate_transforms
from so101_mujoco_utils import (
    set_initial_pose, 
    send_position_command, 
    hold_position,
    show_cylinder,
    show_frame,
    degrees_to_mujoco
)


def get_mujoco_gripper_position(model, data):
    """
    Get the actual gripper position from MuJoCo simulation.
    This is the ground truth we're comparing against.
    """
    # Get the gripper site position
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
    if site_id >= 0:
        return data.site_xpos[site_id].copy()
    else:
        print("Warning: Could not find 'gripperframe' site in MuJoCo model")
        return None


def show_sphere_at_fk(viewer, position, rgba=[1, 0, 0, 0.9], radius=0.02):
    """
    Add a sphere at the FK-computed position.
    
    If FK is correct, this sphere should overlap with the gripper.
    """
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[radius, 0, 0],
        pos=np.array(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.array(rgba, dtype=np.float32)
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()


def run_fk_test():
    """Main FK test function."""
    
    print("=" * 70)
    print("Lab 1.2: Forward Kinematics Test in MuJoCo")
    print("=" * 70)
    
    # Load MuJoCo model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Make sure you have the scene.xml file in the model/ directory")
        return
    
    print(f"\nLoading model from: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # ==========================================================================
    # IMPORTANT: MuJoCo uses different coordinate conventions than URDF
    # Use mode='mujoco' for MuJoCo simulations
    # ==========================================================================
    FK_MODE = 'mujoco'
    
    print(f"\nUsing FK mode: {FK_MODE}")
    print("(MuJoCo and URDF use different coordinate frame conventions)")
    
    # ==========================================================================
    # Test Configuration 1: Home pose
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 1: Home Configuration")
    print("-" * 70)
    
    test_config_1 = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    
    print(f"\nJoint angles (degrees): {test_config_1}")
    
    # Set pose in MuJoCo
    set_initial_pose(data, test_config_1, use_degrees=True)
    send_position_command(data, test_config_1, use_degrees=True)
    
    # Step simulation to settle
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    # Get MuJoCo ground truth
    mujoco_pos_1 = get_mujoco_gripper_position(model, data)
    
    # Compute FK
    position_1, rotation_1 = get_forward_kinematics(test_config_1, mode=FK_MODE)
    
    if mujoco_pos_1 is not None:
        error_1 = np.linalg.norm(position_1 - mujoco_pos_1) * 1000.0  # mm
        print(f"\nFK Position:     [{position_1[0]:.4f}, {position_1[1]:.4f}, {position_1[2]:.4f}]")
        print(f"MuJoCo Position: [{mujoco_pos_1[0]:.4f}, {mujoco_pos_1[1]:.4f}, {mujoco_pos_1[2]:.4f}]")
        print(f"Error: {error_1:.2f} mm")
    else:
        print(f"\nFK Position: [{position_1[0]:.4f}, {position_1[1]:.4f}, {position_1[2]:.4f}]")
    
    # ==========================================================================
    # Test Configuration 2: Complex pose
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 2: Complex Configuration")
    print("-" * 70)
    
    test_config_2 = {
        'shoulder_pan': -45.0,
        'shoulder_lift': 45.0,
        'elbow_flex': -45.0,
        'wrist_flex': 90.0,
        'wrist_roll': 0.0,
        'gripper': 10.0
    }
    
    print(f"\nJoint angles (degrees): {test_config_2}")
    
    # Set pose in MuJoCo
    set_initial_pose(data, test_config_2, use_degrees=True)
    send_position_command(data, test_config_2, use_degrees=True)
    
    # Step simulation to settle
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    # Get MuJoCo ground truth
    mujoco_pos_2 = get_mujoco_gripper_position(model, data)
    
    # Compute FK
    position_2, rotation_2 = get_forward_kinematics(test_config_2, mode=FK_MODE)
    
    if mujoco_pos_2 is not None:
        error_2 = np.linalg.norm(position_2 - mujoco_pos_2) * 1000.0  # mm
        print(f"\nFK Position:     [{position_2[0]:.4f}, {position_2[1]:.4f}, {position_2[2]:.4f}]")
        print(f"MuJoCo Position: [{mujoco_pos_2[0]:.4f}, {mujoco_pos_2[1]:.4f}, {mujoco_pos_2[2]:.4f}]")
        print(f"Error: {error_2:.2f} mm")
    else:
        print(f"\nFK Position: [{position_2[0]:.4f}, {position_2[1]:.4f}, {position_2[2]:.4f}]")
    
    # ==========================================================================
    # Test Configuration 3: Extended reach
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 3: Extended Reach")
    print("-" * 70)
    
    test_config_3 = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 30.0,
        'elbow_flex': -60.0,
        'wrist_flex': 30.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    
    print(f"\nJoint angles (degrees): {test_config_3}")
    
    # Set pose in MuJoCo
    set_initial_pose(data, test_config_3, use_degrees=True)
    send_position_command(data, test_config_3, use_degrees=True)
    
    # Step simulation to settle
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    # Get MuJoCo ground truth
    mujoco_pos_3 = get_mujoco_gripper_position(model, data)
    
    # Compute FK
    position_3, rotation_3 = get_forward_kinematics(test_config_3, mode=FK_MODE)
    
    if mujoco_pos_3 is not None:
        error_3 = np.linalg.norm(position_3 - mujoco_pos_3) * 1000.0  # mm
        print(f"\nFK Position:     [{position_3[0]:.4f}, {position_3[1]:.4f}, {position_3[2]:.4f}]")
        print(f"MuJoCo Position: [{mujoco_pos_3[0]:.4f}, {mujoco_pos_3[1]:.4f}, {mujoco_pos_3[2]:.4f}]")
        print(f"Error: {error_3:.2f} mm")
    else:
        print(f"\nFK Position: [{position_3[0]:.4f}, {position_3[1]:.4f}, {position_3[2]:.4f}]")
    
    # ==========================================================================
    # Visualization in MuJoCo
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Starting MuJoCo Visualization")
    print("=" * 70)
    print("\nA RED SPHERE marks the FK-calculated position.")
    print("It should overlap with the gripper if FK is correct.")
    print("\nThe robot will cycle through all three test configurations.")
    print("\nPress Ctrl+C or close the window to exit.\n")
    
    # Prepare test configurations
    test_configs = [
        {'name': 'Home', 'config': test_config_1, 'fk_pos': position_1},
        {'name': 'Complex', 'config': test_config_2, 'fk_pos': position_2},
        {'name': 'Extended', 'config': test_config_3, 'fk_pos': position_3},
    ]
    
    # Start with first config
    current_config_idx = 0
    set_initial_pose(data, test_configs[0]['config'], use_degrees=True)
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print(f"→ Configuration: {test_configs[0]['name']}")
        
        start_time = time.time()
        last_switch = time.time()
        SWITCH_INTERVAL = 5.0  # seconds
        
        while viewer.is_running():
            # Switch configurations periodically
            if time.time() - last_switch > SWITCH_INTERVAL:
                current_config_idx = (current_config_idx + 1) % len(test_configs)
                current_test = test_configs[current_config_idx]
                print(f"\n→ Configuration: {current_test['name']}")
                
                # Update robot position
                send_position_command(data, current_test['config'], use_degrees=True)
                
                # Step to settle
                for _ in range(50):
                    mujoco.mj_step(model, data)
                
                last_switch = time.time()
            
            # Get current test config
            current_test = test_configs[current_config_idx]
            
            # Keep position command active
            send_position_command(data, current_test['config'], use_degrees=True)
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Show FK sphere at computed position
            show_sphere_at_fk(viewer, current_test['fk_pos'])
            
            # Get current MuJoCo position and print error
            mujoco_pos = get_mujoco_gripper_position(model, data)
            if mujoco_pos is not None:
                error = np.linalg.norm(current_test['fk_pos'] - mujoco_pos) * 1000.0
                print(f"\r[{current_test['name']:8s}] FK: [{current_test['fk_pos'][0]:.4f}, {current_test['fk_pos'][1]:.4f}, {current_test['fk_pos'][2]:.4f}] | "
                      f"MuJoCo: [{mujoco_pos[0]:.4f}, {mujoco_pos[1]:.4f}, {mujoco_pos[2]:.4f}] | "
                      f"Err: {error:6.2f} mm", end="")
            
            # Sync viewer
            viewer.sync()
            
            # Small delay
            time.sleep(0.01)
            
            # Exit after 30 seconds
            if time.time() - start_time > 30.0:
                break
    
    print("\n\n" + "=" * 70)
    print("FK Test Complete!")
    print("=" * 70)
    
    # Final summary
    print("\nSummary:")
    if mujoco_pos_1 is not None and mujoco_pos_2 is not None and mujoco_pos_3 is not None:
        print(f"  Home config error:     {error_1:.2f} mm")
        print(f"  Complex config error:  {error_2:.2f} mm")
        print(f"  Extended config error: {error_3:.2f} mm")
        
        max_error = max(error_1, error_2, error_3)
        if max_error < 1.0:
            print("\n✓ EXCELLENT: All errors < 1mm")
        elif max_error < 10.0:
            print("\n✓ GOOD: All errors < 10mm")
        elif max_error < 50.0:
            print("\n⚠ ACCEPTABLE: Some errors 10-50mm (check FK implementation)")
        else:
            print("\n✗ LARGE ERRORS: Errors > 50mm (FK may need debugging)")
            print("\nTroubleshooting:")
            print("  1. Make sure you're using mode='mujoco' for MuJoCo tests")
            print("  2. Check that joint angle conventions match")
            print("  3. Verify kinematic parameters match MuJoCo XML")


def run_interactive_fk_test():
    """
    Interactive FK test where you can modify joint angles and see results.
    """
    print("\n" + "=" * 70)
    print("Interactive FK Test")
    print("=" * 70)
    print("\nThis test lets you input joint angles and visualize the result.")
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    FK_MODE = 'mujoco'
    
    # Default configuration
    joint_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    
    print("\nDefault joint configuration:")
    for joint, angle in joint_config.items():
        print(f"  {joint}: {angle}°")
    
    # Ask for input
    print("\nEnter new joint angles (press Enter to keep default):")
    
    for joint in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']:
        try:
            user_input = input(f"  {joint} [{joint_config[joint]}]: ")
            if user_input.strip():
                joint_config[joint] = float(user_input)
        except ValueError:
            print(f"  Invalid input, keeping {joint_config[joint]}")
    
    # Compute FK
    position, rotation = get_forward_kinematics(joint_config, mode=FK_MODE)
    
    # Set pose in MuJoCo and get ground truth
    set_initial_pose(data, joint_config, use_degrees=True)
    send_position_command(data, joint_config, use_degrees=True)
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    mujoco_pos = get_mujoco_gripper_position(model, data)
    
    print(f"\nFK Result:")
    print(f"  FK Position:     [{position[0]:.4f}, {position[1]:.4f}, {position[2]:.4f}]")
    if mujoco_pos is not None:
        error = np.linalg.norm(position - mujoco_pos) * 1000.0
        print(f"  MuJoCo Position: [{mujoco_pos[0]:.4f}, {mujoco_pos[1]:.4f}, {mujoco_pos[2]:.4f}]")
        print(f"  Error: {error:.2f} mm")
    print(f"  Distance from origin: {np.linalg.norm(position):.4f} m")
    
    # Visualize
    print("\nStarting visualization...")
    print("The RED SPHERE shows the FK-calculated position.")
    print("It should overlap with the gripper.\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        show_sphere_at_fk(viewer, position)
        
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < 30.0:
            send_position_command(data, joint_config, use_degrees=True)
            mujoco.mj_step(model, data)
            
            # Update sphere position
            show_sphere_at_fk(viewer, position)
            
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        run_interactive_fk_test()
    else:
        run_fk_test()
        