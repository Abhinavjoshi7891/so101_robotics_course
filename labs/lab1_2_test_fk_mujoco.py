#!/usr/bin/env python3
"""
Lab 1.2: Test Forward Kinematics in MuJoCo
==========================================

This lab verifies your forward kinematics implementation by visualizing
the computed tool frame position and orientation in MuJoCo.

If your FK is correct, a red cylinder will appear at the gripper location,
aligned with the tool frame's Z-axis.

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


def show_cylinder_at_fk(viewer, position, rotation, radius=0.0245, halfheight=0.05, rgba=[1, 0, 0, 0.7]):
    """
    Add a cylinder aligned with the FK-computed tool frame Z-axis.
    
    If FK is correct, this cylinder should appear in the gripper.
    """
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[radius, halfheight, 0],
        pos=np.array(position, dtype=np.float64),
        mat=np.array(rotation, dtype=np.float64).flatten(),
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
    # Test Configuration 1: Simple pose
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 1: Simple Configuration")
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
    
    # Compute FK
    position_1, rotation_1 = get_forward_kinematics(test_config_1)
    print(f"FK Tool position: x={position_1[0]:.4f}, y={position_1[1]:.4f}, z={position_1[2]:.4f}")
    
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
    
    # Compute FK
    position_2, rotation_2 = get_forward_kinematics(test_config_2)
    print(f"FK Tool position: x={position_2[0]:.4f}, y={position_2[1]:.4f}, z={position_2[2]:.4f}")
    
    # ==========================================================================
    # Visualization in MuJoCo
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Starting MuJoCo Visualization")
    print("=" * 70)
    print("\nA red cylinder should appear at the gripper if FK is correct.")
    print("The cylinder is aligned with the tool frame Z-axis.")
    print("\nPress Ctrl+C or close the window to exit.\n")
    
    # Set initial pose
    set_initial_pose(data, test_config_2, use_degrees=True)
    send_position_command(data, test_config_2, use_degrees=True)
    
    # Step simulation to settle
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Add the FK-computed cylinder
        show_cylinder_at_fk(viewer, position_2, rotation_2)
        
        print("Visualization active. Observe the red cylinder position.")
        print("If FK is correct, the cylinder should be at the gripper.")
        
        # Also show coordinate frames at each joint
        transforms = get_intermediate_transforms(test_config_2)
        print("\nIntermediate frame positions:")
        for name, T in transforms.items():
            pos = T[0:3, 3]
            print(f"  {name:8s}: [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
        
        # Hold position and visualize
        start_time = time.time()
        duration = 30.0  # seconds
        
        while viewer.is_running() and (time.time() - start_time) < duration:
            # Keep the position command active
            send_position_command(data, test_config_2, use_degrees=True)
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Sync viewer
            viewer.sync()
            
            # Small delay
            time.sleep(0.01)
    
    print("\n" + "=" * 70)
    print("FK Test Complete!")
    print("=" * 70)


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
    position, rotation = get_forward_kinematics(joint_config)
    
    print(f"\nFK Result:")
    print(f"  Position: x={position[0]:.4f}, y={position[1]:.4f}, z={position[2]:.4f}")
    print(f"  Distance from origin: {np.linalg.norm(position):.4f} m")
    
    # Visualize
    print("\nStarting visualization...")
    
    set_initial_pose(data, joint_config, use_degrees=True)
    send_position_command(data, joint_config, use_degrees=True)
    
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        show_cylinder_at_fk(viewer, position, rotation)
        
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < 30.0:
            send_position_command(data, joint_config, use_degrees=True)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    # Check for command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        run_interactive_fk_test()
    else:
        run_fk_test()
