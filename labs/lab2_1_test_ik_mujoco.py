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
    python labs/lab2_1_test_ik_mujoco.py --interactive

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


def show_target_cube(viewer, position, halfwidth=0.015, rgba=[1, 0, 0, 0.5]):
    """Show a semi-transparent cube at the target position."""
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


def run_ik_test():
    """Main IK test function."""
    
    print("=" * 70)
    print("Lab 2.1: Inverse Kinematics Test in MuJoCo")
    print("=" * 70)
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # ==========================================================================
    # Test 1: Forward-then-Inverse
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 1: Forward Kinematics → Inverse Kinematics")
    print("-" * 70)
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
    
    print(f"\nOriginal joint angles: {original_config}")
    
    # Compute FK
    target_position, target_rotation = get_forward_kinematics(original_config)
    print(f"FK result: x={target_position[0]:.4f}, y={target_position[1]:.4f}, z={target_position[2]:.4f}")
    
    # Compute IK (numerical method)
    print("\nRunning numerical IK...")
    recovered_config, success, error = inverse_kinematics_numerical(
        target_position,
        initial_guess=None,  # Start from home
        verbose=True,
        max_iterations=50
    )
    
    print(f"\nIK Result:")
    print(f"  Success: {success}")
    print(f"  Final error: {error:.6f} m")
    print(f"  Recovered angles: {recovered_config}")
    
    # Verify with FK
    verify_pos, _ = get_forward_kinematics(recovered_config)
    print(f"  Verification: x={verify_pos[0]:.4f}, y={verify_pos[1]:.4f}, z={verify_pos[2]:.4f}")
    
    # ==========================================================================
    # Test 2: Reach to arbitrary target
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 2: Reach to Arbitrary Target")
    print("-" * 70)
    
    # Target position (should be within workspace)
    target_position_2 = np.array([0.15, 0.15, 0.15])
    print(f"\nTarget position: {target_position_2}")
    
    # Try geometric IK
    print("\n1. Geometric IK:")
    geo_config, geo_success = inverse_kinematics_geometric(target_position_2)
    print(f"   Success: {geo_success}")
    if geo_success:
        geo_verify, _ = get_forward_kinematics(geo_config)
        geo_error = np.linalg.norm(geo_verify - target_position_2)
        print(f"   Error: {geo_error:.6f} m")
        print(f"   Joint angles: {geo_config}")
    
    # Try numerical IK
    print("\n2. Numerical IK:")
    num_config, num_success, num_error = inverse_kinematics_numerical(
        target_position_2,
        verbose=False
    )
    print(f"   Success: {num_success}")
    print(f"   Error: {num_error:.6f} m")
    print(f"   Joint angles: {num_config}")
    
    # ==========================================================================
    # Visualization
    # ==========================================================================
    print("\n" + "=" * 70)
    print("Starting MuJoCo Visualization")
    print("=" * 70)
    print("\nThe robot will move to reach the target (red cube).")
    print("Press Ctrl+C or close the window to exit.\n")
    
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
        # Show target cube
        show_target_cube(viewer, target_position_2)
        
        print("Phase 1: Moving to target position...")
        
        # Move to IK solution
        start_time = time.time()
        move_duration = 3.0
        
        while viewer.is_running() and (time.time() - start_time) < move_duration:
            t = (time.time() - start_time) / move_duration
            s = 3 * t**2 - 2 * t**3  # Smooth interpolation
            
            # Interpolate joint angles
            interp_config = {}
            for joint in initial_config:
                interp_config[joint] = initial_config[joint] + s * (ik_result.get(joint, 0) - initial_config[joint])
            
            send_position_command(data, interp_config, use_degrees=True)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
        
        print("Phase 2: Holding at target...")
        
        # Hold at target
        hold_start = time.time()
        while viewer.is_running() and (time.time() - hold_start) < 10.0:
            send_position_command(data, ik_result, use_degrees=True)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
    
    print("\n" + "=" * 70)
    print("IK Test Complete!")
    print("=" * 70)


def run_multiple_targets_test():
    """Test IK with multiple target positions."""
    
    print("\n" + "=" * 70)
    print("Multiple Targets IK Test")
    print("=" * 70)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Define multiple target positions
    targets = [
        np.array([0.2, 0.0, 0.15]),   # Front
        np.array([0.15, 0.15, 0.15]), # Front-left
        np.array([0.15, -0.15, 0.15]),# Front-right
        np.array([0.1, 0.2, 0.2]),    # Left-high
        np.array([0.1, -0.2, 0.2]),   # Right-high
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
        config, success, error = inverse_kinematics_numerical(target, verbose=False)
        ik_solutions.append(config)
        print(f"  Target {i+1}: pos={target}, success={success}, error={error:.6f}")
    
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
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Show all target cubes
        for i, (target, color) in enumerate(zip(targets, target_colors)):
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.015, 0.015, 0.015],
                pos=np.array(target, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).flatten(),
                rgba=np.array(color, dtype=np.float32)
            )
        viewer.user_scn.ngeom = len(targets)
        viewer.sync()
        
        current_config = initial_config.copy()
        
        # Visit each target
        for target_idx, target_config in enumerate(ik_solutions):
            if not viewer.is_running():
                break
                
            print(f"\nMoving to target {target_idx + 1}...")
            
            # Smooth move to target
            start_time = time.time()
            move_duration = 2.0
            
            while viewer.is_running() and (time.time() - start_time) < move_duration:
                t = (time.time() - start_time) / move_duration
                s = 3 * t**2 - 2 * t**3
                
                interp_config = {}
                for joint in current_config:
                    start_val = current_config.get(joint, 0)
                    end_val = target_config.get(joint, 0)
                    interp_config[joint] = start_val + s * (end_val - start_val)
                
                send_position_command(data, interp_config, use_degrees=True)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)
            
            # Brief pause at target
            hold_start = time.time()
            while viewer.is_running() and (time.time() - hold_start) < 1.0:
                send_position_command(data, target_config, use_degrees=True)
                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(0.01)
            
            current_config = target_config.copy()
        
        # Return to home
        print("\nReturning to home...")
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < 2.0:
            t = (time.time() - start_time) / 2.0
            s = 3 * t**2 - 2 * t**3
            
            interp_config = {}
            for joint in current_config:
                interp_config[joint] = current_config.get(joint, 0) * (1 - s)
            
            send_position_command(data, interp_config, use_degrees=True)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
    
    print("\nMultiple targets test complete!")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--multi':
        run_multiple_targets_test()
    else:
        run_ik_test()
