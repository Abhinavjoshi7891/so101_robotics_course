#!/usr/bin/env python3
"""
Lab: Random Gripper-Joint Command Test
=======================================

This lab tests gripper control combined with single-joint motion.
Random commands are generated as tuples: (gripper_value, joint_selector)

Command format:
    - gripper_value: 0-100 (0=fully closed, 100=fully open)
    - joint_selector: 'a' or 'b' ('a'=joint5/wrist_roll, 'b'=joint4/wrist_flex)

Example commands:
    (100, 'b') → Fully open gripper + move wrist_flex
    (0, 'a')   → Fully closed gripper + move wrist_roll
    (50, 'b')  → Half-open gripper + move wrist_flex

Learning Objectives:
    1. Understand gripper control (joint 6)
    2. Visualize single-joint motion
    3. See FK validation for end-effector position
    4. Test random command generation

Usage:
    python labs/lab_random_gripper_joint_test.py

Author: SO-101 Robotics Course
"""

import sys
import os
import time
import random
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
    degrees_to_mujoco
)


def generate_random_command():
    """
    Generate a random command tuple: (gripper_value, joint_selector)
    
    Returns:
        tuple: (int 0-100, str 'a' or 'b')
    """
    gripper_value = random.randint(0, 100)
    joint_selector = random.choice(['a', 'b'])
    return (gripper_value, joint_selector)


def map_gripper_value(gripper_value):
    """
    Map gripper value (0-100) to actual gripper percentage.
    
    Args:
        gripper_value: 0-100 (0=closed, 100=open)
    
    Returns:
        float: Gripper percentage (0-100)
    """
    # Simple linear mapping
    return float(gripper_value)


def create_config_from_command(command, joint_angle_range=(-90, 90)):
    """
    Create a joint configuration from a command tuple.
    
    Args:
        command: tuple (gripper_value, joint_selector)
        joint_angle_range: tuple (min_angle, max_angle) for the selected joint
    
    Returns:
        dict: Joint configuration
    """
    gripper_value, joint_selector = command
    
    # Base configuration (neutral pose)
    config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': map_gripper_value(gripper_value)
    }
    
    # Generate a random angle within the range for the selected joint
    min_angle, max_angle = joint_angle_range
    random_angle = random.uniform(min_angle, max_angle)
    
    if joint_selector == 'a':
        # 'a' maps to joint 5 (wrist_roll)
        config['wrist_roll'] = random_angle
        joint_name = 'wrist_roll (joint 5)'
    elif joint_selector == 'b':
        # 'b' maps to joint 4 (wrist_flex)
        config['wrist_flex'] = random_angle
        joint_name = 'wrist_flex (joint 4)'
    else:
        joint_name = 'none'
    
    return config, joint_name, random_angle


def get_mujoco_gripper_position(model, data):
    """Get the actual gripper position from MuJoCo simulation."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
    if site_id >= 0:
        return data.site_xpos[site_id].copy()
    return None


def show_sphere_at_fk(viewer, position, rgba=[1, 0, 0, 0.9], radius=0.025):
    """Show a red sphere at the FK-computed position."""
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


def interpolate_configs(config_start, config_target, alpha):
    """
    Linearly interpolate between two joint configurations.
    
    Args:
        config_start: Starting configuration dict
        config_target: Target configuration dict
        alpha: Interpolation factor (0.0 = start, 1.0 = target)
    
    Returns:
        Interpolated configuration dict
    """
    result = {}
    for key in config_start.keys():
        start_val = config_start[key]
        target_val = config_target[key]
        result[key] = start_val + alpha * (target_val - start_val)
    return result


def run_random_command_test():
    """Main test function with random command generation."""
    
    print("=" * 80)
    print("Random Gripper-Joint Command Test")
    print("=" * 80)
    print("\nCommand Format: (gripper_value, joint_selector)")
    print("  - gripper_value: 0-100 (0=closed, 100=open)")
    print("  - joint_selector: 'a'=joint5 (wrist_roll), 'b'=joint4 (wrist_flex)")
    print("\nThe script will generate random commands and visualize them.")
    print("=" * 80 + "\n")
    
    # Load MuJoCo model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    print(f"Loading model from: {model_path}\n")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    FK_MODE = 'mujoco'
    
    # Generate initial random commands
    num_commands = 5
    commands = [generate_random_command() for _ in range(num_commands)]
    
    print("Generated Commands:")
    print("-" * 80)
    for i, cmd in enumerate(commands, 1):
        gripper_val, joint_sel = cmd
        joint_name = 'wrist_roll (joint 5)' if joint_sel == 'a' else 'wrist_flex (joint 4)'
        gripper_state = 'closed' if gripper_val < 30 else ('half-open' if gripper_val < 70 else 'open')
        print(f"  {i}. {cmd} → Gripper: {gripper_val}% ({gripper_state}), Move: {joint_name}")
    print("-" * 80 + "\n")
    
    # Initial neutral configuration
    neutral_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    
    current_config = neutral_config.copy()
    set_initial_pose(data, current_config, use_degrees=True)
    
    # Step simulation to settle
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    print("Starting MuJoCo Visualization...")
    print("Watch the gripper open/close and joints move!\n")
    print("Commands will cycle every 5 seconds.")
    print("Press Ctrl+C or close window to exit.\n")
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        command_idx = 0
        state = 'MOVING'  # 'MOVING' or 'HOLDING'
        state_start_time = time.time()
        
        MOVE_DURATION = 3.0   # 3 seconds to move
        HOLD_DURATION = 2.0   # 2 seconds to hold
        
        # Generate config for first command
        target_config, joint_name, joint_angle = create_config_from_command(commands[command_idx])
        start_config = current_config.copy()
        
        gripper_val, joint_sel = commands[command_idx]
        print(f"→ Command {command_idx + 1}: {commands[command_idx]}")
        print(f"  Gripper: {gripper_val}% | Joint: {joint_name} → {joint_angle:.1f}°")
        
        while viewer.is_running():
            elapsed = time.time() - state_start_time
            
            # State machine
            if state == 'MOVING':
                # Interpolate to target
                alpha = min(1.0, elapsed / MOVE_DURATION)
                current_config = interpolate_configs(start_config, target_config, alpha)
                
                if elapsed >= MOVE_DURATION:
                    state = 'HOLDING'
                    state_start_time = time.time()
                    current_config = target_config.copy()
                    
            elif state == 'HOLDING':
                # Hold at target
                if elapsed >= HOLD_DURATION:
                    # Switch to next command
                    command_idx = (command_idx + 1) % len(commands)
                    
                    # Generate new target config
                    start_config = current_config.copy()
                    target_config, joint_name, joint_angle = create_config_from_command(commands[command_idx])
                    
                    gripper_val, joint_sel = commands[command_idx]
                    print(f"\n→ Command {command_idx + 1}: {commands[command_idx]}")
                    print(f"  Gripper: {gripper_val}% | Joint: {joint_name} → {joint_angle:.1f}°")
                    
                    state = 'MOVING'
                    state_start_time = time.time()
            
            # Send position command
            send_position_command(data, current_config, use_degrees=True)
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Compute FK for current configuration
            fk_pos, fk_rot = get_forward_kinematics(current_config, mode=FK_MODE)
            
            # Get MuJoCo ground truth
            mujoco_pos = get_mujoco_gripper_position(model, data)
            
            # Show FK sphere
            #show_sphere_at_fk(viewer, fk_pos)
            
            # Display current state
            if mujoco_pos is not None:
                error = np.linalg.norm(fk_pos - mujoco_pos) * 1000.0
                
                state_str = f"[{state:7s}]" if state == 'MOVING' else f"[{state:7s}]"
                progress = f"{int(alpha * 100):3d}%" if state == 'MOVING' else "HOLD"
                
                gripper_str = f"G:{current_config['gripper']:5.1f}%"
                
                print(f"\r{state_str} {progress} | {gripper_str} | "
                      f"FK: [{fk_pos[0]:.3f}, {fk_pos[1]:.3f}, {fk_pos[2]:.3f}] | "
                      f"Err: {error:5.2f}mm", end="")
            
            # Sync viewer
            viewer.sync()
            
            # Small delay
            time.sleep(0.01)
    
    print("\n\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)


def run_interactive_command_test():
    """
    Interactive mode - manually enter commands.
    """
    print("\n" + "=" * 80)
    print("Interactive Command Test")
    print("=" * 80)
    print("\nManually enter commands to test specific scenarios.")
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    FK_MODE = 'mujoco'
    
    # Get user input
    print("\nEnter command:")
    try:
        gripper_value = int(input("  Gripper value (0-100): "))
        if gripper_value < 0 or gripper_value > 100:
            print("  Invalid gripper value, using 50")
            gripper_value = 50
    except ValueError:
        print("  Invalid input, using 50")
        gripper_value = 50
    
    joint_selector = input("  Joint selector (a=wrist_roll, b=wrist_flex): ").strip().lower()
    if joint_selector not in ['a', 'b']:
        print("  Invalid selector, using 'a'")
        joint_selector = 'a'
    
    command = (gripper_value, joint_selector)
    
    # Create configuration
    config, joint_name, joint_angle = create_config_from_command(command)
    
    print(f"\nCommand: {command}")
    print(f"  Gripper: {gripper_value}%")
    print(f"  Joint: {joint_name} → {joint_angle:.1f}°")
    
    # Set initial pose
    set_initial_pose(data, config, use_degrees=True)
    
    for _ in range(100):
        mujoco.mj_step(model, data)
    
    # Compute FK
    fk_pos, fk_rot = get_forward_kinematics(config, mode=FK_MODE)
    mujoco_pos = get_mujoco_gripper_position(model, data)
    
    if mujoco_pos is not None:
        error = np.linalg.norm(fk_pos - mujoco_pos) * 1000.0
        print(f"\nFK Position:     [{fk_pos[0]:.4f}, {fk_pos[1]:.4f}, {fk_pos[2]:.4f}]")
        print(f"MuJoCo Position: [{mujoco_pos[0]:.4f}, {mujoco_pos[1]:.4f}, {mujoco_pos[2]:.4f}]")
        print(f"Error: {error:.2f} mm")
    
    print("\nStarting visualization...")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        show_sphere_at_fk(viewer, fk_pos)
        
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < 30.0:
            send_position_command(data, config, use_degrees=True)
            mujoco.mj_step(model, data)
            show_sphere_at_fk(viewer, fk_pos)
            viewer.sync()
            time.sleep(0.01)


def print_usage():
    """Print usage information."""
    print("\nUsage:")
    print("  python labs/lab_random_gripper_joint_test.py           # Random commands (default)")
    print("  python labs/lab_random_gripper_joint_test.py --random  # Random commands")
    print("  python labs/lab_random_gripper_joint_test.py --manual  # Manual input")
    print("\nCommand Format:")
    print("  (gripper_value, joint_selector)")
    print("    - gripper_value: 0-100 (0=closed, 100=open)")
    print("    - joint_selector: 'a' or 'b'")
    print("      'a' = joint 5 (wrist_roll)")
    print("      'b' = joint 4 (wrist_flex)")
    print("\nExamples:")
    print("  (100, 'b') → Fully open gripper + move wrist_flex")
    print("  (0, 'a')   → Fully closed gripper + move wrist_roll")
    print("  (50, 'b')  → Half-open gripper + move wrist_flex")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '--manual':
            run_interactive_command_test()
        elif sys.argv[1] == '--help':
            print_usage()
        elif sys.argv[1] == '--random':
            run_random_command_test()
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print_usage()
    else:
        # Default: random commands
        run_random_command_test()