#!/usr/bin/env python3
"""
Lab 2.2: Test GEOMETRIC IK in MuJoCo
====================================

This lab tests the ANALYTICAL (Geometric) Inverse Kinematics solver.
It uses the `so101_inverse_kinematics_geometric.py` module.

Features:
    - Uses exact geometric dimensions (no iterative guessing).
    - Extremely fast computation (< 0.1ms).
    - Visualizes the "Radial Reach" strategy.

Usage:
    python labs/lab2_2_test_ik_geometric_mujoco.py

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
# IMPORT THE NEW GEOMETRIC SOLVER CLASS
from so101_inverse_kinematics_geometric import SO101GeometricIK

from so101_mujoco_utils import (
    set_initial_pose,
    send_position_command,
    degrees_to_mujoco
)

def get_mujoco_gripper_position(model, data):
    """Get the actual gripper position from MuJoCo simulation."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
    if site_id >= 0:
        return data.site_xpos[site_id].copy()
    return None

def show_target_marker(viewer, position, color=[1, 0, 0, 0.5]):
    """Show a semi-transparent cube at the target position."""
    if viewer.user_scn.ngeom >= 10: return # Limit markers
    
    idx = viewer.user_scn.ngeom
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.015, 0.015],
        pos=np.array(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.array(color, dtype=np.float32)
    )
    viewer.user_scn.ngeom += 1

def interpolate_configs(config_start, config_target, alpha):
    """Linearly interpolate between two joint configurations."""
    result = {}
    for key in config_start.keys():
        start_val = config_start[key]
        target_val = config_target.get(key, 0)
        result[key] = start_val + alpha * (target_val - start_val)
    return result

def main():
    print("=" * 80)
    print("Lab 2.2: Geometric IK Test (MuJoCo)")
    print("=" * 80)
    
    # 1. Load Model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 2. Initialize Solver with 'mujoco' mode
    # This ensures it uses the correct coordinate system
    print("\nInitializing Geometric Solver (mode='mujoco')...")
    ik_solver = SO101GeometricIK(mode='mujoco')
    
    # 3. Define Test Targets
    # Note: MuJoCo Z is "Up", Y is "Left/Right", X is "Back/Front" usually
    # But SO-101 base might be rotated. The solver handles this.
    targets = [
        {'name': 'Table Center', 'pos': [0.20, 0.0, 0.02]},   # Perfect for picking
        {'name': 'Table Left',   'pos': [0.20, 0.15, 0.02]},
        {'name': 'Table Right',  'pos': [0.20, -0.15, 0.02]},
        {'name': 'Close Pick',   'pos': [0.15, 0.0, 0.02]},
        {'name': 'Far Pick',     'pos': [0.30, 0.0, 0.02]},   # Max vertical reach
    ]
    
    # Initial Configuration
    current_config = {
        'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 'elbow_flex': 0.0,
        'wrist_flex': 0.0, 'wrist_roll': 0.0, 'gripper': 50.0
    }
    
    set_initial_pose(data, current_config, use_degrees=True)
    
    print("\nStarting Visualization...")
    print("  RED Cube   = Target Position")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        
        # Loop through targets
        for i, target in enumerate(targets):
            if not viewer.is_running(): break
            
            target_pos = np.array(target['pos'])
            name = target['name']
            
            print(f"\n[{i+1}/{len(targets)}] Testing: {name}")
            print(f"  Target: {target_pos}")
            
            # --- SOLVE IK ---
            start_ik = time.time()
            ik_config, success, message = ik_solver.get_inverse_kinematics(
                target_pos,
                validate_workspace=True
            )
            ik_time_ms = (time.time() - start_ik) * 1000
            
            if success:
                print(f"  ✓ IK Solved in {ik_time_ms:.3f} ms")
                target_config = ik_config
                color = [0, 1, 0, 0.5] # Green for success
            else:
                print(f"  ✗ IK Failed: {message}")
                # Stay at current config if failed
                target_config = current_config
                color = [1, 0, 0, 0.5] # Red for fail
            
            # --- MOVE ROBOT ---
            # Interpolate for smooth motion
            move_start = time.time()
            duration = 2.0
            
            while viewer.is_running() and (time.time() - move_start) < duration:
                elapsed = time.time() - move_start
                t = min(1.0, elapsed / duration)
                s = 3*t**2 - 2*t**3 # Ease-in-out
                
                # Interpolate
                interp_config = interpolate_configs(current_config, target_config, s)
                send_position_command(data, interp_config, use_degrees=True)
                mujoco.mj_step(model, data)
                
                # Visualize
                viewer.user_scn.ngeom = 0 # Reset markers
                show_target_marker(viewer, target_pos, color)
                
                # Check error
                curr_pos = get_mujoco_gripper_position(model, data)
                if curr_pos is not None:
                    err = np.linalg.norm(curr_pos - target_pos) * 1000
                    print(f"\r  Moving... Error: {err:6.1f} mm", end="")
                
                viewer.sync()
                time.sleep(0.01)
            
            # --- HOLD ---
            print(f"\r  Final Error: {err:6.1f} mm            ")
            current_config = target_config # Update start for next move
            time.sleep(0.5)
            
    print("\nTest Complete.")

if __name__ == "__main__":
    main()