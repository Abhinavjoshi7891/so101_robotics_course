#!/usr/bin/env python3
"""
Lab 2.2: Compare Geometric vs Numerical IK in MuJoCo
=====================================================

This lab demonstrates both IK approaches side-by-side:
1. Geometric IK - Fast analytical solution
2. Numerical IK - Optimization-based solution

Learning Objectives:
- Understand trade-offs between IK methods
- See speed differences in real-time
- Compare accuracy and reliability
- Learn when to use each approach

Usage:
    python labs/lab2_2_compare_ik_methods.py

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
from so101_inverse_kinematics import inverse_kinematics_numerical
from so101_inverse_kinematics_geometric import SO101GeometricIK
from so101_mujoco_utils import (
    set_initial_pose,
    send_position_command,
    move_to_pose,
    hold_position
)


def show_target_cube(viewer, position, color_rgba=[1, 0, 0, 0.5]):
    """Show a cube at target position."""
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.015, 0.015],
        pos=np.array(position, dtype=np.float64),
        mat=np.eye(3, dtype=np.float64).flatten(),
        rgba=np.array(color_rgba, dtype=np.float32)
    )
    viewer.user_scn.ngeom = 1
    viewer.sync()


def compare_ik_methods(target_position, mode='mujoco'):
    """
    Compare geometric vs numerical IK for a given target.
    
    Args:
        target_position: [x, y, z] target position
        mode: 'mujoco' or 'urdf_native'
    
    Returns:
        Dictionary with comparison results
    """
    print("\n" + "=" * 70)
    print(f"Comparing IK Methods for Target: {target_position}")
    print("=" * 70)
    
    results = {
        'target': target_position,
        'geometric': {},
        'numerical': {}
    }
    
    # Test Geometric IK
    print("\n1. GEOMETRIC IK (Analytical)")
    print("-" * 70)
    solver = SO101GeometricIK(mode=mode)
    
    start_time = time.time()
    config_geo, success_geo, msg_geo = solver.get_inverse_kinematics(
        target_position,
        validate_workspace=True
    )
    geo_time = time.time() - start_time
    
    print(f"Time: {geo_time*1000:.2f} ms")
    print(f"Success: {success_geo}")
    print(f"Message: {msg_geo}")
    
    if success_geo:
        # Verify with FK
        verify_pos_geo, _ = get_forward_kinematics(config_geo, mode=mode)
        error_geo = np.linalg.norm(verify_pos_geo - target_position)
        
        print(f"Joint angles:")
        for joint, angle in config_geo.items():
            if joint != 'gripper':
                print(f"  {joint:15s}: {angle:7.2f}°")
        
        print(f"\nFK Verification:")
        print(f"  Achieved: [{verify_pos_geo[0]:.4f}, {verify_pos_geo[1]:.4f}, {verify_pos_geo[2]:.4f}]")
        print(f"  Error: {error_geo*1000:.2f} mm")
        
        results['geometric'] = {
            'success': success_geo,
            'config': config_geo,
            'time_ms': geo_time * 1000,
            'error_mm': error_geo * 1000,
            'message': msg_geo
        }
    else:
        results['geometric'] = {
            'success': success_geo,
            'config': config_geo,
            'time_ms': geo_time * 1000,
            'error_mm': float('inf'),
            'message': msg_geo
        }
    
    # Test Numerical IK
    print("\n2. NUMERICAL IK (Optimization)")
    print("-" * 70)
    
    start_time = time.time()
    config_num, success_num, error_num = inverse_kinematics_numerical(
        target_position,
        initial_guess=None,
        verbose=False,
        mode=mode,
        max_iterations=200
    )
    num_time = time.time() - start_time
    
    print(f"Time: {num_time*1000:.2f} ms")
    print(f"Success: {success_num}")
    print(f"Error: {error_num*1000:.2f} mm")
    
    if success_num:
        print(f"Joint angles:")
        for joint, angle in config_num.items():
            if joint != 'gripper':
                print(f"  {joint:15s}: {angle:7.2f}°")
    
    results['numerical'] = {
        'success': success_num,
        'config': config_num,
        'time_ms': num_time * 1000,
        'error_mm': error_num * 1000,
        'message': 'Converged' if success_num else 'Failed'
    }
    
    # Comparison Summary
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    if success_geo and success_num:
        print(f"Speed:    Geometric {geo_time*1000:.2f}ms vs Numerical {num_time*1000:.2f}ms")
        print(f"          → Geometric is {num_time/geo_time:.1f}x FASTER")
        print(f"Accuracy: Geometric {error_geo*1000:.2f}mm vs Numerical {error_num*1000:.2f}mm")
        
        if error_geo < error_num:
            print(f"          → Geometric is MORE ACCURATE")
        else:
            print(f"          → Numerical is more accurate")
    elif success_geo and not success_num:
        print("✓ Geometric succeeded")
        print("✗ Numerical failed")
        print("  → Geometric is more RELIABLE for this target")
    elif not success_geo and success_num:
        print("✗ Geometric failed")
        print("✓ Numerical succeeded")
        print("  → Numerical found a solution when geometric couldn't")
    else:
        print("✗ Both methods failed")
        print("  → Target is outside workspace")
    
    return results


def run_comparison_demo():
    """Run interactive demo comparing both IK methods."""
    
    print("=" * 70)
    print("Lab 2.2: Geometric vs Numerical IK Comparison")
    print("=" * 70)
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Test targets (within workspace)
    test_targets = [
        {'name': 'Easy Reach', 'pos': np.array([0.25, 0.00, 0.20])},
        {'name': 'Side Reach', 'pos': np.array([0.20, 0.15, 0.15])},
        {'name': 'High Reach', 'pos': np.array([0.15, 0.00, 0.30])},
        {'name': 'Low Grasp', 'pos': np.array([0.20, 0.10, 0.014])},
        {'name': 'Far Reach', 'pos': np.array([0.30, 0.05, 0.20])},
    ]
    
    print("\nTest targets:")
    for i, target in enumerate(test_targets):
        print(f"  {i+1}. {target['name']:15s}: {target['pos']}")
    
    # Initialize robot
    initial_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }
    set_initial_pose(data, initial_config, use_degrees=True)
    
    # Compare all targets
    all_results = []
    for target_info in test_targets:
        results = compare_ik_methods(target_info['pos'], mode='mujoco')
        results['name'] = target_info['name']
        all_results.append(results)
    
    # Print overall comparison
    print("\n\n" + "=" * 70)
    print("OVERALL COMPARISON")
    print("=" * 70)
    
    geo_successes = sum(1 for r in all_results if r['geometric']['success'])
    num_successes = sum(1 for r in all_results if r['numerical']['success'])
    
    print(f"\nSuccess Rate:")
    print(f"  Geometric: {geo_successes}/{len(all_results)} ({geo_successes/len(all_results)*100:.0f}%)")
    print(f"  Numerical: {num_successes}/{len(all_results)} ({num_successes/len(all_results)*100:.0f}%)")
    
    if geo_successes > 0:
        avg_geo_time = np.mean([r['geometric']['time_ms'] for r in all_results if r['geometric']['success']])
        avg_geo_error = np.mean([r['geometric']['error_mm'] for r in all_results if r['geometric']['success']])
        print(f"\nGeometric IK (successful cases):")
        print(f"  Avg time:  {avg_geo_time:.2f} ms")
        print(f"  Avg error: {avg_geo_error:.2f} mm")
    
    if num_successes > 0:
        avg_num_time = np.mean([r['numerical']['time_ms'] for r in all_results if r['numerical']['success']])
        avg_num_error = np.mean([r['numerical']['error_mm'] for r in all_results if r['numerical']['success']])
        print(f"\nNumerical IK (successful cases):")
        print(f"  Avg time:  {avg_num_time:.2f} ms")
        print(f"  Avg error: {avg_num_error:.2f} mm")
    
    if geo_successes > 0 and num_successes > 0:
        speedup = avg_num_time / avg_geo_time
        print(f"\n→ Geometric is {speedup:.0f}x FASTER on average")
    
    # Visualization in MuJoCo
    print("\n" + "=" * 70)
    print("Starting MuJoCo Visualization")
    print("=" * 70)
    print("\nRobot will visit each target using GEOMETRIC IK (fastest)")
    print("Watch the smooth motion!")
    print("\nPress Ctrl+C or close window to exit.\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        target_idx = 0
        last_switch = time.time()
        
        # Initialize config and success before loop
        config = initial_config
        success = False
        
        while viewer.is_running():
            if time.time() - last_switch > 4.0:
                # Switch to next target
                target_idx = (target_idx + 1) % len(test_targets)
                target_info = test_targets[target_idx]
                target_pos = target_info['pos']
                
                print(f"\n→ Moving to: {target_info['name']}")
                
                # Show target cube
                show_target_cube(viewer, target_pos)
                
                # Use geometric IK (fast!)
                solver = SO101GeometricIK(mode='mujoco')
                config, success, msg = solver.get_inverse_kinematics(target_pos)
                
                if success:
                    # Move to target
                    move_to_pose(model, data, viewer, config, duration=2.0)
                    print(f"  ✓ Reached target")
                else:
                    print(f"  ✗ Failed: {msg}")
                
                last_switch = time.time()
            
            # Keep position
            send_position_command(data, config if success else initial_config, use_degrees=True)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
    
    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison_demo()