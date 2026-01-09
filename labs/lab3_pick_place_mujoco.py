#!/usr/bin/env python3
"""
Lab 3: Pick and Place in MuJoCo
================================

This lab implements a complete pick-and-place task using:
    1. Forward kinematics to compute gripper positions
    2. Predefined waypoints for motion planning
    3. Smooth trajectory interpolation

The task sequence:
    1. Start at home position with gripper open
    2. Move above pick location
    3. Move down to pick location
    4. Close gripper
    5. Lift object
    6. Move above place location
    7. Move down to place location
    8. Open gripper
    9. Lift and return to home

Learning Objectives:
    1. Understand task-level motion planning
    2. Implement waypoint-based trajectories
    3. Coordinate gripper actions with arm motion

Usage:
    python labs/lab3_pick_place_mujoco.py

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
from so101_mujoco_utils import (
    set_initial_pose,
    send_position_command,
    move_to_pose,
    hold_position,
    degrees_to_mujoco
)


# ==============================================================================
# WAYPOINT DEFINITIONS
# ==============================================================================

# Starting configuration (above pick location, gripper open)
STARTING_CONFIG = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50.0  # Open
}

# Pick position (at object, gripper open)
PICK_CONFIG = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 60.0,  # Lower
    'elbow_flex': -60.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50.0  # Open
}

# Pick position with closed gripper
PICK_CONFIG_CLOSED = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 60.0,
    'elbow_flex': -60.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# Lifted position (gripper closed)
LIFT_CONFIG = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 30.0,  # Higher
    'elbow_flex': -30.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# Transit position (high, for obstacle clearance)
TRANSIT_CONFIG = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 30.0,
    'elbow_flex': -30.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# Above place position
PLACE_ABOVE_CONFIG = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 30.0,
    'elbow_flex': -30.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# Place position (gripper closed)
PLACE_CONFIG_CLOSED = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 60.0,
    'elbow_flex': -60.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# Place position (gripper open - release object)
PLACE_CONFIG_OPEN = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 60.0,
    'elbow_flex': -60.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50.0  # Open
}

# Final position (above place, gripper open)
FINAL_CONFIG = {
    'shoulder_pan': 45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50.0  # Open
}

# Home configuration
HOME_CONFIG = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 50.0
}


def show_pick_place_markers(viewer, pick_config, place_config, halfwidth=0.015):
    """
    Show markers at pick and place locations using FK.
    
    Red cube: Pick location
    Green cube: Place location
    """
    # Compute positions from FK
    pick_pos, pick_rot = get_forward_kinematics(pick_config)
    place_pos, place_rot = get_forward_kinematics(place_config)
    
    # Pick marker (red, semi-transparent)
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[halfwidth, halfwidth, halfwidth],
        pos=np.array(pick_pos, dtype=np.float64),
        mat=np.array(pick_rot, dtype=np.float64).flatten(),
        rgba=np.array([1, 0, 0, 0.3], dtype=np.float32)
    )
    
    # Place marker (green, semi-transparent)
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[1],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[halfwidth, halfwidth, halfwidth],
        pos=np.array(place_pos, dtype=np.float64),
        mat=np.array(place_rot, dtype=np.float64).flatten(),
        rgba=np.array([0, 1, 0, 0.3], dtype=np.float32)
    )
    
    viewer.user_scn.ngeom = 2
    viewer.sync()
    
    return pick_pos, place_pos


def smooth_move(model, data, viewer, start_config, end_config, duration):
    """
    Smoothly interpolate between two configurations.
    
    Uses cubic easing for smooth acceleration/deceleration.
    """
    start_time = time.time()
    
    while viewer.is_running() and (time.time() - start_time) < duration:
        t = (time.time() - start_time) / duration
        # Cubic ease in-out
        s = 3 * t**2 - 2 * t**3
        
        # Interpolate each joint
        interp_config = {}
        for joint in start_config:
            start_val = start_config[joint]
            end_val = end_config[joint]
            interp_config[joint] = start_val + s * (end_val - start_val)
        
        send_position_command(data, interp_config, use_degrees=True)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.005)
    
    # Ensure we reach exactly the end configuration
    send_position_command(data, end_config, use_degrees=True)
    for _ in range(10):
        mujoco.mj_step(model, data)
    viewer.sync()


def run_pick_and_place():
    """Execute the pick and place sequence."""
    
    print("=" * 70)
    print("Lab 3: Pick and Place in MuJoCo")
    print("=" * 70)
    
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Print waypoint positions using FK
    print("\nWaypoint Positions (computed via FK):")
    waypoints = [
        ("Starting", STARTING_CONFIG),
        ("Pick", PICK_CONFIG),
        ("Lift", LIFT_CONFIG),
        ("Transit", TRANSIT_CONFIG),
        ("Place Above", PLACE_ABOVE_CONFIG),
        ("Place", PLACE_CONFIG_CLOSED),
        ("Final", FINAL_CONFIG),
    ]
    
    for name, config in waypoints:
        pos, _ = get_forward_kinematics(config)
        print(f"  {name:12s}: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
    
    # Initialize at starting position
    print("\n" + "-" * 70)
    print("Initializing robot...")
    set_initial_pose(data, STARTING_CONFIG, use_degrees=True)
    send_position_command(data, STARTING_CONFIG, use_degrees=True)
    
    # Step to settle
    for _ in range(200):
        mujoco.mj_step(model, data)
    
    print("\nStarting pick and place sequence...")
    print("Watch the robot move through the waypoints.\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Show pick and place markers
        pick_pos, place_pos = show_pick_place_markers(
            viewer, PICK_CONFIG, PLACE_CONFIG_CLOSED
        )
        
        print(f"Pick location:  {pick_pos}")
        print(f"Place location: {place_pos}")
        
        # Brief pause to show initial state
        print("\nPhase 0: Starting position")
        time.sleep(1.0)
        
        # ======================================================================
        # Phase 1: Move down to pick
        # ======================================================================
        print("Phase 1: Moving to pick position...")
        smooth_move(model, data, viewer, STARTING_CONFIG, PICK_CONFIG, duration=2.0)
        time.sleep(0.5)
        
        # ======================================================================
        # Phase 2: Close gripper
        # ======================================================================
        print("Phase 2: Closing gripper...")
        smooth_move(model, data, viewer, PICK_CONFIG, PICK_CONFIG_CLOSED, duration=0.5)
        time.sleep(0.5)
        
        # ======================================================================
        # Phase 3: Lift object
        # ======================================================================
        print("Phase 3: Lifting object...")
        smooth_move(model, data, viewer, PICK_CONFIG_CLOSED, LIFT_CONFIG, duration=1.5)
        time.sleep(0.3)
        
        # ======================================================================
        # Phase 4: Transit to place location
        # ======================================================================
        print("Phase 4: Transit to place location...")
        smooth_move(model, data, viewer, LIFT_CONFIG, TRANSIT_CONFIG, duration=1.5)
        smooth_move(model, data, viewer, TRANSIT_CONFIG, PLACE_ABOVE_CONFIG, duration=1.5)
        time.sleep(0.3)
        
        # ======================================================================
        # Phase 5: Move down to place
        # ======================================================================
        print("Phase 5: Moving down to place...")
        smooth_move(model, data, viewer, PLACE_ABOVE_CONFIG, PLACE_CONFIG_CLOSED, duration=1.5)
        time.sleep(0.5)
        
        # ======================================================================
        # Phase 6: Open gripper (release object)
        # ======================================================================
        print("Phase 6: Releasing object...")
        smooth_move(model, data, viewer, PLACE_CONFIG_CLOSED, PLACE_CONFIG_OPEN, duration=0.5)
        time.sleep(0.5)
        
        # ======================================================================
        # Phase 7: Retreat
        # ======================================================================
        print("Phase 7: Retreating...")
        smooth_move(model, data, viewer, PLACE_CONFIG_OPEN, FINAL_CONFIG, duration=1.5)
        time.sleep(0.5)
        
        # ======================================================================
        # Phase 8: Return to home
        # ======================================================================
        print("Phase 8: Returning to home...")
        smooth_move(model, data, viewer, FINAL_CONFIG, HOME_CONFIG, duration=2.0)
        
        print("\n" + "=" * 70)
        print("Pick and Place Complete!")
        print("=" * 70)
        
        # Hold at home for viewing
        print("\nHolding at home position. Close window to exit.")
        while viewer.is_running():
            send_position_command(data, HOME_CONFIG, use_degrees=True)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)


def run_continuous_demo():
    """Run pick and place in a continuous loop."""
    
    print("=" * 70)
    print("Continuous Pick and Place Demo")
    print("=" * 70)
    
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'scene.xml')
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # Define the waypoint sequence
    sequence = [
        (STARTING_CONFIG, 2.0, "Move to start"),
        (PICK_CONFIG, 1.5, "Approach pick"),
        (PICK_CONFIG_CLOSED, 0.5, "Close gripper"),
        (LIFT_CONFIG, 1.5, "Lift"),
        (TRANSIT_CONFIG, 1.5, "Transit"),
        (PLACE_ABOVE_CONFIG, 1.5, "Approach place"),
        (PLACE_CONFIG_CLOSED, 1.5, "Lower"),
        (PLACE_CONFIG_OPEN, 0.5, "Open gripper"),
        (FINAL_CONFIG, 1.5, "Retreat"),
        (HOME_CONFIG, 2.0, "Return home"),
    ]
    
    set_initial_pose(data, HOME_CONFIG, use_degrees=True)
    
    for _ in range(200):
        mujoco.mj_step(model, data)
    
    cycle_count = 0
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        show_pick_place_markers(viewer, PICK_CONFIG, PLACE_CONFIG_CLOSED)
        
        while viewer.is_running():
            cycle_count += 1
            print(f"\n--- Cycle {cycle_count} ---")
            
            current_config = HOME_CONFIG
            
            for target_config, duration, description in sequence:
                if not viewer.is_running():
                    break
                    
                print(f"  {description}...")
                smooth_move(model, data, viewer, current_config, target_config, duration)
                current_config = target_config
            
            # Brief pause between cycles
            time.sleep(1.0)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--continuous':
        run_continuous_demo()
    else:
        run_pick_and_place()
