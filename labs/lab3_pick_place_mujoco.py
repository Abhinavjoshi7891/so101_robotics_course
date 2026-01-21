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
    python3 labs/lab3_pick_place_mujoco.py

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

# ==============================================================================
# FINE-TUNED WAYPOINTS (CENTERED & SAFE HEIGHTS)
# ==============================================================================

# 1. ADJUSTED PAN: -45.4 centers exactly on x=0.216, y=-0.219
# 2. ADJUSTED LIFT: 52.0 is safer (higher) for Place to avoid table collision

# ==============================================================================
# IMPROVED WAYPOINT DEFINITIONS - Within Safe Workspace
# ==============================================================================

# Home configuration
HOME_CONFIG = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 50.0
}

# Ready position (safe starting point)
READY_CONFIG = {
    'shoulder_pan': 0.0,
    'shoulder_lift': 30.0,
    'elbow_flex': -30.0,
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 50.0
}

# ======================================================================
# PICK SEQUENCE (Right side of robot)
# ======================================================================

# Approach pick (above object)
PICK_APPROACH_CONFIG = {
    'shoulder_pan': -30.0,   # Less extreme than -45°
    'shoulder_lift': 30.0,    # Moderate height
    'elbow_flex': -30.0,
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 50.0  # Open
}

# At pick location (lower down to object)
PICK_CONFIG = {
    'shoulder_pan': -30.0,
    'shoulder_lift': 45.0,    # Lower, but not extreme (was 60°)
    'elbow_flex': -45.0,      # More moderate
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 50.0  # Open
}

# Grasping
PICK_GRASP_CONFIG = {
    'shoulder_pan': -30.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# Lift after grasp
PICK_LIFT_CONFIG = {
    'shoulder_pan': -30.0,
    'shoulder_lift': 30.0,    # Back to moderate height
    'elbow_flex': -30.0,
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# ======================================================================
# TRANSIT (Move through center with clearance)
# ======================================================================

TRANSIT_CONFIG = {
    'shoulder_pan': 0.0,      # Center position
    'shoulder_lift': 20.0,    # High for clearance
    'elbow_flex': -20.0,
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# ======================================================================
# PLACE SEQUENCE (Left side of robot - MIRROR of pick)
# ======================================================================

# Approach place
PLACE_APPROACH_CONFIG = {
    'shoulder_pan': 30.0,     # Mirror: +30° instead of -30°
    'shoulder_lift': 30.0,    # Same moderate height
    'elbow_flex': -30.0,
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# At place location
PLACE_CONFIG = {
    'shoulder_pan': 30.0,
    'shoulder_lift': 45.0,    # Same as pick (was 60° - too extreme)
    'elbow_flex': -45.0,      # Same as pick
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 5.0  # Closed
}

# Release
PLACE_RELEASE_CONFIG = {
    'shoulder_pan': 30.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 50.0  # Open
}

# Retreat after placing
PLACE_RETREAT_CONFIG = {
    'shoulder_pan': 30.0,
    'shoulder_lift': 30.0,
    'elbow_flex': -30.0,
    'wrist_flex': 60.0,
    'wrist_roll': 0.0,
    'gripper': 50.0  # Open
}


def get_current_config(model, data):
    """Read current joint angles from MuJoCo and convert to degrees."""
    # Helper to safely get joint value
    def get_j(name):
        return data.joint(name).qpos[0]

    return {
        'shoulder_pan': np.rad2deg(get_j('shoulder_pan')),
        'shoulder_lift': np.rad2deg(get_j('shoulder_lift')),
        'elbow_flex': np.rad2deg(get_j('elbow_flex')),
        'wrist_flex': np.rad2deg(get_j('wrist_flex')),
        'wrist_roll': np.rad2deg(get_j('wrist_roll')),
        'gripper': np.rad2deg(get_j('gripper'))  # Note: Prismatic might need scaling if not 1:1 in XML
    }



def show_pick_place_markers(viewer, pick_config, place_config, halfwidth=0.015):
    """
    Show markers at pick and place locations using FK.
    
    Red cube: Pick location
    Green cube: Place location
    """
    # IMPORTANT: Use mode='mujoco' for correct coordinate system!
    pick_pos, pick_rot = get_forward_kinematics(pick_config, mode='mujoco')
    print(f"[DEBUG] Pick FK Position: {pick_pos}")
    print(f"[DEBUG] Pick FK Rotation:\n{pick_rot}")
    
    place_pos, place_rot = get_forward_kinematics(place_config, mode='mujoco')
    print(f"[DEBUG] Place FK Position: {place_pos}")
    print(f"[DEBUG] Place FK Rotation:\n{place_rot}")
    
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
    Smoothly interpolate with EXPLICIT LOGGING of Position Error.
    """
    start_time = time.time()
    steps = 0
    
    # Pre-calculate target position for logging comparison
    target_pos, _ = get_forward_kinematics(end_config)
    print(f"\n[MOVE START] Target: x={target_pos[0]:.4f}, y={target_pos[1]:.4f}, z={target_pos[2]:.4f}")
    
    while viewer.is_running() and (time.time() - start_time) < duration:
        t = (time.time() - start_time) / duration
        # Cubic ease in-out
        s = 3 * t**2 - 2 * t**3
        
        # Interpolate command
        interp_config = {}
        for joint in start_config:
            start_val = start_config[joint]
            end_val = end_config[joint]
            interp_config[joint] = start_val + s * (end_val - start_val)
        
        send_position_command(data, interp_config, use_degrees=True)
        mujoco.mj_step(model, data)
        viewer.sync()
        
        # --- LOGGING BLOCK ---
        steps += 1
        if steps % 20 == 0:  # Log every 20 simulation steps
            # 1. Get actual robot state
            current_conf = get_current_config(model, data)
            
            # 2. Compute where the physical robot ACTUALLY is
            curr_pos, _ = get_forward_kinematics(current_conf)
            
            # 3. Calculate distance to target
            dist_err = np.linalg.norm(curr_pos - target_pos)
            
            print(f"  [Step {steps:03d}] "
                  f"Curr: ({curr_pos[0]:.3f}, {curr_pos[1]:.3f}, {curr_pos[2]:.3f}) "
                  f"-> Dist to Target: {dist_err*1000:.1f} mm")
        # ---------------------
        
        time.sleep(0.005)
    
    # Final check
    send_position_command(data, end_config, use_degrees=True)
    for _ in range(50): # Give it a moment to settle
        mujoco.mj_step(model, data)
    
    final_conf = get_current_config(model, data)
    final_pos, _ = get_forward_kinematics(final_conf)
    final_err = np.linalg.norm(final_pos - target_pos)
    print(f"[MOVE END] Final Error: {final_err*1000:.2f} mm\n")


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
    waypoints_preview = [
        ("Ready", READY_CONFIG),
        ("Pick Approach", PICK_APPROACH_CONFIG),
        ("Pick", PICK_CONFIG),
        ("Lift", PICK_LIFT_CONFIG),
        ("Transit", TRANSIT_CONFIG),
        ("Place Approach", PLACE_APPROACH_CONFIG),
        ("Place", PLACE_CONFIG),
        ("Retreat", PLACE_RETREAT_CONFIG),
    ]
    
    for name, config in waypoints_preview:
        pos, _ = get_forward_kinematics(config, mode='mujoco')
        print(f"  {name:15s}: x={pos[0]:7.4f}, y={pos[1]:7.4f}, z={pos[2]:7.4f}")
    
    # Initialize at home
    print("\n" + "-" * 70)
    print("Initializing robot...")
    set_initial_pose(data, HOME_CONFIG, use_degrees=True)
    send_position_command(data, HOME_CONFIG, use_degrees=True)
    
    # Step to settle
    for _ in range(200):
        mujoco.mj_step(model, data)
    
    print("\nStarting pick and place sequence...")
    print("Watch the robot move through the waypoints.\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Show pick and place markers
        pick_pos, place_pos = show_pick_place_markers(
            viewer, PICK_CONFIG, PLACE_CONFIG
        )
        
        print(f"Pick location:  {pick_pos}")
        print(f"Place location: {place_pos}")
        
        # EXECUTION SEQUENCE
        time.sleep(1.0)
        
        print("\nPhase 1: Move to ready position")
        smooth_move(model, data, viewer, HOME_CONFIG, READY_CONFIG, duration=2.0)
        time.sleep(0.5)
        
        print("Phase 2: Approach pick location")
        smooth_move(model, data, viewer, READY_CONFIG, PICK_APPROACH_CONFIG, duration=2.0)
        time.sleep(0.5)
        
        print("Phase 3: Lower to pick")
        smooth_move(model, data, viewer, PICK_APPROACH_CONFIG, PICK_CONFIG, duration=1.5)
        time.sleep(0.5)
        
        print("Phase 4: Close gripper")
        smooth_move(model, data, viewer, PICK_CONFIG, PICK_GRASP_CONFIG, duration=0.5)
        time.sleep(0.5)
        
        print("Phase 5: Lift object")
        smooth_move(model, data, viewer, PICK_GRASP_CONFIG, PICK_LIFT_CONFIG, duration=1.5)
        time.sleep(0.5)
        
        print("Phase 6: Transit to place area")
        smooth_move(model, data, viewer, PICK_LIFT_CONFIG, TRANSIT_CONFIG, duration=2.0)
        time.sleep(0.5)
        
        print("Phase 7: Approach place location")
        smooth_move(model, data, viewer, TRANSIT_CONFIG, PLACE_APPROACH_CONFIG, duration=2.0)
        time.sleep(0.5)
        
        print("Phase 8: Lower to place")
        smooth_move(model, data, viewer, PLACE_APPROACH_CONFIG, PLACE_CONFIG, duration=1.5)
        time.sleep(0.5)
        
        print("Phase 9: Release gripper")
        smooth_move(model, data, viewer, PLACE_CONFIG, PLACE_RELEASE_CONFIG, duration=0.5)
        time.sleep(0.5)
        
        print("Phase 10: Retreat from place")
        smooth_move(model, data, viewer, PLACE_RELEASE_CONFIG, PLACE_RETREAT_CONFIG, duration=1.5)
        time.sleep(0.5)
        
        print("Phase 11: Return to ready")
        smooth_move(model, data, viewer, PLACE_RETREAT_CONFIG, READY_CONFIG, duration=2.0)
        time.sleep(0.5)
        
        print("Phase 12: Return to home")
        smooth_move(model, data, viewer, READY_CONFIG, HOME_CONFIG, duration=2.0)
        
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
