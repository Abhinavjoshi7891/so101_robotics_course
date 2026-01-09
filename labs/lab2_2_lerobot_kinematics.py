#!/usr/bin/env python3
"""
Lab 2.2: Using LeRobot's Built-in Kinematics
=============================================

This lab demonstrates how to use LeRobot's built-in kinematics module
which is powered by the Placo library.

LeRobot's RobotKinematics class provides:
    - Forward kinematics using the robot's URDF
    - Inverse kinematics with optimization
    - Jacobian computation
    - Integration with the lerobot ecosystem

Prerequisites:
    pip install lerobot[kinematics]
    
    Or install placo directly:
    pip install placo

Usage:
    python labs/lab2_2_lerobot_kinematics.py

Learning Objectives:
    1. Understand how LeRobot handles kinematics internally
    2. Compare hand-coded FK/IK with library implementation
    3. Learn when to use built-in vs custom kinematics

Author: SO-101 Robotics Course
"""

import sys
import os
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our hand-coded implementations for comparison
from so101_forward_kinematics import get_forward_kinematics as our_fk
from so101_inverse_kinematics import inverse_kinematics_numerical as our_ik

# Check for LeRobot kinematics
LEROBOT_KINEMATICS_AVAILABLE = False
try:
    # The actual LeRobot kinematics import path
    # Note: This requires lerobot to be installed with kinematics extras
    from lerobot.common.kinematics import RobotKinematics
    LEROBOT_KINEMATICS_AVAILABLE = True
except ImportError:
    try:
        # Try placo directly
        import placo
        PLACO_AVAILABLE = True
    except ImportError:
        PLACO_AVAILABLE = False


def demonstrate_placo_kinematics():
    """
    Demonstrate kinematics using placo library directly.
    
    This is what LeRobot uses under the hood.
    """
    try:
        import placo
    except ImportError:
        print("Placo not available. Install with: pip install placo")
        return None
    
    print("\n" + "-" * 70)
    print("Using Placo Library (LeRobot's kinematics backend)")
    print("-" * 70)
    
    # Path to URDF (you'll need to adjust this to your actual URDF location)
    urdf_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'ros2_ws', 'src', 
                     'lerobot_description', 'urdf', 'so101.urdf'),
        '/home/claude/so101_robotics_course/ros2_ws/src/lerobot_description/urdf/so101.urdf',
        # Add more potential paths
    ]
    
    urdf_path = None
    for path in urdf_paths:
        if os.path.exists(path):
            urdf_path = path
            break
    
    if urdf_path is None:
        print("URDF file not found. Demonstrating concept without actual robot model.")
        print("\nTo use placo with SO-101:")
        print("  1. Build the lerobot_ws ROS2 workspace")
        print("  2. The URDF will be at: ros2_ws/src/lerobot_description/urdf/so101.urdf")
        return None
    
    print(f"Loading URDF from: {urdf_path}")
    
    # Load robot model
    robot = placo.RobotWrapper(urdf_path)
    
    # Get joint names
    print(f"\nRobot joints: {robot.joint_names()}")
    print(f"Robot frames: {robot.frame_names()}")
    
    # Set a configuration
    q = np.zeros(robot.model.nq)  # Home position
    robot.update_kinematics(q)
    
    # Get end-effector pose
    ee_frame = "tool0"  # Adjust based on your URDF
    if ee_frame in robot.frame_names():
        T = robot.frame_pose(ee_frame)
        print(f"\nEnd-effector pose at home:")
        print(f"  Position: {T.translation}")
        print(f"  Rotation:\n{T.rotation}")
    
    return robot


def compare_implementations():
    """
    Compare our hand-coded FK with what a library would compute.
    """
    print("\n" + "-" * 70)
    print("Comparing Hand-coded FK with Expected Results")
    print("-" * 70)
    
    # Test configurations
    test_configs = [
        {
            'name': 'Home (all zeros)',
            'angles': {
                'shoulder_pan': 0.0,
                'shoulder_lift': 0.0,
                'elbow_flex': 0.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 0.0
            }
        },
        {
            'name': 'Extended forward',
            'angles': {
                'shoulder_pan': 0.0,
                'shoulder_lift': 45.0,
                'elbow_flex': -45.0,
                'wrist_flex': 0.0,
                'wrist_roll': 0.0,
                'gripper': 0.0
            }
        },
        {
            'name': 'Rotated with wrist',
            'angles': {
                'shoulder_pan': -45.0,
                'shoulder_lift': 30.0,
                'elbow_flex': -60.0,
                'wrist_flex': 90.0,
                'wrist_roll': 45.0,
                'gripper': 0.0
            }
        },
    ]
    
    print("\nOur hand-coded FK results:")
    print("-" * 50)
    
    for test in test_configs:
        position, rotation = our_fk(test['angles'])
        
        print(f"\n{test['name']}:")
        print(f"  Joint angles: {test['angles']}")
        print(f"  Position: x={position[0]:.4f}, y={position[1]:.4f}, z={position[2]:.4f}")
        print(f"  Distance from origin: {np.linalg.norm(position):.4f} m")


def demonstrate_ik_workflow():
    """
    Demonstrate the typical IK workflow in a robotics application.
    """
    print("\n" + "-" * 70)
    print("IK Workflow Demonstration")
    print("-" * 70)
    
    # Define a target position (in robot workspace)
    target_position = np.array([0.2, 0.1, 0.15])
    
    print(f"\nTarget position: {target_position}")
    
    # Use our numerical IK
    print("\n1. Computing IK solution...")
    joint_config, success, error = our_ik(target_position, verbose=False)
    
    print(f"   Success: {success}")
    print(f"   Final error: {error:.6f} m")
    print(f"   Joint configuration: {joint_config}")
    
    # Verify with FK
    print("\n2. Verifying with FK...")
    verify_position, _ = our_fk(joint_config)
    verification_error = np.linalg.norm(verify_position - target_position)
    
    print(f"   FK result: {verify_position}")
    print(f"   Verification error: {verification_error:.6f} m")
    
    # Try multiple targets
    print("\n3. Testing multiple target positions...")
    
    targets = [
        np.array([0.15, 0.0, 0.2]),
        np.array([0.2, 0.15, 0.1]),
        np.array([0.1, -0.15, 0.15]),
        np.array([0.25, 0.0, 0.05]),  # Near table
    ]
    
    for i, target in enumerate(targets):
        config, success, error = our_ik(target, verbose=False, max_iterations=100)
        status = "✓" if success else "✗"
        print(f"   Target {i+1}: {target} -> {status} (error: {error:.4f} m)")


def explain_when_to_use_what():
    """
    Explain when to use hand-coded vs library kinematics.
    """
    print("\n" + "=" * 70)
    print("When to Use What?")
    print("=" * 70)
    
    print("""
    Hand-coded Kinematics (like our implementation):
    ------------------------------------------------
    ✓ Educational purposes - understanding the math
    ✓ Simple robots with analytical IK solutions
    ✓ Performance-critical applications (can be optimized)
    ✓ When you need full control over the algorithm
    ✓ Debugging and verification
    
    Library Kinematics (LeRobot/Placo, MoveIt, Pinocchio):
    -------------------------------------------------------
    ✓ Complex robots (many DoF)
    ✓ Production systems needing robustness
    ✓ Integration with planning/control pipelines
    ✓ Collision checking and constraints
    ✓ Dynamics and simulation
    
    LeRobot's Approach:
    -------------------
    - Uses Placo library under the hood
    - Integrated with data collection pipelines
    - Used in lerobot-find-joint-limits script
    - Works with RL gym environments
    
    Example usage in LeRobot (gym_manipulator.py):
    ```python
    from lerobot.common.kinematics import RobotKinematics
    
    kinematics = RobotKinematics(urdf_path)
    
    # Forward kinematics
    ee_pose = kinematics.forward_kinematics(joint_positions_deg)
    
    # Inverse kinematics  
    joint_solution = kinematics.inverse_kinematics(
        current_joints, 
        target_pose
    )
    ```
    """)


def main():
    print("=" * 70)
    print("Lab 2.2: LeRobot's Built-in Kinematics")
    print("=" * 70)
    
    print(f"\nLeRobot Kinematics Available: {LEROBOT_KINEMATICS_AVAILABLE}")
    
    if LEROBOT_KINEMATICS_AVAILABLE:
        print("\nYou have LeRobot kinematics installed!")
        print("This uses the Placo library internally.")
    else:
        print("\nLeRobot kinematics not available.")
        print("Install with: pip install lerobot[kinematics]")
        print("\nProceeding with demonstration using our hand-coded implementation...")
    
    # Compare our implementation with expected results
    compare_implementations()
    
    # Demonstrate IK workflow
    demonstrate_ik_workflow()
    
    # Try placo if available
    if not LEROBOT_KINEMATICS_AVAILABLE:
        demonstrate_placo_kinematics()
    
    # Explain when to use what
    explain_when_to_use_what()
    
    print("\n" + "=" * 70)
    print("Lab 2.2 Complete!")
    print("=" * 70)
    
    print("\nKey Takeaways:")
    print("  1. Hand-coded FK/IK helps understand the fundamentals")
    print("  2. Libraries provide robust, optimized implementations")
    print("  3. LeRobot uses Placo for production kinematics")
    print("  4. Always verify IK solutions with FK!")


if __name__ == "__main__":
    main()
