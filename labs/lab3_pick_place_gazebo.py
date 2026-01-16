#!/usr/bin/env python3
"""
Lab 3.3: Pick and Place with Gazebo + MoveIt
=============================================

This lab implements pick-and-place in Gazebo using ROS2 Control and MoveIt.

Prerequisites:
    Terminal 1: ros2 launch lerobot_description so101_gazebo.launch.py
    Terminal 2: ros2 launch lerobot_controller so101_controller.launch.py  
    Terminal 3: ros2 launch lerobot_moveit so101_moveit.launch.py
    Terminal 4: python labs/lab3_pick_place_gazebo.py

The workflow:
    1. Connect to MoveIt's move_group action server
    2. Define pick and place poses
    3. Plan and execute motions using MoveIt
    4. Control gripper through joint trajectory

Learning Objectives:
    1. Understand MoveIt motion planning pipeline
    2. Use ROS2 action clients for robot control
    3. Coordinate arm motion with gripper actions

Author: SO-101 Robotics Course
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Check for ROS2
ROS2_AVAILABLE = False
MOVEIT_AVAILABLE = False

try:
    import rclpy
    from rclpy.node import Node
    from rclpy.action import ActionClient
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from control_msgs.action import FollowJointTrajectory
    from builtin_interfaces.msg import Duration
    from geometry_msgs.msg import Pose, Point, Quaternion
    from sensor_msgs.msg import JointState
    ROS2_AVAILABLE = True
    
    # Try MoveIt imports
    try:
        from moveit_msgs.action import MoveGroup
        from moveit_msgs.msg import (
            MotionPlanRequest, 
            Constraints,
            JointConstraint,
            PositionConstraint,
            OrientationConstraint,
            BoundingVolume
        )
        from shape_msgs.msg import SolidPrimitive
        MOVEIT_AVAILABLE = True
    except ImportError:
        pass

except ImportError:
    pass


# ==============================================================================
# JOINT NAME MAPPING
# ==============================================================================
# The URDF uses numeric joint names, but we use descriptive names in code
JOINT_NAME_MAPPING = {
    'shoulder_pan': '1',
    'shoulder_lift': '2',
    'elbow_flex': '3',
    'wrist_flex': '4',
    'wrist_roll': '5',
    'gripper': '6'
}

# For MoveIt and controllers, use numeric names directly
ARM_JOINT_NAMES = ['1', '2', '3', '4', '5']
GRIPPER_JOINT_NAME = '6'


# ==============================================================================
# FALLBACK: Simple trajectory execution without MoveIt
# ==============================================================================

def run_simple_trajectory():
    """
    Run pick and place using direct joint trajectory control.
    This works without MoveIt, using only ROS2 Control.
    """
    if not ROS2_AVAILABLE:
        print_ros2_instructions()
        return
    
    print("\n" + "-" * 70)
    print("Running Simple Trajectory Control (without MoveIt)")
    print("-" * 70)
    
    class TrajectoryNode(Node):
        def __init__(self):
            super().__init__('trajectory_controller')
            
            # Publisher for joint trajectory
            self.trajectory_pub = self.create_publisher(
                JointTrajectory,
                '/arm_controller/joint_trajectory',
                10
            )
            
            self.gripper_pub = self.create_publisher(
                JointTrajectory,
                '/gripper_controller/joint_trajectory',
                10
            )
            
            # Use numeric joint names matching URDF
            self.joint_names = ARM_JOINT_NAMES
            
            self.get_logger().info('Trajectory controller initialized')
        
        def send_arm_trajectory(self, positions_deg, duration_sec):
            """Send arm to specified joint positions."""
            msg = JointTrajectory()
            msg.joint_names = self.joint_names
            
            point = JointTrajectoryPoint()
            for angle in positions_deg:
                point.positions.append(np.deg2rad(angle))
            
            point.time_from_start = Duration(
                sec=int(duration_sec),
                nanosec=int((duration_sec % 1) * 1e9)
            )
            msg.points.append(point)
            
            self.trajectory_pub.publish(msg)
            self.get_logger().info(f'Sent arm trajectory')
        
        def send_gripper_command(self, position, duration_sec=0.5):
            """Control gripper (0=closed, 1=open)."""
            msg = JointTrajectory()
            msg.joint_names = [GRIPPER_JOINT_NAME]
            
            point = JointTrajectoryPoint()
            point.positions.append(position * 0.04)  # Scale to joint range
            point.time_from_start = Duration(
                sec=int(duration_sec),
                nanosec=int((duration_sec % 1) * 1e9)
            )
            msg.points.append(point)
            
            self.gripper_pub.publish(msg)
            self.get_logger().info(f'Sent gripper command: {position}')
    
    # Define waypoints (in degrees)
    waypoints = [
        # [joint1, joint2, joint3, joint4, joint5]
        {'name': 'Home', 'joints': [0, 0, 0, 0, 0], 'gripper': 1.0},
        {'name': 'Above Pick', 'joints': [-45, 45, -45, 90, 0], 'gripper': 1.0},
        {'name': 'Pick', 'joints': [-45, 60, -60, 90, 0], 'gripper': 1.0},
        {'name': 'Grasp', 'joints': [-45, 60, -60, 90, 0], 'gripper': 0.1},
        {'name': 'Lift', 'joints': [-45, 30, -30, 90, 0], 'gripper': 0.1},
        {'name': 'Transit', 'joints': [0, 30, -30, 90, 0], 'gripper': 0.1},
        {'name': 'Above Place', 'joints': [45, 30, -30, 90, 0], 'gripper': 0.1},
        {'name': 'Place', 'joints': [45, 60, -60, 90, 0], 'gripper': 0.1},
        {'name': 'Release', 'joints': [45, 60, -60, 90, 0], 'gripper': 1.0},
        {'name': 'Retreat', 'joints': [45, 45, -45, 90, 0], 'gripper': 1.0},
        {'name': 'Home', 'joints': [0, 0, 0, 0, 0], 'gripper': 1.0},
    ]
    
    rclpy.init()
    node = TrajectoryNode()
    
    try:
        print("\nExecuting pick and place sequence...")
        print("Make sure Gazebo and controllers are running!")
        print("\nRequired terminals:")
        print("  1. ros2 launch lerobot_description so101_gazebo.launch.py")
        print("  2. ros2 launch lerobot_controller so101_controller.launch.py")
        
        input("\nPress Enter to start...")
        
        for i, wp in enumerate(waypoints):
            print(f"\n[{i+1}/{len(waypoints)}] Moving to: {wp['name']}")
            
            # Send arm trajectory
            node.send_arm_trajectory(wp['joints'], duration_sec=2.0)
            
            # Wait for motion
            time.sleep(2.5)
            
            # Send gripper command
            node.send_gripper_command(wp['gripper'])
            time.sleep(0.7)
            
            rclpy.spin_once(node, timeout_sec=0.1)
        
        print("\n" + "=" * 70)
        print("Pick and Place Complete!")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ==============================================================================
# ==============================================================================
# MOVEIT-BASED PICK AND PLACE
# ==============================================================================

def run_moveit_pick_place():
    """
    Run pick and place using MoveIt motion planning.
    """
    if not MOVEIT_AVAILABLE:
        print("\nMoveIt is not available.")
        print("Falling back to simple trajectory control...")
        run_simple_trajectory()
        return
    
    print("\n" + "-" * 70)
    print("Running MoveIt-based Pick and Place")
    print("-" * 70)
    
    class MoveItPickPlace(Node):
        def __init__(self):
            super().__init__('moveit_pick_place')
            
            # Subscribe to joint states
            self.current_state = None
            self.joint_state_sub = self.create_subscription(
                JointState,
                '/joint_states',
                self.joint_state_callback,
                10
            )
            
            # MoveIt action client
            self._action_client = ActionClient(
                self,
                MoveGroup,
                '/move_action'
            )
            
            self.get_logger().info('Waiting for MoveIt action server...')
            self._action_client.wait_for_server()
            self.get_logger().info('Connected to MoveIt!')
            
            # Wait for first joint state
            self.get_logger().info('Waiting for joint states...')
            while self.current_state is None:
                rclpy.spin_once(self, timeout_sec=0.1)
            self.get_logger().info('Joint states received!')
        
        def joint_state_callback(self, msg):
            """Store current joint state"""
            self.current_state = dict(zip(msg.name, msg.position))
        
        def wait_for_motion_complete(self, timeout=5.0):
            """Wait for robot to stop moving"""
            self.get_logger().info('Waiting for motion to complete...')
            time.sleep(timeout)
            
            # Spin a few times to get fresh joint states
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.01)
        
        def move_to_joint_positions(self, joint_positions):
            """Plan and execute motion to joint positions."""
            # Wait a bit to ensure state is synchronized
            self.wait_for_motion_complete(timeout=1.0)
            
            goal_msg = MoveGroup.Goal()
            
            # Set planning group
            goal_msg.request.group_name = 'arm'
            
            # Set joint constraints using NUMERIC joint names
            constraints = Constraints()
            
            for name, position in zip(ARM_JOINT_NAMES, joint_positions):
                jc = JointConstraint()
                jc.joint_name = name
                jc.position = np.deg2rad(position)
                jc.tolerance_above = 0.01
                jc.tolerance_below = 0.01
                jc.weight = 1.0
                constraints.joint_constraints.append(jc)
            
            goal_msg.request.goal_constraints.append(constraints)
            goal_msg.request.num_planning_attempts = 5
            goal_msg.request.allowed_planning_time = 5.0
            
            # IMPORTANT: Set workspace bounds and velocity scaling
            goal_msg.request.workspace_parameters.header.frame_id = "base"
            goal_msg.request.max_velocity_scaling_factor = 0.1  # Slower = more stable
            goal_msg.request.max_acceleration_scaling_factor = 0.1
            
            # Send goal
            future = self._action_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error('Goal rejected')
                return False
            
            # Wait for result
            result_future = goal_handle.get_result_async()
            rclpy.spin_until_future_complete(self, result_future)
            
            result = result_future.result().result
            if result.error_code.val == 1:  # SUCCESS
                self.get_logger().info('Motion succeeded!')
                return True
            else:
                self.get_logger().error(f'Motion failed: {result.error_code.val}')
                return False
    
    rclpy.init()
    node = MoveItPickPlace()
    
    # Define joint waypoints
    waypoints = [
        [0, 0, 0, 0, 0],           # Home
        [-45, 45, -45, 90, 0],     # Above pick
        [-45, 60, -60, 90, 0],     # Pick
        [-45, 30, -30, 90, 0],     # Lift
        [45, 30, -30, 90, 0],      # Above place
        [45, 60, -60, 90, 0],      # Place
        [45, 45, -45, 90, 0],      # Retreat
        [0, 0, 0, 0, 0],           # Home
    ]
    
    try:
        print("\nExecuting MoveIt pick and place sequence...")
        
        for i, wp in enumerate(waypoints):
            print(f"\n[{i+1}/{len(waypoints)}] Planning motion...")
            success = node.move_to_joint_positions(wp)
            
            if not success:
                print("Motion planning failed, stopping.")
                break
            
            time.sleep(2.0)  # Increased wait time
        
        print("\nPick and place complete!")
        
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def print_ros2_instructions():
    """Print instructions for running ROS2-based labs."""
    print("\n" + "=" * 70)
    print("ROS2/Gazebo Setup Instructions")
    print("=" * 70)
    
    print("""
    This lab requires ROS2 and Gazebo. Follow these steps:
    
    1. Install ROS2 Humble:
       Follow: https://docs.ros.org/en/humble/Installation.html
    
    2. Build the lerobot_ws workspace:
       cd so101_robotics_course/ros2_ws
       colcon build
       source install/setup.bash
    
    3. Launch the simulation (in separate terminals):
    
       Terminal 1 - Gazebo:
       $ ros2 launch lerobot_description so101_gazebo.launch.py
       
       Terminal 2 - Controllers:
       $ ros2 launch lerobot_controller so101_controller.launch.py
       
       Terminal 3 - MoveIt (optional):
       $ ros2 launch lerobot_moveit so101_moveit.launch.py
       
       Terminal 4 - This script:
       $ python labs/lab3_pick_place_gazebo.py
    
    4. Alternative: Use the MuJoCo version instead:
       $ python labs/lab3_pick_place_mujoco.py
    """)


def main():
    print("=" * 70)
    print("Lab 3.3: Pick and Place with Gazebo + MoveIt")
    print("=" * 70)
    
    print(f"\nROS2 Available: {ROS2_AVAILABLE}")
    print(f"MoveIt Available: {MOVEIT_AVAILABLE}")
    
    if not ROS2_AVAILABLE:
        print_ros2_instructions()
        
        print("\n" + "-" * 70)
        print("Alternative: Running MuJoCo simulation instead")
        print("-" * 70)
        
        response = input("\nWould you like to run the MuJoCo version? (y/n): ")
        if response.lower() == 'y':
            # Import and run MuJoCo version
            from lab3_pick_place_mujoco import run_pick_and_place
            run_pick_and_place()
        return
    
    print("\nSelect execution mode:")
    print("  1. Simple trajectory control (ROS2 Control only)")
    print("  2. MoveIt motion planning (requires MoveIt)")
    
    choice = input("\nEnter choice (1/2): ").strip()
    
    if choice == '2' and MOVEIT_AVAILABLE:
        run_moveit_pick_place()
    else:
        run_simple_trajectory()


if __name__ == "__main__":
    main()