#!/usr/bin/env python3
"""
SO-101 ROS2 Utilities
=====================

Helper functions for integrating with ROS2, RViz, and Gazebo.

This module provides:
    - Joint state publishing
    - TF frame visualization
    - ROS2 node management
    - Interface with MoveIt

Prerequisites:
    - ROS2 Humble (Ubuntu 22.04) or ROS2 Jazzy (Ubuntu 24.04)
    - lerobot_ws workspace built and sourced

Author: SO-101 Robotics Course
License: MIT
"""

import os
import numpy as np
from typing import Dict, List, Optional, Tuple

# Detect ROS2 distribution
ROS_DISTRO = os.environ.get('ROS_DISTRO', None)

# ROS2 imports (will fail gracefully if not available)
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from geometry_msgs.msg import TransformStamped, Pose, Point, Quaternion
    from std_msgs.msg import Header
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from builtin_interfaces.msg import Duration
    import tf2_ros
    from tf_transformations import quaternion_from_matrix
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    if ROS_DISTRO:
        print(f"Warning: ROS2 {ROS_DISTRO} detected but imports failed.")
    else:
        print("Warning: ROS2 not available. ROS2 features disabled.")
        print("To enable: source /opt/ros/humble/setup.bash  (Ubuntu 22.04)")
        print("       or: source /opt/ros/jazzy/setup.bash   (Ubuntu 24.04)")


# ==============================================================================
# JOINT CONFIGURATION
# ==============================================================================

JOINT_NAMES = [
    'shoulder_pan',
    'shoulder_lift',
    'elbow_flex',
    'wrist_flex',
    'wrist_roll',
    'gripper'
]

# ROS2 topic names
JOINT_STATE_TOPIC = '/joint_states'
JOINT_TRAJECTORY_TOPIC = '/arm_controller/joint_trajectory'


# ==============================================================================
# ROS2 NODE FOR VISUALIZATION
# ==============================================================================

if ROS2_AVAILABLE:
    
    class SO101VisualizationNode(Node):
        """
        ROS2 node for visualizing SO-101 arm in RViz.
        
        Publishes joint states and TF transforms for visualization.
        """
        
        def __init__(self, node_name: str = 'so101_visualization'):
            super().__init__(node_name)
            
            # Publishers
            self.joint_state_pub = self.create_publisher(
                JointState, 
                JOINT_STATE_TOPIC, 
                10
            )
            
            # TF broadcaster
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
            
            # Timer for periodic publishing
            self.timer = self.create_timer(0.02, self.timer_callback)  # 50 Hz
            
            # Current joint positions (degrees)
            self.joint_positions = {joint: 0.0 for joint in JOINT_NAMES}
            
            self.get_logger().info('SO-101 Visualization Node initialized')
        
        def set_joint_positions(self, positions: Dict[str, float]):
            """
            Set the current joint positions.
            
            Args:
                positions: Dictionary of joint angles in degrees
            """
            for joint, angle in positions.items():
                if joint in self.joint_positions:
                    self.joint_positions[joint] = angle
        
        def timer_callback(self):
            """Periodic callback to publish joint states."""
            self.publish_joint_states()
        
        def publish_joint_states(self):
            """Publish current joint states."""
            msg = JointState()
            msg.header.stamp = self.get_clock().now().to_msg()
            
            for joint in JOINT_NAMES:
                msg.name.append(joint)
                # Convert degrees to radians for ROS
                if joint == 'gripper':
                    msg.position.append(self.joint_positions[joint] / 100.0 * 0.04)
                else:
                    msg.position.append(np.deg2rad(self.joint_positions[joint]))
            
            self.joint_state_pub.publish(msg)
        
        def publish_frame(self, frame_id: str, parent_frame: str, 
                         position: np.ndarray, rotation: np.ndarray):
            """
            Publish a TF transform.
            
            Args:
                frame_id: Name of the frame to publish
                parent_frame: Parent frame name
                position: [x, y, z] position
                rotation: 3x3 rotation matrix
            """
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = parent_frame
            t.child_frame_id = frame_id
            
            t.transform.translation.x = float(position[0])
            t.transform.translation.y = float(position[1])
            t.transform.translation.z = float(position[2])
            
            # Convert rotation matrix to quaternion
            rot_4x4 = np.eye(4)
            rot_4x4[0:3, 0:3] = rotation
            quat = quaternion_from_matrix(rot_4x4)
            
            t.transform.rotation.x = quat[0]
            t.transform.rotation.y = quat[1]
            t.transform.rotation.z = quat[2]
            t.transform.rotation.w = quat[3]
            
            self.tf_broadcaster.sendTransform(t)
    
    
    class SO101TrajectoryNode(Node):
        """
        ROS2 node for sending trajectories to the SO-101 arm.
        
        Interfaces with ROS2 Control and MoveIt.
        """
        
        def __init__(self, node_name: str = 'so101_trajectory'):
            super().__init__(node_name)
            
            # Trajectory publisher
            self.trajectory_pub = self.create_publisher(
                JointTrajectory,
                JOINT_TRAJECTORY_TOPIC,
                10
            )
            
            self.get_logger().info('SO-101 Trajectory Node initialized')
        
        def send_joint_trajectory(
            self,
            waypoints: List[Dict[str, float]],
            durations: List[float]
        ):
            """
            Send a joint trajectory to the arm controller.
            
            Args:
                waypoints: List of joint angle dictionaries (degrees)
                durations: Time to reach each waypoint (seconds)
            """
            msg = JointTrajectory()
            msg.header.stamp = self.get_clock().now().to_msg()
            
            # Set joint names (excluding gripper for arm trajectory)
            msg.joint_names = [j for j in JOINT_NAMES if j != 'gripper']
            
            cumulative_time = 0.0
            
            for i, (waypoint, duration) in enumerate(zip(waypoints, durations)):
                point = JointTrajectoryPoint()
                
                for joint in msg.joint_names:
                    angle_rad = np.deg2rad(waypoint.get(joint, 0.0))
                    point.positions.append(angle_rad)
                
                cumulative_time += duration
                point.time_from_start = Duration(
                    sec=int(cumulative_time),
                    nanosec=int((cumulative_time % 1) * 1e9)
                )
                
                msg.points.append(point)
            
            self.trajectory_pub.publish(msg)
            self.get_logger().info(f'Published trajectory with {len(waypoints)} waypoints')


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def init_ros2(args=None):
    """Initialize ROS2."""
    if not ROS2_AVAILABLE:
        raise RuntimeError("ROS2 is not available")
    rclpy.init(args=args)


def shutdown_ros2():
    """Shutdown ROS2."""
    if ROS2_AVAILABLE:
        rclpy.shutdown()


def rotation_to_quaternion(rotation: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Convert a 3x3 rotation matrix to quaternion (x, y, z, w).
    
    Args:
        rotation: 3x3 rotation matrix
        
    Returns:
        Quaternion as (x, y, z, w)
    """
    if not ROS2_AVAILABLE:
        # Fallback implementation
        trace = np.trace(rotation)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (rotation[2, 1] - rotation[1, 2]) * s
            y = (rotation[0, 2] - rotation[2, 0]) * s
            z = (rotation[1, 0] - rotation[0, 1]) * s
        else:
            if rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
                w = (rotation[2, 1] - rotation[1, 2]) / s
                x = 0.25 * s
                y = (rotation[0, 1] + rotation[1, 0]) / s
                z = (rotation[0, 2] + rotation[2, 0]) / s
            elif rotation[1, 1] > rotation[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
                w = (rotation[0, 2] - rotation[2, 0]) / s
                x = (rotation[0, 1] + rotation[1, 0]) / s
                y = 0.25 * s
                z = (rotation[1, 2] + rotation[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
                w = (rotation[1, 0] - rotation[0, 1]) / s
                x = (rotation[0, 2] + rotation[2, 0]) / s
                y = (rotation[1, 2] + rotation[2, 1]) / s
                z = 0.25 * s
        return (x, y, z, w)
    else:
        rot_4x4 = np.eye(4)
        rot_4x4[0:3, 0:3] = rotation
        quat = quaternion_from_matrix(rot_4x4)
        return tuple(quat)


def create_joint_state_msg(positions: Dict[str, float]) -> 'JointState':
    """
    Create a JointState message from a dictionary of positions.
    
    Args:
        positions: Dictionary of joint angles in degrees
        
    Returns:
        JointState message
    """
    if not ROS2_AVAILABLE:
        raise RuntimeError("ROS2 is not available")
    
    msg = JointState()
    
    for joint in JOINT_NAMES:
        msg.name.append(joint)
        if joint == 'gripper':
            msg.position.append(positions.get(joint, 0.0) / 100.0 * 0.04)
        else:
            msg.position.append(np.deg2rad(positions.get(joint, 0.0)))
    
    return msg


# ==============================================================================
# LAUNCH FILE HELPERS
# ==============================================================================

def get_rviz_launch_command() -> str:
    """Get the command to launch RViz for SO-101 visualization."""
    return "ros2 launch lerobot_description so101_display.launch.py"


def get_gazebo_launch_command() -> str:
    """Get the command to launch Gazebo with SO-101."""
    return "ros2 launch lerobot_description so101_gazebo.launch.py"


def get_moveit_launch_command() -> str:
    """Get the command to launch MoveIt for SO-101."""
    return "ros2 launch lerobot_moveit so101_moveit.launch.py"


# ==============================================================================
# MAIN - Test ROS2 utilities
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SO-101 ROS2 Utilities Test")
    print("=" * 60)
    
    print(f"\nROS2 Available: {ROS2_AVAILABLE}")
    
    if ROS2_AVAILABLE:
        print("\nTesting quaternion conversion...")
        test_rotation = np.eye(3)
        quat = rotation_to_quaternion(test_rotation)
        print(f"  Identity rotation -> quaternion: {quat}")
        
        print("\nLaunch commands:")
        print(f"  RViz:   {get_rviz_launch_command()}")
        print(f"  Gazebo: {get_gazebo_launch_command()}")
        print(f"  MoveIt: {get_moveit_launch_command()}")
    else:
        print("\nTo enable ROS2 features:")
        print("  1. Install ROS2 Jazzy")
        print("  2. source /opt/ros/jazzy/setup.bash")
        print("  3. Build and source the lerobot_ws workspace")
    
    print("\n" + "=" * 60)
