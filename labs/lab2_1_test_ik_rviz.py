#!/usr/bin/env python3
"""
Lab 2.1 (RViz): Test Inverse Kinematics with ROS2
=================================================

This lab visualizes inverse kinematics using ROS2 and RViz.
Tests IK by commanding the robot to reach target positions.

Prerequisites:
    1. ROS2 installed (Humble or Jazzy)
    2. lerobot_ws workspace built
    3. Terminal 1: ros2 launch lerobot_description so101_display.launch.py

Usage:
    python labs/lab2_1_test_ik_rviz.py

Learning Objectives:
    1. Understand the IK problem and solutions
    2. Visualize IK-computed configurations in RViz
    3. Compare numerical vs geometric IK approaches

Author: SO-101 Robotics Course
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from so101_forward_kinematics import get_forward_kinematics
from so101_inverse_kinematics import (
    inverse_kinematics_numerical,
    inverse_kinematics_geometric
)

# Check for ROS2
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


def rotation_to_quaternion(R):
    """Convert rotation matrix to quaternion (x, y, z, w)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return (x, y, z, w)


class IKVisualizationNode(Node):
    """ROS2 node for IK visualization."""
    
    def __init__(self):
        super().__init__('ik_visualization')
        
        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, '/ik_markers', 10)
        
        # TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # Joint names
        self.joint_names = [
            'shoulder_pan', 'shoulder_lift', 'elbow_flex',
            'wrist_flex', 'wrist_roll', 'gripper'
        ]
        
        # Current configuration
        self.joint_config = {
            'shoulder_pan': 0.0,
            'shoulder_lift': 0.0,
            'elbow_flex': 0.0,
            'wrist_flex': 0.0,
            'wrist_roll': 0.0,
            'gripper': 50.0
        }
        
        # Target position (for visualization)
        self.target_position = np.array([0.2, 0.1, 0.15])
        
        # Timer for updates
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        self.get_logger().info('IK Visualization Node started')
    
    def set_configuration(self, config):
        """Update joint configuration."""
        self.joint_config.update(config)
    
    def set_target(self, target_pos):
        """Set target position for IK."""
        self.target_position = np.array(target_pos)
    
    def timer_callback(self):
        """Publish joint states, markers, and TF frames."""
        # Publish joint states
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.header.frame_id = ''
        
        for joint in self.joint_names:
            js_msg.name.append(joint)
            if joint == 'gripper':
                js_msg.position.append(self.joint_config[joint] / 100.0 * 0.04)
            else:
                js_msg.position.append(np.deg2rad(self.joint_config[joint]))
        
        self.joint_pub.publish(js_msg)
        
        # Compute FK for current configuration
        position, rotation = get_forward_kinematics(self.joint_config)
        
        # Publish FK frame as TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = ''
        t.child_frame_id = 'ik_current_tool'
        
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        
        quat = rotation_to_quaternion(rotation)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)
        
        # Publish target marker (red sphere)
        self.publish_target_marker()
        
        # Publish achieved position marker (green sphere)
        self.publish_achieved_marker(position)
    
    def publish_target_marker(self):
        """Publish red sphere at target position."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base_link'
        marker.ns = 'ik_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(self.target_position[0])
        marker.pose.position.y = float(self.target_position[1])
        marker.pose.position.z = float(self.target_position[2])
        
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.04
        marker.scale.y = 0.04
        marker.scale.z = 0.04
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        self.marker_pub.publish(marker)
    
    def publish_achieved_marker(self, position):
        """Publish green sphere at achieved position."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base_link'
        marker.ns = 'ik_achieved'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.03
        marker.scale.y = 0.03
        marker.scale.z = 0.03
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.8
        
        self.marker_pub.publish(marker)


def main():
    if not ROS2_AVAILABLE:
        print("Error: ROS2 is not available.")
        print("\nTo run this lab:")
        print("  1. Install ROS2 (Humble or Jazzy)")
        print("     Ubuntu 22.04: source /opt/ros/humble/setup.bash")
        print("     Ubuntu 24.04: source /opt/ros/jazzy/setup.bash")
        print("  2. Build lerobot_ws: cd ros2_ws && colcon build")
        print("  3. source ros2_ws/install/setup.bash")
        print("\nThen in another terminal:")
        print("  ros2 launch lerobot_description so101_display.launch.py")
        return
    
    print("=" * 70)
    print("Lab 2.1 (RViz): Inverse Kinematics Visualization")
    print("=" * 70)
    
    print("\nMake sure RViz is running in another terminal:")
    print("  ros2 launch lerobot_description so101_display.launch.py")
    print("\nIn RViz, add:")
    print("  - TF display (to see ik_current_tool frame)")
    print("  - Marker display (topic: /ik_markers)")
    print("    * Red sphere = Target position")
    print("    * Green sphere = Achieved position (should overlap if IK succeeded)")
    
    # Test targets within workspace
    test_cases = [
        {
            'name': 'Forward Reach',
            'target': np.array([0.25, 0.0, 0.15]),
            'method': 'numerical'
        },
        {
            'name': 'Right Side',
            'target': np.array([0.15, 0.15, 0.10]),
            'method': 'numerical'
        },
        {
            'name': 'Left Side',
            'target': np.array([0.15, -0.15, 0.10]),
            'method': 'geometric'
        },
        {
            'name': 'High Reach',
            'target': np.array([0.20, 0.05, 0.25]),
            'method': 'numerical'
        },
    ]
    
    rclpy.init()
    node = IKVisualizationNode()
    
    try:
        test_idx = 0
        last_switch = time.time()
        switch_interval = 6.0  # Switch every 6 seconds
        
        print(f"\nCycling through {len(test_cases)} test cases every {switch_interval}s")
        print("Press Ctrl+C to exit.\n")
        
        while rclpy.ok():
            # Switch to next test case
            if time.time() - last_switch > switch_interval:
                test_idx = (test_idx + 1) % len(test_cases)
                last_switch = time.time()
                
                current_test = test_cases[test_idx]
                target_pos = current_test['target']
                method = current_test['method']
                
                print("\n" + "=" * 70)
                print(f"Test Case {test_idx + 1}/{len(test_cases)}: {current_test['name']}")
                print("=" * 70)
                print(f"Target Position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                print(f"IK Method: {method}")
                
                # Update target marker
                node.set_target(target_pos)
                
                # Compute IK
                if method == 'numerical':
                    joint_config, success, error = inverse_kinematics_numerical(
                        target_pos,
                        initial_guess=None,
                        verbose=False
                    )
                    print(f"\nNumerical IK Result:")
                    print(f"  Success: {success}")
                    print(f"  Final error: {error:.6f} m")
                else:
                    joint_config, success = inverse_kinematics_geometric(
                        target_pos,
                        elbow_up=True
                    )
                    print(f"\nGeometric IK Result:")
                    print(f"  Success: {success}")
                
                if success:
                    print(f"  Joint angles:")
                    for joint, angle in joint_config.items():
                        if joint != 'gripper':
                            print(f"    {joint}: {angle:.2f}°")
                    
                    # Verify with FK
                    verify_pos, _ = get_forward_kinematics(joint_config)
                    error = np.linalg.norm(verify_pos - target_pos)
                    print(f"\n  FK Verification:")
                    print(f"    Achieved: [{verify_pos[0]:.3f}, {verify_pos[1]:.3f}, {verify_pos[2]:.3f}]")
                    print(f"    Position error: {error:.6f} m")
                    
                    # Update robot configuration
                    node.set_configuration(joint_config)
                else:
                    print("  ✗ IK failed! Target may be outside workspace.")
                
                print("\nWatch RViz:")
                print("  - Robot should move to reach target")
                print("  - Green sphere should overlap red sphere if successful")
            
            rclpy.spin_once(node, timeout_sec=0.01)
    
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()