#!/usr/bin/env python3
"""
Lab 1.2 (RViz): Test Forward Kinematics with ROS2
=================================================

This lab visualizes forward kinematics using ROS2 and RViz.
The FK-computed tool frame is published as a TF frame for visualization.

Prerequisites:
    1. ROS2 Jazzy installed
    2. lerobot_ws workspace built
    3. Terminal 1: ros2 launch lerobot_description so101_display.launch.py

Usage:
    python labs/lab1_2_test_fk_rviz.py

Learning Objectives:
    1. Understand ROS2 joint states and TF
    2. Visualize FK results in RViz
    3. Compare FK computation with URDF visualization

Author: SO-101 Robotics Course
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from so101_forward_kinematics import get_forward_kinematics, get_intermediate_transforms

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


class FKVisualizationNode(Node):
    """ROS2 node for FK visualization."""
    
    def __init__(self):
        super().__init__('fk_visualization')
        
        # Publishers
        self.joint_pub = self.create_publisher(JointState, '/joint_states', 10)
        self.marker_pub = self.create_publisher(Marker, '/fk_marker', 10)
        
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
        
        # Timer
        self.timer = self.create_timer(0.02, self.timer_callback)
        
        self.get_logger().info('FK Visualization Node started')
    
    def set_configuration(self, config):
        """Update joint configuration."""
        self.joint_config.update(config)
    
    def timer_callback(self):
        """Publish joint states and FK frame."""
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
        
        # Compute FK
        position, rotation = get_forward_kinematics(self.joint_config)
        
        # Publish FK frame as TF
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base_link'
        t.child_frame_id = 'fk_tool_frame'
        
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        
        quat = rotation_to_quaternion(rotation)
        t.transform.rotation.x = quat[0]
        t.transform.rotation.y = quat[1]
        t.transform.rotation.z = quat[2]
        t.transform.rotation.w = quat[3]
        
        self.tf_broadcaster.sendTransform(t)
        
        # Publish marker at FK position
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base_link'
        marker.ns = 'fk_visualization'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        
        marker.scale.x = 0.03  # diameter
        marker.scale.y = 0.03
        marker.scale.z = 0.08  # height
        
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.7
        
        self.marker_pub.publish(marker)


def main():
    if not ROS2_AVAILABLE:
        print("Error: ROS2 is not available.")
        print("\nTo run this lab:")
        print("  1. Install ROS2 Jazzy")
        print("  2. source /opt/ros/jazzy/setup.bash")
        print("  3. Build lerobot_ws: cd ros2_ws && colcon build")
        print("  4. source ros2_ws/install/setup.bash")
        print("\nThen in another terminal:")
        print("  ros2 launch lerobot_description so101_display.launch.py")
        return
    
    print("=" * 70)
    print("Lab 1.2 (RViz): Forward Kinematics Visualization")
    print("=" * 70)
    
    print("\nMake sure RViz is running:")
    print("  ros2 launch lerobot_description so101_display.launch.py")
    print("\nIn RViz, add:")
    print("  - TF display (to see fk_tool_frame)")
    print("  - Marker display (topic: /fk_marker)")
    
    # Test configurations
    test_configs = [
        {
            'name': 'Home',
            'config': {
                'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 'elbow_flex': 0.0,
                'wrist_flex': 0.0, 'wrist_roll': 0.0, 'gripper': 50.0
            }
        },
        {
            'name': 'Test 1',
            'config': {
                'shoulder_pan': -45.0, 'shoulder_lift': 45.0, 'elbow_flex': -45.0,
                'wrist_flex': 90.0, 'wrist_roll': 0.0, 'gripper': 10.0
            }
        },
        {
            'name': 'Test 2',
            'config': {
                'shoulder_pan': 30.0, 'shoulder_lift': 30.0, 'elbow_flex': -60.0,
                'wrist_flex': 45.0, 'wrist_roll': 45.0, 'gripper': 50.0
            }
        },
    ]
    
    rclpy.init()
    node = FKVisualizationNode()
    
    try:
        config_idx = 0
        last_switch = time.time()
        switch_interval = 5.0  # Switch every 5 seconds
        
        print(f"\nSwitching between {len(test_configs)} configurations every {switch_interval}s")
        print("Press Ctrl+C to exit.\n")
        
        while rclpy.ok():
            # Switch configuration periodically
            if time.time() - last_switch > switch_interval:
                config_idx = (config_idx + 1) % len(test_configs)
                last_switch = time.time()
                
                current = test_configs[config_idx]
                node.set_configuration(current['config'])
                
                print(f"\n--- Configuration: {current['name']} ---")
                print(f"Joint angles: {current['config']}")
                
                pos, rot = get_forward_kinematics(current['config'])
                print(f"FK Position: x={pos[0]:.4f}, y={pos[1]:.4f}, z={pos[2]:.4f}")
            
            rclpy.spin_once(node, timeout_sec=0.01)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
