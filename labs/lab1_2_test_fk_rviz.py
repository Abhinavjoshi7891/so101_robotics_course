#!/usr/bin/env python3
"""
Lab 1.2 (RViz): Test Forward Kinematics with ROS2 (Debug Mode)
==============================================================

This script visualizes FK and compares it against the URDF ground truth.
It logs the precise error between your math (Red Marker) and the URDF (Robot Mesh).

Prerequisites:
    1. ros2 launch lerobot_description so101_display_no_jsp.launch.py

Usage:
    python3 labs/lab1_2_test_fk_rviz.py
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import intermediate transforms to compare specific joints
from so101_forward_kinematics import get_forward_kinematics, get_intermediate_transforms

# Check for ROS2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import TransformStamped
    import tf2_ros
    from tf2_ros import TransformException
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener
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
    """ROS2 node for FK visualization and debugging."""
    
    def __init__(self):
        super().__init__('fk_visualization')
        
        # 1. SETUP PUBLISHERS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.joint_pub = self.create_publisher(JointState, '/joint_states', qos_profile)
        self.marker_pub = self.create_publisher(Marker, '/fk_marker', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # 2. SETUP TF LISTENER (To spy on the real URDF position)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 3. JOINT MAPPING & CONFIG
        self.joint_mapping = {
            'shoulder_pan': '1', 'shoulder_lift': '2', 'elbow_flex': '3',
            'wrist_flex': '4', 'wrist_roll': '5', 'gripper': '6'
        }
        self.joint_config = {
            'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 'elbow_flex': 0.0,
            'wrist_flex': 0.0, 'wrist_roll': 0.0, 'gripper': 50.0
        }
        
        self.timer = self.create_timer(0.05, self.timer_callback) # 20Hz
        self.get_logger().info('FK Debug Node started - Watch terminal for errors!')

    def set_configuration(self, config):
        self.joint_config.update(config)
    
    def timer_callback(self):
        # --- 1. Publish Joint States (Drive the Robot) ---
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.header.frame_id = ''
        
        for name, value in self.joint_config.items():
            if name in self.joint_mapping:
                js_msg.name.append(self.joint_mapping[name])
                if name == 'gripper':
                    js_msg.position.append(value / 100.0 * 0.04)
                else:
                    js_msg.position.append(np.deg2rad(value))
        self.joint_pub.publish(js_msg)
        
        # --- 2. Compute FK (Where WE think the robot is) ---
        # We use 'joint5' (base of gripper) to compare with URDF 'gripper' link
        transforms = get_intermediate_transforms(self.joint_config)
        
        # Get Joint 5 position (this corresponds to the 'gripper' link frame)
        fk_mat = transforms['joint5']
        fk_pos = fk_mat[0:3, 3]
        fk_rot = fk_mat[0:3, 0:3]
        
        # --- 3. Lookup URDF Transform (Where ROS thinks the robot is) ---
        urdf_pos = None
        target_frame = 'gripper'  # <--- CHANGED FROM 'gripperframe'
        
        try:
            # Look up transform from 'base' to 'gripper'
            t = self.tf_buffer.lookup_transform(
                'base',
                target_frame,
                rclpy.time.Time()
            )
            urdf_pos = np.array([
                t.transform.translation.x,
                t.transform.translation.y,
                t.transform.translation.z
            ])
        except TransformException as e:
            # PRINT THE ERROR so we know what's wrong!
            # Only print once every second to avoid spamming
            if int(time.time()) % 2 == 0:
                print(f"\r[Waiting for TF] {e}", end="")
            pass

        # --- 4. Log the Difference ---
        if urdf_pos is not None:
            error = fk_pos - urdf_pos
            dist_error = np.linalg.norm(error) * 1000.0 # to mm
            
            # Print cleanly
            print(f"\rFK(J5): [{fk_pos[0]:.4f}, {fk_pos[1]:.4f}, {fk_pos[2]:.4f}] | "
                  f"URDF: [{urdf_pos[0]:.4f}, {urdf_pos[1]:.4f}, {urdf_pos[2]:.4f}] | "
                  f"Err: {dist_error:6.2f} mm", end="")

        # --- 5. Publish Visualization ---
        # TF Frame
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base'
        t.child_frame_id = 'fk_debug_frame'
        t.transform.translation.x = float(fk_pos[0])
        t.transform.translation.y = float(fk_pos[1])
        t.transform.translation.z = float(fk_pos[2])
        quat = rotation_to_quaternion(fk_rot)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)
        
        # Marker
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base'
        marker.ns = 'fk_visualization'
        marker.id = 0
        marker.type = Marker.SPHERE  # Changed to sphere to see the point clearly
        marker.action = Marker.ADD
        marker.pose.position.x = float(fk_pos[0])
        marker.pose.position.y = float(fk_pos[1])
        marker.pose.position.z = float(fk_pos[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.03; marker.scale.y = 0.03; marker.scale.z = 0.03
        marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 0.8
        self.marker_pub.publish(marker)


def main():
    if not ROS2_AVAILABLE:
        print("ROS2 not found.")
        return
    
    rclpy.init()
    node = FKVisualizationNode()
    
    # Test Configs
    test_configs = [
        {'name': 'Home', 'config': {'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 'elbow_flex': 0.0, 'wrist_flex': 0.0, 'wrist_roll': 0.0, 'gripper': 50.0}},
        {'name': 'Reach', 'config': {'shoulder_pan': 0.0, 'shoulder_lift': 30.0, 'elbow_flex': -45.0, 'wrist_flex': 20.0, 'wrist_roll': 0.0, 'gripper': 50.0}},
        {'name': 'Side',  'config': {'shoulder_pan': 45.0, 'shoulder_lift': 20.0, 'elbow_flex': -20.0, 'wrist_flex': 0.0, 'wrist_roll': 90.0, 'gripper': 50.0}},
    ]

    try:
        config_idx = 0
        last_switch = time.time()
        
        print("\n" + "="*80)
        print("FK DEBUG MODE: Comparing Math (Joint 5) vs URDF (Gripper Link)")
        print("="*80 + "\n")
        
        while rclpy.ok():
            if time.time() - last_switch > 5.0:
                print() # Newline after carriage return prints
                config_idx = (config_idx + 1) % len(test_configs)
                current = test_configs[config_idx]
                node.set_configuration(current['config'])
                print(f"--- Config: {current['name']} ---")
                last_switch = time.time()
            
            rclpy.spin_once(node, timeout_sec=0.01)
            
    except KeyboardInterrupt:
        print("\nShutdown.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()