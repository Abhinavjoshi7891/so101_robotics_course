#!/usr/bin/env python3
"""
Lab 2.1 (RViz): Test Inverse Kinematics with ROS2 - FIXED
==========================================================

This lab visualizes inverse kinematics using ROS2 and RViz.
Tests IK by commanding the robot to reach target positions.

FIXED: Targets chosen for URDF coordinate system workspace

Prerequisites:
    1. Terminal 1: ros2 launch lerobot_description so101_display_no_jsp.launch.py
    2. Terminal 2: python labs/lab2_1_test_ik_rviz.py

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
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from sensor_msgs.msg import JointState
    from visualization_msgs.msg import Marker
    from geometry_msgs.msg import Point, TransformStamped
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


def interpolate_configs(config_start, config_target, alpha):
    """Linearly interpolate between two joint configurations."""
    result = {}
    for key in config_start.keys():
        start_val = config_start[key]
        target_val = config_target[key]
        result[key] = start_val + alpha * (target_val - start_val)
    return result


class IKVisualizationNode(Node):
    """ROS2 node for IK visualization."""
    
    def __init__(self):
        super().__init__('ik_visualization')
        
        # IK/FK Mode - use urdf_native for RViz
        self.mode = 'urdf_native'
        
        # SETUP PUBLISHERS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.joint_pub = self.create_publisher(JointState, '/joint_states', qos_profile)
        self.marker_pub = self.create_publisher(Marker, '/ik_markers', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # SETUP TF LISTENER
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # JOINT MAPPING (match so101_display_no_jsp.launch.py expectations)
        self.joint_mapping = {
            'shoulder_pan': '1', 'shoulder_lift': '2', 'elbow_flex': '3',
            'wrist_flex': '4', 'wrist_roll': '5', 'gripper': '6'
        }
        
        # Configurations
        self.start_joint_config = {
            'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 'elbow_flex': 0.0,
            'wrist_flex': 0.0, 'wrist_roll': 0.0, 'gripper': 50.0
        }
        self.target_joint_config = self.start_joint_config.copy()
        self.current_joint_config = self.start_joint_config.copy()
        
        # Target position
        self.target_position = np.array([0.02, -0.28, 0.27])
        
        # State machine
        self.state = 'HOLDING'  # 'MOVING' or 'HOLDING'
        self.state_start_time = time.time()
        self.MOVE_DURATION = 3.0
        self.HOLD_DURATION = 2.0
        
        # Timer for updates
        self.timer = self.create_timer(0.05, self.timer_callback)
        
        # Frame tracking
        self.target_frames = ['gripper', '5', 'link5', 'wrist_roll_link']
        self.active_target_frame = None
        
        self.get_logger().info(f'IK Visualization Node started (mode={self.mode})')
    
    def set_target_configuration(self, config, target_pos):
        """Set new target configuration and position."""
        self.start_joint_config = self.current_joint_config.copy()
        self.target_joint_config = config.copy()
        self.target_position = np.array(target_pos)
        self.state = 'MOVING'
        self.state_start_time = time.time()
    
    def timer_callback(self):
        """Publish joint states, markers, and TF frames."""
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        # State machine
        if self.state == 'MOVING':
            # Interpolate to target
            alpha = min(1.0, elapsed / self.MOVE_DURATION)
            # Smooth interpolation (ease in-out)
            s = 3 * alpha**2 - 2 * alpha**3
            self.current_joint_config = interpolate_configs(
                self.start_joint_config,
                self.target_joint_config,
                s
            )
            
            if elapsed >= self.MOVE_DURATION:
                self.state = 'HOLDING'
                self.state_start_time = current_time
                self.current_joint_config = self.target_joint_config.copy()
                
        elif self.state == 'HOLDING':
            # Hold at target
            pass
        
        # --- 1. Publish Joint States ---
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.header.frame_id = ''
        
        for name, value in self.current_joint_config.items():
            if name in self.joint_mapping:
                js_msg.name.append(self.joint_mapping[name])
                if name == 'gripper':
                    js_msg.position.append(value / 100.0 * 0.04)
                else:
                    js_msg.position.append(np.deg2rad(value))
        
        self.joint_pub.publish(js_msg)
        
        # --- 2. Compute FK for current configuration ---
        position, rotation = get_forward_kinematics(self.current_joint_config, mode=self.mode)
        
        # --- 3. Get URDF position for comparison ---
        urdf_pos = None
        
        if self.active_target_frame is None:
            for frame in self.target_frames:
                try:
                    t = self.tf_buffer.lookup_transform(
                        'base',
                        frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    self.active_target_frame = frame
                    self.get_logger().info(f'Using target frame: {frame}')
                    break
                except:
                    pass
        
        if self.active_target_frame:
            try:
                t = self.tf_buffer.lookup_transform(
                    'base',
                    self.active_target_frame,
                    rclpy.time.Time()
                )
                urdf_pos = np.array([
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z
                ])
            except TransformException:
                pass
        
        # --- 4. Print status ---
        if urdf_pos is not None:
            fk_error = np.linalg.norm(position - urdf_pos) * 1000.0
            target_error = np.linalg.norm(urdf_pos - self.target_position) * 1000.0
            
            state_str = f"[{self.state:7s}]"
            if self.state == 'MOVING':
                progress = f"{int(s * 100):3d}%"
            else:
                progress = "HOLD"
            
            print(f"\r{state_str} {progress} | "
                  f"FK: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] | "
                  f"URDF: [{urdf_pos[0]:.3f}, {urdf_pos[1]:.3f}, {urdf_pos[2]:.3f}] | "
                  f"FK_Err: {fk_error:5.2f}mm | Target_Err: {target_error:5.2f}mm", end="")
        
        # --- 5. Publish Markers ---
        # RED sphere at target position
        self.publish_target_marker()
        
        # GREEN sphere at achieved position (FK)
        self.publish_achieved_marker(position)
        
        # --- 6. Publish TF frame ---
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base'
        t.child_frame_id = 'ik_current_tool'
        t.transform.translation.x = float(position[0])
        t.transform.translation.y = float(position[1])
        t.transform.translation.z = float(position[2])
        quat = rotation_to_quaternion(rotation)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)
    
    def publish_target_marker(self):
        """Publish red sphere at target position."""
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base'
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
        marker.header.frame_id = 'base'
        marker.ns = 'ik_achieved'
        marker.id = 1
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.035
        marker.scale.y = 0.035
        marker.scale.z = 0.035
        
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 0.9
        
        self.marker_pub.publish(marker)


def main():
    if not ROS2_AVAILABLE:
        print("Error: ROS2 is not available.")
        print("\nTo run this lab:")
        print("  1. Install ROS2 (Humble or Jazzy)")
        print("  2. source ros2_ws/install/setup.bash")
        print("\nThen in another terminal:")
        print("  ros2 launch lerobot_description so101_display_no_jsp.launch.py")
        return
    
    print("=" * 80)
    print("Lab 2.1 (RViz): Inverse Kinematics Visualization")
    print("=" * 80)
    
    print("\nPrerequisites:")
    print("  Terminal 1: ros2 launch lerobot_description so101_display_no_jsp.launch.py")
    print("  Terminal 2: python labs/lab2_1_test_ik_rviz.py")
    
    print("\nIn RViz, add:")
    print("  - Marker display (topic: /ik_markers)")
    print("    * RED sphere = IK target position")
    print("    * GREEN sphere = Achieved position (should overlap if IK succeeded)")
    
    # CRITICAL: Test targets chosen for URDF workspace!
    # URDF home position: [0.0206, -0.2775, 0.2669]
    # Workspace: X: -0.2 to 0.2, Y: -0.4 to -0.1, Z: 0.1 to 0.4
    test_cases = [
        {
            'name': 'Home',
            'target': np.array([0.02, -0.28, 0.27]),  # Near home
            'method': 'numerical'
        },
        {
            'name': 'Forward Reach',
            'target': np.array([0.02, -0.35, 0.25]),  # More negative Y
            'method': 'numerical'
        },
        {
            'name': 'Right Side',
            'target': np.array([0.12, -0.25, 0.24]),  # Positive X
            'method': 'numerical'
        },
        {
            'name': 'Left Side',
            'target': np.array([-0.12, -0.25, 0.24]),  # Negative X
            'method': 'numerical'
        },
        {
            'name': 'High Reach',
            'target': np.array([0.02, -0.22, 0.35]),  # Higher Z
            'method': 'numerical'
        },
    ]
    
    rclpy.init()
    node = IKVisualizationNode()
    
    try:
        test_idx = 0
        last_switch = time.time()
        switch_interval = 5.0  # 3s move + 2s hold
        
        print(f"\n\nCycling through {len(test_cases)} test cases every {switch_interval}s")
        print("Press Ctrl+C to exit.\n")
        
        # Start at home
        print("=" * 80)
        print("Starting at HOME position")
        print("=" * 80)
        
        while rclpy.ok():
            # Switch to next test case
            if time.time() - last_switch > switch_interval:
                test_idx = (test_idx + 1) % len(test_cases)
                last_switch = time.time()
                
                current_test = test_cases[test_idx]
                target_pos = current_test['target']
                method = current_test['method']
                
                print("\n\n" + "=" * 80)
                print(f"Test Case {test_idx + 1}/{len(test_cases)}: {current_test['name']}")
                print("=" * 80)
                print(f"Target Position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                print(f"IK Method: {method}")
                
                # Compute IK with higher tolerance and more iterations
                joint_config, success, error = inverse_kinematics_numerical(
                    target_pos,
                    initial_guess=node.current_joint_config,  # Use current as initial guess
                    verbose=False,
                    mode='urdf_native',
                    max_iterations=300,  # More iterations
                    tolerance=0.02  # More lenient tolerance (20mm)
                )
                
                print(f"\nNumerical IK Result:")
                print(f"  Success: {success}")
                print(f"  Final error: {error:.6f} m ({error*1000:.2f} mm)")
                
                if not success:
                    print(f"  ⚠ IK didn't fully converge, but using best solution")
                
                # Always show joint angles
                print(f"  Joint angles:")
                for joint, angle in joint_config.items():
                    if joint != 'gripper':
                        print(f"    {joint:15s}: {angle:7.2f}°")
                
                # Verify with FK
                verify_pos, _ = get_forward_kinematics(joint_config, mode='urdf_native')
                fk_error = np.linalg.norm(verify_pos - target_pos)
                print(f"\n  FK Verification:")
                print(f"    Target:   [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                print(f"    Achieved: [{verify_pos[0]:.3f}, {verify_pos[1]:.3f}, {verify_pos[2]:.3f}]")
                print(f"    Position error: {fk_error:.6f} m ({fk_error*1000:.2f} mm)")
                
                if fk_error * 1000 < 1.0:
                    print("    ✓ EXCELLENT: < 1mm error")
                elif fk_error * 1000 < 10.0:
                    print("    ✓ GOOD: < 10mm error")
                elif fk_error * 1000 < 50.0:
                    print("    ✓ ACCEPTABLE: < 50mm error")
                else:
                    print("    ⚠ LARGE ERROR: > 50mm")
                
                # ALWAYS update robot - even if IK didn't fully converge
                node.set_target_configuration(joint_config, target_pos)
                
                print("\nWatch RViz:")
                print("  - Robot will move to reach target (3 seconds)")
                print("  - Green sphere tracks robot position")
                print("  - Red sphere shows IK target")
            
            rclpy.spin_once(node, timeout_sec=0.01)
    
    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("Shutting down...")
        print("=" * 80)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()