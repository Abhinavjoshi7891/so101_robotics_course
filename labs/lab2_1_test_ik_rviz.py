#!/usr/bin/env python3
"""
Lab 2.1 (RViz): Test Inverse Kinematics - Using FK Test Configurations
=======================================================================

This version uses the SAME joint configurations from the working FK test.
We compute their positions using FK, then verify IK can recover them.

Prerequisites:
    1. Terminal 1: ros2 launch lerobot_description so101_display_no_jsp.launch.py
    2. Terminal 2: python labs/lab2_1_test_ik_rviz.py

Author: SO-101 Robotics Course
"""

import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from so101_forward_kinematics import get_forward_kinematics
from so101_inverse_kinematics import inverse_kinematics_numerical

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
        
        # JOINT MAPPING
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
        self.state = 'PREVIEW'  # 'PREVIEW' or 'MOVING'
        self.state_start_time = time.time()
        self.PREVIEW_DURATION = 2.0
        self.MOVING_DURATION = 3.0
        
        # Timer
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
        self.state = 'PREVIEW'
        self.state_start_time = time.time()
    
    def timer_callback(self):
        """Publish joint states, markers, and TF frames."""
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        # State machine (SAME AS FK NODE!)
        if self.state == 'PREVIEW':
            # Keep robot at start, show RED marker at target
            self.current_joint_config = self.start_joint_config.copy()
            
            if elapsed >= self.PREVIEW_DURATION:
                self.state = 'MOVING'
                self.state_start_time = current_time
                self.get_logger().info('→ Robot moving to target...')
                
        elif self.state == 'MOVING':
            # Interpolate to target
            alpha = min(1.0, elapsed / self.MOVING_DURATION)
            self.current_joint_config = interpolate_configs(
                self.start_joint_config,
                self.target_joint_config,
                alpha
            )
        
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
        
        # --- 2. Compute FK ---
        position, rotation = get_forward_kinematics(self.current_joint_config, mode=self.mode)
        
        # Compute FK for target too
        target_fk_pos, _ = get_forward_kinematics(self.target_joint_config, mode=self.mode)
        
        # --- 3. Get URDF position ---
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
            target_error = np.linalg.norm(urdf_pos - target_fk_pos) * 1000.0
            
            if self.state == 'PREVIEW':
                state_str = "PREVIEW"
                progress = f"{elapsed:.1f}/{self.PREVIEW_DURATION:.1f}s"
            else:
                state_str = "MOVING "
                alpha = min(1.0, elapsed / self.MOVING_DURATION)
                progress = f"{int(alpha*100):3d}%"
            
            print(f"\r[{state_str} {progress}] FK: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}] | "
                  f"URDF: [{urdf_pos[0]:.3f}, {urdf_pos[1]:.3f}, {urdf_pos[2]:.3f}] | "
                  f"FK_Err: {fk_error:5.2f}mm | Target_Err: {target_error:5.2f}mm", end="")
        
        # --- 5. Publish Markers ---
        # RED sphere at target (during PREVIEW), GREEN during MOVING
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base'
        marker.ns = 'ik_target'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(target_fk_pos[0])
        marker.pose.position.y = float(target_fk_pos[1])
        marker.pose.position.z = float(target_fk_pos[2])
        marker.pose.orientation.w = 1.0
        
        marker.scale.x = 0.04
        marker.scale.y = 0.04
        marker.scale.z = 0.04
        
        if self.state == 'PREVIEW':
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 0.9
        else:
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 0.7
        
        self.marker_pub.publish(marker)
        
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


def main():
    if not ROS2_AVAILABLE:
        print("Error: ROS2 is not available.")
        return
    
    print("=" * 80)
    print("Lab 2.1 (RViz): Inverse Kinematics Visualization")
    print("=" * 80)
    
    print("\nUsing FK test's proven joint configurations!")
    print("These are known to work - we'll use IK to verify we can recover them.\n")
    
    # Use SAME configurations as FK test (we know these work!)
    fk_test_configs = [
        {'name': 'Home',  'config': {'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 'elbow_flex': 0.0, 'wrist_flex': 0.0, 'wrist_roll': 0.0, 'gripper': 50.0}},
        {'name': 'Reach', 'config': {'shoulder_pan': 0.0, 'shoulder_lift': 30.0, 'elbow_flex': -45.0, 'wrist_flex': 20.0, 'wrist_roll': 0.0, 'gripper': 50.0}},
        {'name': 'Side',  'config': {'shoulder_pan': 45.0, 'shoulder_lift': 20.0, 'elbow_flex': -20.0, 'wrist_flex': 0.0, 'wrist_roll': 90.0, 'gripper': 50.0}},
        {'name': 'Up',    'config': {'shoulder_pan': 0.0, 'shoulder_lift': -30.0, 'elbow_flex': -60.0, 'wrist_flex': 90.0, 'wrist_roll': 0.0, 'gripper': 50.0}},
    ]
    
    # Compute FK positions for each config
    test_cases = []
    print("Computing target positions from known joint configurations:")
    print("-" * 80)
    for item in fk_test_configs:
        config = item['config']
        position, _ = get_forward_kinematics(config, mode='urdf_native')
        test_cases.append({
            'name': item['name'],
            'joint_config': config,
            'target_position': position
        })
        print(f"{item['name']:10s}: pos=[{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")
    
    rclpy.init()
    node = IKVisualizationNode()
    
    try:
        test_idx = 0
        last_switch = time.time()
        switch_interval = 5.0  # 2s preview + 3s moving
        
        print(f"\n\nCycling through {len(test_cases)} test cases every {switch_interval}s")
        print("Press Ctrl+C to exit.\n")
        
        print("=" * 80)
        print("Starting tests...")
        print("=" * 80)
        
        while rclpy.ok():
            # Switch to next test case
            if time.time() - last_switch > switch_interval:
                test_idx = (test_idx + 1) % len(test_cases)
                last_switch = time.time()
                
                current_test = test_cases[test_idx]
                target_pos = current_test['target_position']
                original_config = current_test['joint_config']
                
                print("\n\n" + "=" * 80)
                print(f"Test Case {test_idx + 1}/{len(test_cases)}: {current_test['name']}")
                print("=" * 80)
                print(f"Original joint config: {original_config}")
                print(f"Target position (from FK): [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
                
                # Run IK to recover the joint angles
                print("\nRunning IK to recover joint angles...")
                recovered_config, success, error = inverse_kinematics_numerical(
                    target_pos,
                    initial_guess=None,  # Start from home, not current
                    verbose=True,
                    mode='urdf_native',
                    max_iterations=200
                )
                
                print(f"\nIK Result:")
                print(f"  Success: {success}")
                print(f"  Error: {error*1000:.2f} mm")
                print(f"  Recovered angles:")
                for joint in ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll']:
                    orig = original_config[joint]
                    recov = recovered_config[joint]
                    diff = abs(orig - recov)
                    print(f"    {joint:15s}: orig={orig:7.2f}°, recovered={recov:7.2f}°, diff={diff:6.2f}°")
                
                if success:
                    print("  ✓ IK succeeded! Using recovered configuration.")
                    node.set_target_configuration(recovered_config, target_pos)
                else:
                    print("  ⚠ IK didn't fully converge. Using original configuration instead.")
                    node.set_target_configuration(original_config, target_pos)
            
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