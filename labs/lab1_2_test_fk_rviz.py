#!/usr/bin/env python3
"""
Lab 1.2 (RViz): Test Forward Kinematics with Preview Mode - FIXED
==================================================================

This script shows the target position FIRST (red marker), then moves
the robot SMOOTHLY to that position.

Timeline for each configuration:
- Seconds 0-2: Show RED MARKER at target position (robot stays in old position)
- Seconds 2-5: Move ROBOT smoothly to target (marker turns green, robot interpolates)
- Repeat with next configuration

Prerequisites:
    1. ros2 launch lerobot_description so101_display_no_jsp.launch.py

Usage:
    python3 labs/lab1_2_test_fk_rviz.py

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
    """Linearly interpolate between two joint configurations.
    
    Args:
        config_start: Starting configuration dict
        config_target: Target configuration dict
        alpha: Interpolation factor (0.0 = start, 1.0 = target)
    
    Returns:
        Interpolated configuration dict
    """
    result = {}
    for key in config_start.keys():
        start_val = config_start[key]
        target_val = config_target[key]
        result[key] = start_val + alpha * (target_val - start_val)
    return result


class FKVisualizationNode(Node):
    """ROS2 node for FK visualization with preview mode."""
    
    def __init__(self):
        super().__init__('fk_visualization')
        
        # FK Mode
        self.fk_mode = 'urdf_native'
        
        # SETUP PUBLISHERS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10
        )
        self.joint_pub = self.create_publisher(JointState, '/joint_states', qos_profile)
        self.marker_pub = self.create_publisher(Marker, '/fk_marker', 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        
        # SETUP TF LISTENER
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # JOINT MAPPING & CONFIG
        self.joint_mapping = {
            'shoulder_pan': '1', 'shoulder_lift': '2', 'elbow_flex': '3',
            'wrist_flex': '4', 'wrist_roll': '5', 'gripper': '6'
        }
        
        # Configurations for interpolation
        self.start_joint_config = {
            'shoulder_pan': 0.0, 'shoulder_lift': 0.0, 'elbow_flex': 0.0,
            'wrist_flex': 0.0, 'wrist_roll': 0.0, 'gripper': 50.0
        }
        self.target_joint_config = self.start_joint_config.copy()
        self.current_joint_config = self.start_joint_config.copy()
        
        # State machine for preview mode
        self.state = 'PREVIEW'  # 'PREVIEW' or 'MOVING'
        self.state_start_time = time.time()
        
        # Timing
        self.PREVIEW_DURATION = 2.0  # Show marker for 2 seconds
        self.MOVING_DURATION = 3.0   # Robot moves for 3 seconds
        
        self.timer = self.create_timer(0.05, self.timer_callback)
        self.get_logger().info(f'FK Debug Node started (mode={self.fk_mode}) with PREVIEW!')
        
        # Frame tracking
        self.target_frames = ['gripper', '5', 'link5', 'wrist_roll_link']
        self.active_target_frame = None
    
    def set_target_configuration(self, config):
        """Set the next target configuration"""
        # Current config becomes the start for interpolation
        self.start_joint_config = self.current_joint_config.copy()
        self.target_joint_config = config.copy()
        self.state = 'PREVIEW'
        self.state_start_time = time.time()
    
    def timer_callback(self):
        current_time = time.time()
        elapsed = current_time - self.state_start_time
        
        # State machine
        if self.state == 'PREVIEW':
            # Show marker at target, keep robot at start position
            self.current_joint_config = self.start_joint_config.copy()
            
            if elapsed >= self.PREVIEW_DURATION:
                self.state = 'MOVING'
                self.state_start_time = current_time
                self.get_logger().info('→ Robot moving to target...')
                
        elif self.state == 'MOVING':
            # Interpolate robot from start to target
            alpha = min(1.0, elapsed / self.MOVING_DURATION)
            self.current_joint_config = interpolate_configs(
                self.start_joint_config,
                self.target_joint_config,
                alpha
            )
        
        # --- 1. Publish Joint States (Robot Position) ---
        js_msg = JointState()
        js_msg.header.stamp = self.get_clock().now().to_msg()
        js_msg.header.frame_id = ''
        
        # Use current config for robot
        for name, value in self.current_joint_config.items():
            if name in self.joint_mapping:
                js_msg.name.append(self.joint_mapping[name])
                if name == 'gripper':
                    js_msg.position.append(value / 100.0 * 0.04)
                else:
                    js_msg.position.append(np.deg2rad(value))
        self.joint_pub.publish(js_msg)
        
        # --- 2. Compute FK for TARGET position (for marker) ---
        transforms_target = get_intermediate_transforms(self.target_joint_config, mode=self.fk_mode)
        fk_mat_target = transforms_target['joint5']
        fk_pos_target = fk_mat_target[0:3, 3]
        fk_rot_target = fk_mat_target[0:3, 0:3]
        
        # --- 3. Compute FK for CURRENT position (for error reporting) ---
        transforms_current = get_intermediate_transforms(self.current_joint_config, mode=self.fk_mode)
        fk_mat_current = transforms_current['joint5']
        fk_pos_current = fk_mat_current[0:3, 3]
        
        # --- 4. Lookup URDF Transform ---
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
        
        # --- 5. Log the Difference ---
        if urdf_pos is not None:
            error = fk_pos_current - urdf_pos
            dist_error = np.linalg.norm(error) * 1000.0
            
            if self.state == 'PREVIEW':
                state_str = "PREVIEW"
                progress = f"{elapsed:.1f}/{self.PREVIEW_DURATION:.1f}s"
            else:
                state_str = "MOVING "
                alpha = min(1.0, elapsed / self.MOVING_DURATION)
                progress = f"{int(alpha*100):3d}%"
            
            print(f"\r[{state_str} {progress}] FK: [{fk_pos_current[0]:.4f}, {fk_pos_current[1]:.4f}, {fk_pos_current[2]:.4f}] | "
                  f"URDF: [{urdf_pos[0]:.4f}, {urdf_pos[1]:.4f}, {urdf_pos[2]:.4f}] | "
                  f"Err: {dist_error:6.2f} mm", end="")
        
        # --- 6. Publish Visualization (Always show TARGET position) ---
        # Marker shows where robot SHOULD move to
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = 'base'
        marker.ns = 'fk_visualization'
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(fk_pos_target[0])
        marker.pose.position.y = float(fk_pos_target[1])
        marker.pose.position.z = float(fk_pos_target[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.04; marker.scale.y = 0.04; marker.scale.z = 0.04
        
        # Color: RED during preview, GREEN during moving
        if self.state == 'PREVIEW':
            marker.color.r = 1.0; marker.color.g = 0.0; marker.color.b = 0.0; marker.color.a = 0.9
        else:
            marker.color.r = 0.0; marker.color.g = 1.0; marker.color.b = 0.0; marker.color.a = 0.7
        
        self.marker_pub.publish(marker)
        
        # TF Frame for target
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'base'
        t.child_frame_id = 'fk_target_frame'
        t.transform.translation.x = float(fk_pos_target[0])
        t.transform.translation.y = float(fk_pos_target[1])
        t.transform.translation.z = float(fk_pos_target[2])
        quat = rotation_to_quaternion(fk_rot_target)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = quat
        self.tf_broadcaster.sendTransform(t)


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
        {'name': 'Up',    'config': {'shoulder_pan': 0.0, 'shoulder_lift': -30.0, 'elbow_flex': -60.0, 'wrist_flex': 90.0, 'wrist_roll': 0.0, 'gripper': 50.0}},
    ]

    try:
        config_idx = 0
        cycle_start_time = time.time()
        CYCLE_TIME = 5.0  # 2s preview + 3s moving
        
        print("\n" + "="*80)
        print(f"FK DEBUG MODE: {node.fk_mode.upper()} with PREVIEW")
        print("="*80)
        print("\nVisualization Guide:")
        print("  🔴 RED sphere   = Target position (where robot WILL move)")
        print("  🟢 GREEN sphere = Robot is moving to target")
        print("  🤖 Robot mesh   = Current robot position")
        print("\nWatch: RED sphere appears, then robot SMOOTHLY moves to overlap it!")
        print("="*80 + "\n")
        
        # Start with first config
        node.set_target_configuration(test_configs[config_idx]['config'])
        print(f"→ Target: {test_configs[config_idx]['name']}")
        
        while rclpy.ok():
            elapsed_cycle = time.time() - cycle_start_time
            
            # Switch to next configuration
            if elapsed_cycle >= CYCLE_TIME:
                print()  # Newline
                config_idx = (config_idx + 1) % len(test_configs)
                current = test_configs[config_idx]
                node.set_target_configuration(current['config'])
                print(f"→ Target: {current['name']}")
                cycle_start_time = time.time()
            
            rclpy.spin_once(node, timeout_sec=0.01)
            
    except KeyboardInterrupt:
        print("\n\nShutdown.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()