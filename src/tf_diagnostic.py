#!/usr/bin/env python3
"""
FK-URDF Diagnostic Tool
=======================
This script helps identify frame mismatches between FK calculations and URDF.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformListener, Buffer, TransformException
import numpy as np
import time


class TFDiagnostic(Node):
    def __init__(self):
        super().__init__('tf_diagnostic')
        
        # Setup TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Wait for TF to populate
        self.get_logger().info("Waiting for TF tree to populate...")
        time.sleep(2.0)
        
        # Check available frames
        self.check_available_frames()
        
        # Check specific transforms
        self.check_key_transforms()
        
    def check_available_frames(self):
        """List all available TF frames"""
        print("\n" + "=" * 70)
        print("AVAILABLE TF FRAMES")
        print("=" * 70)
        
        # Get all frames
        frames = self.tf_buffer.all_frames_as_string()
        print(frames)
        
    def check_key_transforms(self):
        """Check transforms for key frames"""
        print("\n" + "=" * 70)
        print("KEY FRAME TRANSFORMS (from 'base' or 'base_link')")
        print("=" * 70)
        
        # Try different possible base frames
        possible_base_frames = ['base', 'base_link', 'world', 'base_footprint']
        
        # Try different possible target frames
        possible_target_frames = [
            'gripper', 'gripper_link', 'gripperframe', 
            'tool', 'tool_link', 'end_effector',
            '5', 'wrist_roll', 'wrist_roll_link',
            '6', 'gripper_finger'
        ]
        
        for base_frame in possible_base_frames:
            print(f"\nFrom base frame: '{base_frame}'")
            print("-" * 70)
            
            for target_frame in possible_target_frames:
                try:
                    t = self.tf_buffer.lookup_transform(
                        base_frame,
                        target_frame,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=1.0)
                    )
                    
                    pos = [
                        t.transform.translation.x,
                        t.transform.translation.y,
                        t.transform.translation.z
                    ]
                    
                    quat = [
                        t.transform.rotation.x,
                        t.transform.rotation.y,
                        t.transform.rotation.z,
                        t.transform.rotation.w
                    ]
                    
                    print(f"  {target_frame:20s}: pos=[{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
                    
                except TransformException as e:
                    pass  # Frame doesn't exist, skip it
                    
        # Check joint-to-joint transforms
        print("\n" + "=" * 70)
        print("JOINT FRAME TRANSFORMS (if available)")
        print("=" * 70)
        
        joint_frames = ['1', '2', '3', '4', '5', '6']
        base_frame = 'base'
        
        print(f"\nFrom base frame: '{base_frame}' to joint frames:")
        print("-" * 70)
        for joint_frame in joint_frames:
            try:
                t = self.tf_buffer.lookup_transform(
                    base_frame,
                    joint_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                
                pos = [
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z
                ]
                
                print(f"  Joint {joint_frame:3s}: pos=[{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")
                
            except TransformException:
                pass


def main():
    rclpy.init()
    node = TFDiagnostic()
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)
    print("\nUse this information to:")
    print("1. Verify which base frame RViz is using")
    print("2. Identify the correct target frame name")
    print("3. Check if there's a coordinate system rotation")
    
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()