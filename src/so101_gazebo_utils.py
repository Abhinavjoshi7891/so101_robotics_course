import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from visualization_msgs.msg import Marker
import numpy as np

class SO101GazeboController(Node):
    """ROS2 interface for controlling SO-101 in Gazebo"""
    
    def __init__(self):
        super().__init__('so101_gazebo_controller')
        
        # Joint names matching URDF
        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint', 
            'elbow_flex_joint',
            'wrist_flex_joint',
            'wrist_roll_joint',
            'gripper_joint'
        ]
        
        # Current joint state
        self.current_joint_state = None
        
        # Subscriber for joint states
        self.joint_state_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        
        # Action client for trajectory execution
        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            '/arm_controller/follow_joint_trajectory'
        )
        
        # Publisher for markers (FK visualization)
        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', 10)
        
    def joint_state_callback(self, msg):
        """Store current joint positions"""
        self.current_joint_state = {
            name: pos for name, pos in zip(msg.name, msg.position)
        }
    
    def get_current_joint_angles(self):
        """Get current joint configuration as dict"""
        if self.current_joint_state is None:
            return None
        
        return {
            'shoulder_pan': np.rad2deg(self.current_joint_state['shoulder_pan_joint']),
            'shoulder_lift': np.rad2deg(self.current_joint_state['shoulder_lift_joint']),
            'elbow_flex': np.rad2deg(self.current_joint_state['elbow_flex_joint']),
            'wrist_flex': np.rad2deg(self.current_joint_state['wrist_flex_joint']),
            'wrist_roll': np.rad2deg(self.current_joint_state['wrist_roll_joint']),
            'gripper': np.rad2deg(self.current_joint_state['gripper_joint'])
        }
    
    def move_to_joint_positions(self, target_config, duration=2.0):
        """Execute trajectory to target joint configuration"""
        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory.joint_names = self.joint_names
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = [
            np.deg2rad(target_config['shoulder_pan']),
            np.deg2rad(target_config['shoulder_lift']),
            np.deg2rad(target_config['elbow_flex']),
            np.deg2rad(target_config['wrist_flex']),
            np.deg2rad(target_config['wrist_roll']),
            np.deg2rad(target_config['gripper'])
        ]
        point.time_from_start.sec = int(duration)
        point.time_from_start.nanosec = int((duration % 1) * 1e9)
        
        goal_msg.trajectory.points = [point]
        
        # Send goal and wait
        self.trajectory_client.wait_for_server()
        future = self.trajectory_client.send_goal_async(goal_msg)
        rclpy.spin_until_future_complete(self, future)
        
        goal_handle = future.result()
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        return result_future.result()
    
    def publish_cylinder_marker(self, position, rotation, radius=0.025, height=0.1):
        """Visualize FK result with cylinder marker in RViz"""
        marker = Marker()
        marker.header.frame_id = "base_link"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "fk_visualization"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Position
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        
        # Orientation from rotation matrix
        from scipy.spatial.transform import Rotation as R
        quat = R.from_matrix(rotation).as_quat()
        marker.pose.orientation.x = quat[0]
        marker.pose.orientation.y = quat[1]
        marker.pose.orientation.z = quat[2]
        marker.pose.orientation.w = quat[3]
        
        # Scale
        marker.scale.x = radius * 2
        marker.scale.y = radius * 2
        marker.scale.z = height
        
        # Color (red, semi-transparent)
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.7
        
        self.marker_pub.publish(marker)