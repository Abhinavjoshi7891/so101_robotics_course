"""
SO-101 Robotics Course - Core Modules
======================================

This package contains the core modules for the SO-101 robotics course:

- so101_forward_kinematics: Forward kinematics implementation
- so101_inverse_kinematics: Inverse kinematics (geometric and numerical)
- so101_mujoco_utils: MuJoCo simulation utilities
- so101_ros2_utils: ROS2 integration utilities
"""

from .so101_forward_kinematics import (
    get_forward_kinematics,
    get_full_transform,
    get_intermediate_transforms,
    Rx, Ry, Rz,
    SO101Params
)

from .so101_inverse_kinematics import (
    get_inverse_kinematics,
    inverse_kinematics_numerical,
    inverse_kinematics_geometric,
    compute_jacobian,
    compute_position_jacobian
)

from .so101_mujoco_utils import (
    set_initial_pose,
    send_position_command,
    get_current_pose,
    hold_position,
    move_to_pose,
    move_trajectory,
    show_cylinder,
    show_cube,
    show_sphere,
    show_frame,
    clear_visualizations,
    degrees_to_mujoco,
    mujoco_to_degrees
)

__version__ = "1.0.0"
__all__ = [
    # FK
    'get_forward_kinematics',
    'get_full_transform', 
    'get_intermediate_transforms',
    'Rx', 'Ry', 'Rz',
    'SO101Params',
    # IK
    'get_inverse_kinematics',
    'inverse_kinematics_numerical',
    'inverse_kinematics_geometric',
    'compute_jacobian',
    'compute_position_jacobian',
    # MuJoCo utils
    'set_initial_pose',
    'send_position_command',
    'get_current_pose',
    'hold_position',
    'move_to_pose',
    'move_trajectory',
    'show_cylinder',
    'show_cube',
    'show_sphere',
    'show_frame',
    'clear_visualizations',
    'degrees_to_mujoco',
    'mujoco_to_degrees',
]
