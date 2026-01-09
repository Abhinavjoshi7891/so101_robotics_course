# SO-101 Robotics Labs - Quick Reference Guide

## Project Structure

```
so101_robotics_course/
├── README.md                      # Main documentation
├── requirements.txt               # Python dependencies
├── setup.py                       # Package installation
│
├── src/                           # Core modules
│   ├── so101_forward_kinematics.py    # FK implementation
│   ├── so101_inverse_kinematics.py    # IK implementation  
│   ├── so101_mujoco_utils.py          # MuJoCo helpers
│   └── so101_ros2_utils.py            # ROS2 helpers
│
├── labs/                          # Lab exercises
│   ├── lab1_1_visualize_urdf.py
│   ├── lab1_2_test_fk_mujoco.py
│   ├── lab1_2_test_fk_rviz.py
│   ├── lab2_1_test_ik_mujoco.py
│   ├── lab2_2_lerobot_kinematics.py
│   ├── lab3_pick_place_mujoco.py
│   └── lab3_pick_place_gazebo.py
│
├── model/                         # MuJoCo models
│   └── scene.xml
│
└── config/                        # Configuration files
    └── so101_params.yaml
```

## Quick Start Commands

### MuJoCo Labs (No ROS2 Required)

```bash
# Setup
cd so101_robotics_course
pip install -r requirements.txt
pip install mujoco

# Lab 1.2: Test Forward Kinematics
python labs/lab1_2_test_fk_mujoco.py
python labs/lab1_2_test_fk_mujoco.py --interactive  # Interactive mode

# Lab 2.1: Test Inverse Kinematics
python labs/lab2_1_test_ik_mujoco.py
python labs/lab2_1_test_ik_mujoco.py --multi  # Multiple targets

# Lab 3: Pick and Place
python labs/lab3_pick_place_mujoco.py
python labs/lab3_pick_place_mujoco.py --continuous  # Loop mode
```

### ROS2/RViz Labs

```bash
# Setup (one time)
cd so101_robotics_course/ros2_ws
colcon build
source install/setup.bash

# Lab 1.1: Visualize URDF
ros2 launch lerobot_description so101_display.launch.py

# Lab 1.2: FK with RViz (requires separate terminal)
python labs/lab1_2_test_fk_rviz.py

# Lab 3: Pick and Place with Gazebo
# Terminal 1:
ros2 launch lerobot_description so101_gazebo.launch.py
# Terminal 2:
ros2 launch lerobot_controller so101_controller.launch.py
# Terminal 3 (optional MoveIt):
ros2 launch lerobot_moveit so101_moveit.launch.py
# Terminal 4:
python labs/lab3_pick_place_gazebo.py
```

## Lab Summaries

### Lab 1.1: Visualize URDF
**Objective**: Understand robot structure through URDF visualization
- Launch RViz with robot model
- Explore joint state publisher
- Observe TF frames
- Move joints interactively

### Lab 1.2: Forward Kinematics
**Objective**: Implement and verify FK computation
- Compute tool position from joint angles
- Visualize FK result with cylinder marker
- Compare multiple configurations
- Understand coordinate transformations

**Key Code**:
```python
from so101_forward_kinematics import get_forward_kinematics

joint_angles = {
    'shoulder_pan': -45.0,
    'shoulder_lift': 45.0,
    'elbow_flex': -45.0,
    'wrist_flex': 90.0,
    'wrist_roll': 0.0,
    'gripper': 50.0
}

position, rotation = get_forward_kinematics(joint_angles)
```

### Lab 2.1: Inverse Kinematics
**Objective**: Solve IK to reach target positions
- Numerical IK using Jacobian
- Test with various target positions
- Verify solutions with FK
- Handle workspace limits

**Key Code**:
```python
from so101_inverse_kinematics import inverse_kinematics_numerical

target = np.array([0.2, 0.1, 0.15])
joint_config, success, error = inverse_kinematics_numerical(target)
```

### Lab 2.2: LeRobot Kinematics
**Objective**: Compare hand-coded vs library kinematics
- Understand when to use each approach
- Learn about Placo library
- Integration with LeRobot ecosystem

### Lab 3: Pick and Place
**Objective**: Complete manipulation task
- Define waypoint sequence
- Implement smooth trajectories
- Coordinate gripper with arm motion
- Run in MuJoCo or Gazebo

**Waypoint Sequence**:
1. Home → Above Pick
2. Lower to Pick
3. Close Gripper
4. Lift Object
5. Transit to Place
6. Lower to Place
7. Open Gripper
8. Retreat to Home

## SO-101 Kinematic Parameters

| Parameter | Value (m) | Description |
|-----------|-----------|-------------|
| Base X | 0.0388 | X offset to joint 1 |
| Base Z | 0.0624 | Z offset to joint 1 |
| Upper arm | 0.1126 | Joint 2 to 3 (Z) |
| Forearm | 0.1349 | Joint 3 to 4 (X) |
| Wrist | 0.0611 | Joint 4 to 5 (X) |
| Gripper | 0.1034 | Joint 5 to tool (X) |
| Max reach | ~0.45 | Extended arm |

## Joint Axes

| Joint | Axis | Range |
|-------|------|-------|
| shoulder_pan | Z | ±180° |
| shoulder_lift | Y | ±90° |
| elbow_flex | Y | ±135° |
| wrist_flex | Y | ±90° |
| wrist_roll | X | ±180° |

## Common Issues

### MuJoCo not rendering
```bash
sudo apt install libglfw3 libglfw3-dev
```

### ROS2 packages not found
```bash
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
```

### FK result seems wrong
- Check joint angle units (degrees vs radians)
- Verify frame conventions
- Use visualization to debug

### IK doesn't converge
- Check if target is in workspace
- Try different initial guess
- Increase max iterations
- Use damped least squares

## References

- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [ROS2 Jazzy](https://docs.ros.org/en/jazzy/)
- [MoveIt2 Tutorials](https://moveit.picknik.ai/main/)
