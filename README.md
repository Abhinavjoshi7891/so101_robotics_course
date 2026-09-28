# SO-101 Robotics Course Labs
## Lectures 1-3: Classical Robotics Foundations

This project contains hands-on labs for learning classical robotics concepts using the SO-101 arm.

## Project Structure

```
so101_robotics_course/
├── README.md
├── requirements.txt
├── setup.py
│
├── src/                          # Core Python modules
│   ├── __init__.py
│   ├── so101_forward_kinematics.py    # Lab 1.2: FK implementation
│   ├── so101_inverse_kinematics.py    # Lab 2.1: IK implementation
│   ├── so101_jacobian.py              # Jacobian computation
│   ├── so101_mujoco_utils.py          # MuJoCo helper functions
│   └── so101_ros2_utils.py            # ROS2 helper functions
│
├── labs/                         # Lab exercise scripts
│   ├── lab1_1_visualize_urdf.py       # Lab 1.1: URDF visualization
│   ├── lab1_2_test_fk_mujoco.py       # Lab 1.2: Test FK in MuJoCo
│   ├── lab1_2_test_fk_rviz.py         # Lab 1.2: Test FK in RViz
│   ├── lab2_1_test_ik_mujoco.py       # Lab 2.1: Test IK in MuJoCo
│   ├── lab2_1_test_ik_rviz.py         # Lab 2.1: Test IK in RViz
│   ├── lab2_2_lerobot_kinematics.py   # Lab 2.2: LeRobot kinematics
│   ├── lab3_pick_place_mujoco.py      # Lab 3: Pick and place in MuJoCo
│   └── lab3_pick_place_gazebo.py      # Lab 3: Pick and place in Gazebo
│
├── model/                        # MuJoCo model files
│   └── scene.xml                 # SO-101 MuJoCo scene
│
├── config/                       # Configuration files
│   └── so101_params.yaml         # Robot parameters
│
└── ros2_ws/                      # ROS2 workspace (link to lerobot_ws)
    └── src/
```

## Prerequisites

### System Requirements
- Ubuntu 22.04 LTS (with ROS2 Humble) or Ubuntu 24.04 LTS (with ROS2 Jazzy)
- Python 3.10+
- NVIDIA GPU (recommended for MuJoCo rendering)

### Quick Installation (Recommended)

```bash
# 1. Clone this repository
git clone <your-repo-url> so101_robotics_course
cd so101_robotics_course

# 2. Run the setup script
chmod +x setup.sh
./setup.sh

# 3. Activate the environment
source activate.sh

# 4. Test the installation
./test_installation.sh
```

### Manual Installation

```bash
# 1. Clone this repository
git clone <your-repo-url> so101_robotics_course
cd so101_robotics_course

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. For ROS2 labs, clone the lerobot_ws
cd ros2_ws/src
git clone https://github.com/Pavankv92/lerobot_ws.git .
cd ../..

# 5. Build ROS2 workspace
source /opt/ros/humble/setup.bash  # or /opt/ros/jazzy/setup.bash for 24.04
cd ros2_ws
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

## SO-101 Arm Parameters (from URDF)

The SO-101 is a 5-DoF + gripper robot arm with the following kinematic chain:

| Joint | Name | Type | Axis | Link Length (m) |
|-------|------|------|------|-----------------|
| Base→1 | - | Fixed | - | x=0.0388, z=0.0624 |
| 1 | shoulder_pan | Revolute | Z | z=-0.0304 |
| 1→2 | - | Fixed | - | z=-0.0542 |
| 2 | shoulder_lift | Revolute | Y | z=0.11257 |
| 2→3 | - | Fixed | - | z=0.028 |
| 3 | elbow_flex | Revolute | Y | x=0.1349 |
| 4 | wrist_flex | Revolute | Y | x=0.0611 |
| 5 | wrist_roll | Revolute | X | x=0.1034 |
| 6 | gripper | Revolute | - | (end-effector) |

## Lab Exercises

### Lecture 1: Forward Kinematics

**Lab 1.1: Visualize SO-101 URDF**
```bash
# ROS2/RViz
source ros2_ws/install/setup.bash
ros2 launch lerobot_description so101_display.launch.py
```

**Lab 1.2: Test Forward Kinematics**
```bash
# MuJoCo
python3 labs/lab1_2_test_fk_mujoco.py

# RViz (requires ROS2)
python3 labs/lab1_2_test_fk_rviz.py
```

### Lecture 2: Inverse Kinematics

**Lab 2.1: Test Inverse Kinematics**
```bash
# MuJoCo
python3 labs/lab2_1_test_ik_mujoco.py

# RViz
python3 labs/lab2_1_test_ik_rviz.py
```

**Lab 2.2: Using LeRobot's Built-in Kinematics**
```bash
pip install lerobot[kinematics]
python3 labs/lab2_2_lerobot_kinematics.py
```

### Lecture 3: Simulation & Pick-and-Place

**Lab 3: Full Pipeline**
```bash
# MuJoCo pick and place
python3 labs/lab3_pick_place_mujoco.py

# Gazebo + MoveIt pipeline
# Terminal 1
ros2 launch lerobot_description so101_gazebo.launch.py
# Terminal 2
ros2 launch lerobot_controller so101_controller.launch.py
# Terminal 3
ros2 launch lerobot_moveit so101_moveit.launch.py
# Terminal 4
python3 labs/lab3_pick_place_gazebo.py
```

## Hugging Face leLab GUI

`leLab` is Hugging Face's web-based interface for LeRobot (used for robot data collection, visualization, and teleoperation).

### Launching the GUI

Depending on your workflow, you can launch `leLab` using any of the following methods:

#### Option 1: Direct Execution (Fastest)
Run the executable directly from the local virtual environment without manual activation:
```bash
~/Data_Drive/lelab/.venv/bin/lelab
```

#### Option 2: Activate Environment
Activate the local virtual environment on your data drive and start the server:
```bash
source ~/Data_Drive/lelab/.venv/bin/activate
lelab
```

#### Option 3: Terminal Alias (Recommended)
Add a persistent alias to `~/.bashrc` to run `lelab` from any directory:
```bash
echo 'alias lelab="~/Data_Drive/lelab/.venv/bin/lelab"' >> ~/.bashrc
source ~/.bashrc

# Now launch anytime with:
lelab
```

### Hugging Face Authentication & Troubleshooting

To upload datasets or access Hub resources, authenticate your CLI:
```bash
~/Data_Drive/lelab/.venv/bin/hf auth login
```

**Troubleshooting GUI Login Status ("User is already logged in"):**
If the GUI indicates you are not logged in despite `hf auth login` succeeding:
1. Click **"I've logged in — recheck"** in the GUI.
2. Force re-authentication via CLI:
   ```bash
   ~/Data_Drive/lelab/.venv/bin/hf auth login --force
   ```
3. Alternatively, paste your Hugging Face write-access token directly into the token input field in the GUI.

## Learning Objectives

By completing these labs, you will:
1. Understand homogeneous transformation matrices
2. Implement forward kinematics from scratch
3. Derive and implement inverse kinematics (geometric + numerical)
4. Use simulation environments (RViz, Gazebo, MuJoCo)
5. Integrate motion planning with MoveIt
6. Execute pick-and-place tasks in simulation

## Troubleshooting

### Common Issues

1. **MuJoCo not rendering**: Ensure you have `libglfw3` installed
   ```bash
   sudo apt install libglfw3 libglfw3-dev
   ```

2. **ROS2 packages not found**: Source the workspace
   ```bash
   # Ubuntu 22.04 with ROS2 Humble
   source /opt/ros/humble/setup.bash
   source ros2_ws/install/setup.bash
   
   # Ubuntu 24.04 with ROS2 Jazzy
   source /opt/ros/jazzy/setup.bash
   source ros2_ws/install/setup.bash
   
   # Or just use the convenience script:
   source activate.sh
   ```

3. **Gazebo crashes**: Check GPU drivers and Gazebo version compatibility

4. **Virtual environment issues**: Make sure to activate before running labs
   ```bash
   source activate.sh
   ```

## References

- [SO-ARM100 GitHub](https://github.com/TheRobotStudio/SO-ARM100)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MoveIt2 Tutorials](https://moveit.picknik.ai/main/)
