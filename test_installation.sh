#!/bin/bash
# Test the installation
# Usage: ./test_installation.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/activate.sh"

echo ""
echo "========================================"
echo "Testing SO-101 Installation"
echo "========================================"
echo ""

# Test Python packages
echo "1. Testing Python packages..."
python3 -c "
import sys
print(f'   Python: {sys.version}')

import numpy as np
print(f'   NumPy: {np.__version__}')

import scipy
print(f'   SciPy: {scipy.__version__}')

try:
    import mujoco
    print(f'   MuJoCo: {mujoco.__version__}')
except ImportError:
    print('   MuJoCo: NOT INSTALLED')
"

# Test our modules
echo ""
echo "2. Testing SO-101 modules..."
python3 -c "
from so101_forward_kinematics import get_forward_kinematics
from so101_inverse_kinematics import inverse_kinematics_numerical
import numpy as np

# Test FK
config = {'shoulder_pan': 0, 'shoulder_lift': 0, 'elbow_flex': 0, 
          'wrist_flex': 0, 'wrist_roll': 0, 'gripper': 0}
pos, rot = get_forward_kinematics(config)
print(f'   FK Test: Position = [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]')

# Test IK
target = np.array([0.2, 0.1, 0.15])
solution, success, error = inverse_kinematics_numerical(target, verbose=False)
print(f'   IK Test: Success = {success}, Error = {error:.6f}')
"

# Test ROS2
echo ""
echo "3. Testing ROS2..."
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
    echo "   ROS2 Humble: AVAILABLE"
    ros2 --version 2>/dev/null || echo "   (ros2 command not working)"
    
    if [ -f "$SCRIPT_DIR/ros2_ws/install/setup.bash" ]; then
        echo "   ROS2 Workspace: BUILT"
    else
        echo "   ROS2 Workspace: NOT BUILT (run colcon build)"
    fi
else
    echo "   ROS2 Humble: NOT INSTALLED"
fi

echo ""
echo "========================================"
echo "Installation test complete!"
echo "========================================"
