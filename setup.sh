#!/bin/bash
#===============================================================================
# SO-101 Robotics Course Setup Script
# ====================================
# 
# This script sets up the development environment for the SO-101 robotics labs.
#
# Target System:
#   - Ubuntu 22.04 LTS
#   - ROS2 Humble
#   - Python 3.10+
#
# What this script does:
#   1. Checks system requirements
#   2. Installs system dependencies
#   3. Creates Python virtual environment
#   4. Installs Python packages (numpy, scipy, mujoco, etc.)
#   5. Sets up ROS2 workspace with lerobot packages
#   6. Builds ROS2 packages
#   7. Creates convenience scripts
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh
#
#===============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$SCRIPT_DIR"

echo -e "${BLUE}"
echo "============================================================"
echo "  SO-101 Robotics Course Setup"
echo "============================================================"
echo -e "${NC}"

#===============================================================================
# Helper Functions
#===============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_command() {
    if command -v "$1" &> /dev/null; then
        return 0
    else
        return 1
    fi
}

#===============================================================================
# Step 1: Check System Requirements
#===============================================================================

echo -e "\n${BLUE}[Step 1/7] Checking system requirements...${NC}\n"

# Check Ubuntu version
if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [[ "$ID" == "ubuntu" ]]; then
        log_info "Detected: $PRETTY_NAME"
        if [[ "$VERSION_ID" != "22.04" ]]; then
            log_warn "This script is designed for Ubuntu 22.04. You have $VERSION_ID."
            log_warn "Proceeding anyway, but some things may not work."
        fi
    else
        log_warn "This script is designed for Ubuntu. You have $ID."
    fi
else
    log_warn "Could not detect OS version."
fi

# Check Python
if check_command python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"
else
    log_error "Python3 not found. Please install Python 3.10+."
    exit 1
fi

# Check ROS2
if [ -f /opt/ros/humble/setup.bash ]; then
    log_info "ROS2 Humble found at /opt/ros/humble"
    ROS2_AVAILABLE=true
else
    log_warn "ROS2 Humble not found at /opt/ros/humble"
    log_warn "ROS2-based labs will not work. MuJoCo labs will still work."
    ROS2_AVAILABLE=false
fi

# Check git
if check_command git; then
    log_info "Git is installed"
else
    log_error "Git not found. Installing..."
    sudo apt-get update && sudo apt-get install -y git
fi

#===============================================================================
# Step 2: Install System Dependencies
#===============================================================================

echo -e "\n${BLUE}[Step 2/7] Installing system dependencies...${NC}\n"

log_info "Updating package lists..."
sudo apt-get update

log_info "Installing required system packages..."
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential \
    cmake \
    libglfw3 \
    libglfw3-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libosmesa6-dev \
    patchelf \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev

log_info "System dependencies installed."

#===============================================================================
# Step 3: Create Python Virtual Environment
#===============================================================================

echo -e "\n${BLUE}[Step 3/7] Setting up Python virtual environment...${NC}\n"

VENV_DIR="$PROJECT_DIR/venv"

if [ -d "$VENV_DIR" ]; then
    log_warn "Virtual environment already exists at $VENV_DIR"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Removing existing virtual environment..."
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        log_info "Virtual environment recreated."
    fi
else
    log_info "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    log_info "Virtual environment created."
fi

# Activate virtual environment
log_info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip

#===============================================================================
# Step 4: Install Python Dependencies
#===============================================================================

echo -e "\n${BLUE}[Step 4/7] Installing Python dependencies...${NC}\n"

log_info "Installing core packages..."
pip install numpy scipy matplotlib

log_info "Installing MuJoCo..."
pip install mujoco

log_info "Installing additional packages..."
pip install \
    opencv-python \
    pyyaml \
    tqdm \
    imageio

# Install the project in editable mode
log_info "Installing so101_robotics_course package..."
pip install -e "$PROJECT_DIR"

log_info "Python dependencies installed."

# Test MuJoCo installation
log_info "Testing MuJoCo installation..."
python3 -c "import mujoco; print(f'MuJoCo version: {mujoco.__version__}')" && \
    log_info "MuJoCo installed successfully!" || \
    log_warn "MuJoCo import failed. You may need to set LD_LIBRARY_PATH."

#===============================================================================
# Step 5: Setup ROS2 Workspace
#===============================================================================

echo -e "\n${BLUE}[Step 5/7] Setting up ROS2 workspace...${NC}\n"

ROS2_WS="$PROJECT_DIR/ros2_ws"

if [ "$ROS2_AVAILABLE" = true ]; then
    # Source ROS2
    source /opt/ros/humble/setup.bash
    
    # Create workspace if it doesn't exist
    if [ ! -d "$ROS2_WS/src" ]; then
        log_info "Creating ROS2 workspace at $ROS2_WS..."
        mkdir -p "$ROS2_WS/src"
    fi
    
    # Clone lerobot_ws if not present
    LEROBOT_WS_DIR="$ROS2_WS/src/lerobot_ws"
    
    if [ -d "$LEROBOT_WS_DIR" ] && [ "$(ls -A $LEROBOT_WS_DIR 2>/dev/null)" ]; then
        log_info "lerobot_ws already exists. Pulling latest changes..."
        cd "$LEROBOT_WS_DIR"
        git pull origin main || log_warn "Could not pull updates"
        cd "$PROJECT_DIR"
    else
        log_info "Cloning lerobot_ws repository..."
        # Remove empty directory if exists
        rm -rf "$LEROBOT_WS_DIR"
        
        # Clone the repository
        git clone https://github.com/Pavankv92/lerobot_ws.git "$LEROBOT_WS_DIR" || {
            log_warn "Could not clone lerobot_ws. ROS2 labs will have limited functionality."
            log_warn "You can manually clone later: git clone https://github.com/Pavankv92/lerobot_ws.git $LEROBOT_WS_DIR"
        }
    fi
    
    log_info "ROS2 workspace setup complete."
else
    log_warn "Skipping ROS2 workspace setup (ROS2 not available)."
fi

#===============================================================================
# Step 6: Build ROS2 Packages
#===============================================================================

echo -e "\n${BLUE}[Step 6/7] Building ROS2 packages...${NC}\n"

if [ "$ROS2_AVAILABLE" = true ] && [ -d "$ROS2_WS/src" ]; then
    cd "$ROS2_WS"
    
    # Check if there are packages to build
    if [ -d "$ROS2_WS/src/lerobot_ws" ] || [ -n "$(ls -A $ROS2_WS/src 2>/dev/null)" ]; then
        log_info "Installing ROS2 dependencies with rosdep..."
        
        # Initialize rosdep if needed
        if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
            sudo rosdep init || log_warn "rosdep already initialized"
        fi
        rosdep update || log_warn "rosdep update had warnings"
        
        # Install dependencies
        rosdep install --from-paths src --ignore-src -r -y || \
            log_warn "Some rosdep dependencies could not be installed"
        
        log_info "Building ROS2 workspace..."
        colcon build --symlink-install || {
            log_warn "colcon build failed. Some ROS2 labs may not work."
            log_warn "Check the error messages above and try building manually."
        }
        
        log_info "ROS2 packages built."
    else
        log_warn "No packages found in $ROS2_WS/src to build."
    fi
    
    cd "$PROJECT_DIR"
else
    log_warn "Skipping ROS2 build (ROS2 not available or workspace not setup)."
fi

#===============================================================================
# Step 7: Create Convenience Scripts
#===============================================================================

echo -e "\n${BLUE}[Step 7/7] Creating convenience scripts...${NC}\n"

# Create activation script
ACTIVATE_SCRIPT="$PROJECT_DIR/activate.sh"
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Source this script to activate the SO-101 development environment
# Usage: source activate.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate Python virtual environment
if [ -f "$SCRIPT_DIR/venv/bin/activate" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "✓ Python virtual environment activated"
else
    echo "✗ Virtual environment not found. Run setup.sh first."
fi

# Source ROS2 Humble
if [ -f /opt/ros/humble/setup.bash ]; then
    source /opt/ros/humble/setup.bash
    echo "✓ ROS2 Humble sourced"
fi

# Source ROS2 workspace
if [ -f "$SCRIPT_DIR/ros2_ws/install/setup.bash" ]; then
    source "$SCRIPT_DIR/ros2_ws/install/setup.bash"
    echo "✓ ROS2 workspace sourced"
fi

# Add project to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR/src:$PYTHONPATH"

echo ""
echo "SO-101 Robotics Course environment activated!"
echo "Project directory: $SCRIPT_DIR"
echo ""
EOF
chmod +x "$ACTIVATE_SCRIPT"
log_info "Created: activate.sh"

# Create run script for MuJoCo labs
RUN_MUJOCO_SCRIPT="$PROJECT_DIR/run_mujoco_lab.sh"
cat > "$RUN_MUJOCO_SCRIPT" << 'EOF'
#!/bin/bash
# Run MuJoCo-based labs
# Usage: ./run_mujoco_lab.sh [lab_number]
#   ./run_mujoco_lab.sh 1.2    # Run Lab 1.2 FK
#   ./run_mujoco_lab.sh 2.1    # Run Lab 2.1 IK
#   ./run_mujoco_lab.sh 3      # Run Lab 3 Pick and Place

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/activate.sh"

case "$1" in
    "1.2")
        python "$SCRIPT_DIR/labs/lab1_2_test_fk_mujoco.py" "${@:2}"
        ;;
    "2.1")
        python "$SCRIPT_DIR/labs/lab2_1_test_ik_mujoco.py" "${@:2}"
        ;;
    "2.2")
        python "$SCRIPT_DIR/labs/lab2_2_lerobot_kinematics.py" "${@:2}"
        ;;
    "3")
        python "$SCRIPT_DIR/labs/lab3_pick_place_mujoco.py" "${@:2}"
        ;;
    *)
        echo "SO-101 MuJoCo Lab Runner"
        echo ""
        echo "Usage: $0 [lab_number] [options]"
        echo ""
        echo "Available labs:"
        echo "  1.2  - Forward Kinematics Test"
        echo "         Options: --interactive"
        echo "  2.1  - Inverse Kinematics Test"
        echo "         Options: --multi (multiple targets)"
        echo "  2.2  - LeRobot Kinematics Comparison"
        echo "  3    - Pick and Place"
        echo "         Options: --continuous (loop mode)"
        echo ""
        echo "Examples:"
        echo "  $0 1.2"
        echo "  $0 2.1 --multi"
        echo "  $0 3 --continuous"
        ;;
esac
EOF
chmod +x "$RUN_MUJOCO_SCRIPT"
log_info "Created: run_mujoco_lab.sh"

# Create test script
TEST_SCRIPT="$PROJECT_DIR/test_installation.sh"
cat > "$TEST_SCRIPT" << 'EOF'
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
EOF
chmod +x "$TEST_SCRIPT"
log_info "Created: test_installation.sh"

#===============================================================================
# Final Summary
#===============================================================================

echo -e "\n${BLUE}"
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo -e "${NC}"

echo -e "${GREEN}What was installed:${NC}"
echo "  ✓ Python virtual environment (./venv)"
echo "  ✓ Python packages (numpy, scipy, matplotlib, mujoco)"
echo "  ✓ SO-101 robotics modules"
if [ "$ROS2_AVAILABLE" = true ]; then
    echo "  ✓ ROS2 workspace (./ros2_ws)"
fi

echo ""
echo -e "${GREEN}Quick Start:${NC}"
echo ""
echo "  1. Activate the environment:"
echo "     ${YELLOW}source activate.sh${NC}"
echo ""
echo "  2. Test the installation:"
echo "     ${YELLOW}./test_installation.sh${NC}"
echo ""
echo "  3. Run a lab (MuJoCo):"
echo "     ${YELLOW}./run_mujoco_lab.sh 1.2${NC}    # FK test"
echo "     ${YELLOW}./run_mujoco_lab.sh 2.1${NC}    # IK test"
echo "     ${YELLOW}./run_mujoco_lab.sh 3${NC}      # Pick and place"
echo ""

if [ "$ROS2_AVAILABLE" = true ]; then
    echo "  4. Run ROS2 visualization:"
    echo "     ${YELLOW}source activate.sh${NC}"
    echo "     ${YELLOW}ros2 launch lerobot_description so101_display.launch.py${NC}"
    echo ""
fi

echo -e "${GREEN}Documentation:${NC}"
echo "  - README.md           - Full documentation"
echo "  - LABS_QUICKREF.md    - Quick reference for all labs"
echo ""

echo -e "${YELLOW}Note: Always run 'source activate.sh' before working on the labs!${NC}"
echo ""
