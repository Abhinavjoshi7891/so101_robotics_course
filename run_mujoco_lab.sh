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
