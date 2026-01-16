#!/usr/bin/env python3
"""
Lab 2.2: Test GEOMETRIC IK in MuJoCo (Assignment Version)
========================================================

Tests the FINAL analytic IK solver that follows the
ECE4560 assignment geometry exactly.

Author: SO-101 Robotics Course (modified)
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("Error: MuJoCo not installed. Install with: pip install mujoco")
    sys.exit(1)

from so101_mujoco_utils import (
    set_initial_pose,
    send_position_command
)

# ✅ IMPORT FINAL INTEGRATED IK
from so101_inverse_kinematics_geometric_mujoco import get_inverse_kinematics


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def get_mujoco_gripper_position(model, data):
    """Get the actual gripper position from MuJoCo simulation."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'gripperframe')
    if site_id >= 0:
        return data.site_xpos[site_id].copy()
    return None


def show_target_marker(viewer, position, color=[1, 0, 0, 0.5]):
    """Show a semi-transparent cube at the target position."""
    if viewer.user_scn.ngeom >= 5:
        return
    
    idx = viewer.user_scn.ngeom
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[idx],
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.015, 0.015, 0.015],
        pos=np.array(position, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array(color, dtype=np.float32)
    )
    viewer.user_scn.ngeom += 1


def interpolate_configs(start, target, alpha):
    """Cubic interpolation between joint configs."""
    return {
        k: start[k] + alpha * (target[k] - start[k])
        for k in start.keys()
    }


# ------------------------------------------------------------
# Main Test
# ------------------------------------------------------------

def main():
    print("=" * 80)
    print("Lab 2.2: Geometric IK Test (MuJoCo, Assignment Model)")
    print("=" * 80)

    model_path = os.path.join(
        os.path.dirname(__file__), '..', 'model', 'scene.xml'
    )
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    # Test targets
    targets = [
        {'name': 'Table Center', 'pos': [0.20,  0.00, 0.02]},
        {'name': 'Table Left',   'pos': [0.20,  0.15, 0.02]},
        {'name': 'Table Right',  'pos': [0.20, -0.15, 0.02]},
        {'name': 'Close Pick',   'pos': [0.15,  0.00, 0.02]},
        {'name': 'Far Pick',     'pos': [0.30,  0.00, 0.02]},
    ]

    # Initial pose
    current_config = {
        'shoulder_pan': 0.0,
        'shoulder_lift': 0.0,
        'elbow_flex': 0.0,
        'wrist_flex': 0.0,
        'wrist_roll': 0.0,
        'gripper': 50.0
    }

    set_initial_pose(data, current_config, use_degrees=True)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nVisualization started (RED = target)\n")

        for i, target in enumerate(targets):
            if not viewer.is_running():
                break

            target_pos = np.array(target['pos'])
            print(f"[{i+1}/{len(targets)}] {target['name']} → {target_pos}")

            # ------------------------------------------------
            # SOLVE IK (ASSIGNMENT)
            # ------------------------------------------------
            try:
                ik_config = get_inverse_kinematics(
                    target_pos,
                    target_yaw=0.0   # vertical grasp
                )
                success = True
            except Exception as e:
                print(f"  ✗ IK failed: {e}")
                ik_config = current_config
                success = False

            color = [0, 1, 0, 0.5] if success else [1, 0, 0, 0.5]

            # ------------------------------------------------
            # MOVE ROBOT
            # ------------------------------------------------
            duration = 2.0
            start_time = time.time()

            while viewer.is_running() and (time.time() - start_time) < duration:
                t = (time.time() - start_time) / duration
                t = min(1.0, t)
                s = 3*t**2 - 2*t**3

                interp = interpolate_configs(current_config, ik_config, s)
                send_position_command(data, interp, use_degrees=True)
                mujoco.mj_step(model, data)

                viewer.user_scn.ngeom = 0
                show_target_marker(viewer, target_pos, color)

                curr_pos = get_mujoco_gripper_position(model, data)
                if curr_pos is not None:
                    err = np.linalg.norm(curr_pos - target_pos) * 1000
                    print(f"\r  Moving... error = {err:6.1f} mm", end="")

                viewer.sync()
                time.sleep(0.01)

            print(f"\r  Final error: {err:6.1f} mm          ")
            current_config = ik_config
            time.sleep(0.5)

    print("\nTest complete.")


if __name__ == "__main__":
    main()
