#!/usr/bin/env python3
"""
SO-101 Workspace Analyzer
==========================

Analyzes and visualizes the reachable workspace of the SO-101 robot arm.
Plots positions where IK succeeds (green) vs fails (red).

Usage:
    python3 analyze_workspace.py --mode mujoco
    python3 analyze_workspace.py --mode urdf_native
    python3 analyze_workspace.py --resolution 50  # Lower for faster analysis

Author: SO-101 Robotics Course
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from so101_inverse_kinematics_geometric import SO101GeometricIK


def analyze_workspace_2d(
    mode='mujoco',
    z_height=0.014,  # Height for pick-and-place (just above table)
    x_range=(0.0, 0.35),
    y_range=(-0.35, 0.35),
    resolution=100,
    elbow_up=True
):
    """
    Analyze 2D workspace at a fixed height.
    
    Args:
        mode: 'mujoco' or 'urdf_native'
        z_height: Fixed Z height to analyze
        x_range: (min, max) X range to test
        y_range: (min, max) Y range to test
        resolution: Number of points per axis
        elbow_up: Elbow configuration
    
    Returns:
        (X, Y, reachable) where reachable is 2D boolean array
    """
    print(f"\nAnalyzing workspace for mode={mode} at z={z_height}m...")
    print(f"X range: {x_range}, Y range: {y_range}")
    print(f"Resolution: {resolution}x{resolution} = {resolution**2} points")
    
    # Create IK solver
    solver = SO101GeometricIK(mode=mode)
    
    # Create grid
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    
    # Test each point
    reachable = np.zeros_like(X, dtype=bool)
    
    total_points = resolution * resolution
    for i in range(resolution):
        for j in range(resolution):
            target = np.array([X[i, j], Y[i, j], z_height])
            
            # Try IK
            config, success, msg = solver.get_inverse_kinematics(
                target,
                elbow_up=elbow_up,
                validate_workspace=False  # Don't use built-in validation
            )
            
            reachable[i, j] = success
        
        # Progress
        if (i + 1) % 10 == 0:
            progress = ((i + 1) * resolution) / total_points * 100
            print(f"Progress: {progress:.1f}%")
    
    reachable_count = np.sum(reachable)
    print(f"\nReachable points: {reachable_count}/{total_points} ({reachable_count/total_points*100:.1f}%)")
    
    return X, Y, reachable


def plot_workspace_2d(X, Y, reachable, mode='mujoco', z_height=0.014, save_path=None):
    """
    Plot 2D workspace visualization.
    
    Args:
        X, Y: Meshgrid coordinates
        reachable: Boolean array indicating reachability
        mode: Mode string for title
        z_height: Height for title
        save_path: Optional path to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot reachable (green) and unreachable (red) points
    colors = np.where(reachable, 'green', 'red')
    
    # Scatter plot
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            color = 'green' if reachable[i, j] else 'red'
            ax.plot(X[i, j], Y[i, j], 'o', color=color, markersize=3, alpha=0.6)
    
    # Labels and title
    ax.set_xlabel('X position (m)', fontsize=14)
    ax.set_ylabel('Y position (m)', fontsize=14)
    ax.set_title(f'End-Effector Workspace\n(mode={mode}, z={z_height}m)', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Reachable'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Unreachable')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nWorkspace plot saved to: {save_path}")
    
    plt.show()


def analyze_workspace_3d(
    mode='mujoco',
    x_range=(0.0, 0.35),
    y_range=(-0.35, 0.35),
    z_range=(0.01, 0.40),
    resolution=30,
    elbow_up=True
):
    """
    Analyze 3D workspace (slower, use lower resolution).
    
    Args:
        mode: 'mujoco' or 'urdf_native'
        x_range, y_range, z_range: Ranges to test
        resolution: Points per axis
        elbow_up: Elbow configuration
    
    Returns:
        (X, Y, Z, reachable) where reachable is 3D boolean array
    """
    print(f"\nAnalyzing 3D workspace for mode={mode}...")
    print(f"X: {x_range}, Y: {y_range}, Z: {z_range}")
    print(f"Resolution: {resolution}^3 = {resolution**3} points")
    print("This may take a while...")
    
    solver = SO101GeometricIK(mode=mode)
    
    # Create 3D grid
    x_vals = np.linspace(x_range[0], x_range[1], resolution)
    y_vals = np.linspace(y_range[0], y_range[1], resolution)
    z_vals = np.linspace(z_range[0], z_range[1], resolution)
    X, Y, Z = np.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
    
    # Test each point
    reachable = np.zeros_like(X, dtype=bool)
    
    total_points = resolution ** 3
    count = 0
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                target = np.array([X[i, j, k], Y[i, j, k], Z[i, j, k]])
                
                config, success, msg = solver.get_inverse_kinematics(
                    target,
                    elbow_up=elbow_up,
                    validate_workspace=False
                )
                
                reachable[i, j, k] = success
                count += 1
        
        progress = (i + 1) / resolution * 100
        print(f"Progress: {progress:.1f}%")
    
    reachable_count = np.sum(reachable)
    print(f"\nReachable points: {reachable_count}/{total_points} ({reachable_count/total_points*100:.1f}%)")
    
    return X, Y, Z, reachable


def plot_workspace_3d(X, Y, Z, reachable, mode='mujoco', save_path=None):
    """
    Plot 3D workspace visualization.
    
    Args:
        X, Y, Z: 3D meshgrid coordinates
        reachable: 3D boolean array
        mode: Mode string for title
        save_path: Optional save path
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract reachable points
    x_reach = X[reachable]
    y_reach = Y[reachable]
    z_reach = Z[reachable]
    
    # Extract unreachable points (subsample for visibility)
    unreachable_mask = ~reachable
    if np.sum(unreachable_mask) > 5000:
        # Subsample unreachable points
        indices = np.where(unreachable_mask)
        subsample = np.random.choice(len(indices[0]), 5000, replace=False)
        x_unreach = X[indices[0][subsample], indices[1][subsample], indices[2][subsample]]
        y_unreach = Y[indices[0][subsample], indices[1][subsample], indices[2][subsample]]
        z_unreach = Z[indices[0][subsample], indices[1][subsample], indices[2][subsample]]
    else:
        x_unreach = X[unreachable_mask]
        y_unreach = Y[unreachable_mask]
        z_unreach = Z[unreachable_mask]
    
    # Plot
    ax.scatter(x_reach, y_reach, z_reach, c='green', marker='o', s=1, alpha=0.3, label='Reachable')
    ax.scatter(x_unreach, y_unreach, z_unreach, c='red', marker='o', s=1, alpha=0.3, label='Unreachable')
    
    ax.set_xlabel('X position (m)')
    ax.set_ylabel('Y position (m)')
    ax.set_zlabel('Z position (m)')
    ax.set_title(f'3D Workspace (mode={mode})')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n3D workspace plot saved to: {save_path}")
    
    plt.show()


def compute_workspace_boundaries(X, Y, reachable):
    """
    Compute workspace boundary statistics.
    
    Args:
        X, Y: Meshgrid coordinates
        reachable: Boolean reachability array
    
    Returns:
        Dictionary of boundary statistics
    """
    if not np.any(reachable):
        return {
            'x_min': None, 'x_max': None,
            'y_min': None, 'y_max': None,
            'area': 0,
            'center': None
        }
    
    x_reach = X[reachable]
    y_reach = Y[reachable]
    
    # Compute boundaries
    x_min, x_max = np.min(x_reach), np.max(x_reach)
    y_min, y_max = np.min(y_reach), np.max(y_reach)
    
    # Approximate area (count of reachable points × grid cell area)
    dx = (X[0, 1] - X[0, 0]) if X.shape[1] > 1 else 0
    dy = (Y[1, 0] - Y[0, 0]) if Y.shape[0] > 1 else 0
    cell_area = dx * dy
    area = np.sum(reachable) * cell_area
    
    # Center
    center = (np.mean(x_reach), np.mean(y_reach))
    
    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'area': area,
        'center': center,
        'x_range': (x_min, x_max),
        'y_range': (y_min, y_max)
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze SO-101 robot workspace')
    parser.add_argument('--mode', type=str, default='mujoco', choices=['mujoco', 'urdf_native'],
                       help='Coordinate system mode')
    parser.add_argument('--z', type=float, default=0.014,
                       help='Z height for 2D analysis (meters)')
    parser.add_argument('--resolution', type=int, default=100,
                       help='Grid resolution (higher = more accurate but slower)')
    parser.add_argument('--3d', action='store_true', dest='three_d',
                       help='Perform 3D analysis (slow!)')
    parser.add_argument('--save', type=str, default=None,
                       help='Save plot to file')
    parser.add_argument('--elbow-down', action='store_true',
                       help='Use elbow-down configuration')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("SO-101 Workspace Analyzer")
    print("=" * 70)
    print(f"Mode: {args.mode}")
    print(f"Elbow: {'down' if args.elbow_down else 'up'}")
    
    if args.three_d:
        # 3D analysis
        X, Y, Z, reachable = analyze_workspace_3d(
            mode=args.mode,
            resolution=args.resolution,
            elbow_up=not args.elbow_down
        )
        
        save_path = args.save if args.save else f'workspace_3d_{args.mode}.png'
        plot_workspace_3d(X, Y, Z, reachable, mode=args.mode, save_path=save_path)
    else:
        # 2D analysis
        # Set ranges based on mode
        if args.mode == 'mujoco':
            x_range = (0.0, 0.35)
            y_range = (-0.35, 0.35)
        else:  # urdf_native
            x_range = (-0.25, 0.25)
            y_range = (-0.45, -0.05)
        
        X, Y, reachable = analyze_workspace_2d(
            mode=args.mode,
            z_height=args.z,
            x_range=x_range,
            y_range=y_range,
            resolution=args.resolution,
            elbow_up=not args.elbow_down
        )
        
        # Compute boundaries
        boundaries = compute_workspace_boundaries(X, Y, reachable)
        
        print("\n" + "=" * 70)
        print("Workspace Boundaries")
        print("=" * 70)
        if boundaries['x_min'] is not None:
            print(f"X range: [{boundaries['x_min']:.3f}, {boundaries['x_max']:.3f}] m")
            print(f"Y range: [{boundaries['y_min']:.3f}, {boundaries['y_max']:.3f}] m")
            print(f"Approximate area: {boundaries['area']:.4f} m²")
            print(f"Center: ({boundaries['center'][0]:.3f}, {boundaries['center'][1]:.3f})")
            print(f"\nRecommended target ranges for IK:")
            print(f"  x_pos_range = [{boundaries['x_min']:.2f}, {boundaries['x_max']:.2f}]")
            print(f"  y_pos_range = [{boundaries['y_min']:.2f}, {boundaries['y_max']:.2f}]")
        else:
            print("No reachable points found!")
        
        # Plot
        save_path = args.save if args.save else f'workspace_2d_{args.mode}_z{args.z:.3f}.png'
        plot_workspace_2d(X, Y, reachable, mode=args.mode, z_height=args.z, save_path=save_path)
    
    print("\n" + "=" * 70)
    print("Workspace analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()