#!/usr/bin/env python3
"""
Setup script for SO-101 Robotics Course Labs
"""

from setuptools import setup, find_packages

setup(
    name='so101_robotics_course',
    version='1.0.0',
    description='Hands-on labs for learning classical robotics with the SO-101 arm',
    author='SO-101 Robotics Course',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.10',
    install_requires=[
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'matplotlib>=3.7.0',
    ],
    extras_require={
        'mujoco': ['mujoco>=3.0.0'],
        'ros2': [],  # ROS2 packages installed via rosdep
        'lerobot': ['lerobot[kinematics]'],
        'all': [
            'mujoco>=3.0.0',
            'opencv-python>=4.8.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'lab1_fk_mujoco=labs.lab1_2_test_fk_mujoco:run_fk_test',
            'lab2_ik_mujoco=labs.lab2_1_test_ik_mujoco:run_ik_test',
            'lab3_pick_place=labs.lab3_pick_place_mujoco:run_pick_and_place',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Robotics',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)
