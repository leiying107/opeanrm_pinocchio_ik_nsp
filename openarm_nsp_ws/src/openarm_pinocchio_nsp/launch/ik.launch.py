# Copyright 2026 Enactic, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Launch the NSP IK node for one arm side.

Example::
    ros2 launch openarm_pinocchio_nsp ik.launch.py side:=right
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    side = LaunchConfiguration("side")
    return LaunchDescription(
        [
            DeclareLaunchArgument("side", default_value="right"),
            Node(
                package="openarm_pinocchio_nsp",
                executable="ik_node",
                name="ik_node",
                output="screen",
                parameters=[{"side": side}],
            ),
        ]
    )
