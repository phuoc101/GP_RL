import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
from launch_ros.parameter_descriptions import ParameterValue
import os
import time

"""
This launch file on only launches the gazebo interface controller
"""


def generate_launch_description():

    robot_control_node = launch_ros.actions.Node(
        package='gazebo_control_interface',
        executable='control_client',
    )

    return launch.LaunchDescription([
        robot_control_node
    ])
