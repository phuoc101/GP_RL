"""
This launch file can be run when user wants to launch gazebo, rviz, gazebo control
interface and motion control nodes at the same time.
"""

import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os
import sys


def generate_launch_description():
    # get paths to the file locations on the system
    pkg_share = launch_ros.substitutions.FindPackageShare(
        package="avant_description"
    ).find("avant_description")
    control_pkg_share = get_package_share_directory("gazebo_control_interface")
    default_model_path = os.path.join(pkg_share, "urdf/avant_bucket.urdf")
    default_rviz_config_path = os.path.join(pkg_share, "rviz/urdf_config.rviz")

    training_data = os.path.join(control_pkg_share, "data", "avant_TrainingData.pkl")
    model_data = os.path.join(control_pkg_share, "results", "gp/")
    controller_data = os.path.join(
        control_pkg_share, "results", "controller/",
    )

    dft_training_model = LaunchConfiguration("path_to_training_data")
    dft_model_data = LaunchConfiguration("path_to_model_data")
    dft_controller_data = LaunchConfiguration("controller_data")

    declare_dft_training_model_cmd = DeclareLaunchArgument(
        name="path_to_training_data",
        default_value=training_data,
        description="location of the training data",
    )

    declare_dft_model_data_cmd = DeclareLaunchArgument(
        name="path_to_model_data",
        default_value=model_data,
        description="Location of the model data",
    )

    declare_dft_controller_data_cmd = DeclareLaunchArgument(
        name="controller_data",
        default_value=controller_data,
        description="Location of the controller data",
    )

    robot_control_node = launch_ros.actions.Node(
        package="gazebo_control_interface",
        executable="real_avant_control_node",
        parameters=[
            os.path.join(pkg_share, "config/interface_config.yaml"),
            {"path_to_training_data": dft_training_model},
            {"path_to_model_data": dft_model_data},
            {"controller_data": dft_controller_data},
        ],
    )

    # Launch the defined parameters
    return launch.LaunchDescription(
        [

            declare_dft_training_model_cmd,
            declare_dft_model_data_cmd,
            declare_dft_controller_data_cmd,
            robot_control_node
        ]
    )
