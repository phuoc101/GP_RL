"""
This launch file can be run when user wants to launch gazebo, rviz, gazebo control interface and motion control nodes at the
same time.
"""

import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import (DeclareLaunchArgument, GroupAction,
                            IncludeLaunchDescription, SetEnvironmentVariable)
from ament_index_python.packages import get_package_share_directory
import os
import sys

def generate_launch_description():
    # get paths to the file locations on the system
    pkg_share = launch_ros.substitutions.FindPackageShare(package='avant_description').find('avant_description')
    control_pkg_share = get_package_share_directory('gazebo_control_interface')
    default_model_path = os.path.join(pkg_share, 'urdf/avant_bucket.urdf')
    default_rviz_config_path = os.path.join(pkg_share, 'rviz/urdf_config.rviz')

    training_data = os.path.join(control_pkg_share, 'data', 'avant_TrainingData.pkl')
    model_data = os.path.join(control_pkg_share, 'results', 'gp', 'GPmodel.pkl')
    controller_data = os.path.join(control_pkg_share, 'results', 'controller', '_all.pkl')

    dft_training_model = LaunchConfiguration('path_to_training_data')
    dft_model_data = LaunchConfiguration("path_to_model_data")
    dft_controller_data = LaunchConfiguration('controller_data')

    declare_dft_training_model_cmd = DeclareLaunchArgument(
        name='path_to_training_data',
        default_value=training_data,
        description='location of the training data')

    declare_dft_model_data_cmd = DeclareLaunchArgument(
        name='path_to_model_data',
        default_value=model_data,
        description='Location of the model data')

    declare_dft_controller_data_cmd = DeclareLaunchArgument(
        name='controller_data',
        default_value=controller_data,
        description='Location of the controller data')

    spawn_x_val = "0.0"
    spawn_y_val = '0.0'
    spawn_z_val = '1.0'
    spawn_yaw_val = '0.0'
    for arg in sys.argv:
        if arg.startswith("x:="):
            spawn_x_val = arg.split(":=")[1]

        if arg.startswith("y:="):
            spawn_y_val = arg.split(":=")[1]

        if arg.startswith("z:="):
            spawn_z_val = arg.split(":=")[1]

        if arg.startswith("yaw:="):
            spawn_yaw_val = arg.split(":=")[1]

    # Define the robot state publisher
    robot_state_publisher_node = launch_ros.actions.Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        parameters=[{'robot_description': ParameterValue(Command(['xacro ', LaunchConfiguration('model')]), value_type=str)}]
    )

    # Define the rviz configuration 
    rviz_node = launch_ros.actions.Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', LaunchConfiguration('rvizconfig')],
    )   


    # Define the spawner and the object to spawn
    spawn_entity = launch_ros.actions.Node(
    	package='gazebo_ros', 
    	executable='spawn_entity.py',
        arguments=['-entity', 'avant', '-topic', 'robot_description',
                    '-x', spawn_x_val,
                    '-y', spawn_y_val,
                    '-z', spawn_z_val,
                    '-Y', spawn_yaw_val],
        output='screen',
    )

    robot_control_node = launch_ros.actions.Node(
        package='gazebo_control_interface',
        executable='pose_control_node',
        parameters=[os.path.join(pkg_share, 'config/interface_config.yaml'),
        {"path_to_training_data": dft_training_model},
        {"path_to_model_data": dft_model_data},
        {"controller_data": dft_controller_data}
        ]
    )

    robot_localization_node = launch_ros.actions.Node(
       package='robot_localization',
       executable='ekf_node',
       name='ekf_filter_node',
       output='screen',
       parameters=[os.path.join(pkg_share, 'config/ekf.yaml'), {'use_sim_time': LaunchConfiguration('use_sim_time')}]
    )

    
    # Launch the defined parameters
    return launch.LaunchDescription([
        launch.actions.DeclareLaunchArgument(name='use_sim_time', default_value='true',
                                            description='Flag to enable use_sim_time'),      
        launch.actions.DeclareLaunchArgument(name='model', default_value=default_model_path,
                                            description='Absolute path to robot urdf file'),
        launch.actions.ExecuteProcess(cmd=['gazebo', '--verbose', '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so'], output='screen'),
        launch.actions.DeclareLaunchArgument(name='rvizconfig', default_value=default_rviz_config_path,
                                            description='Absolute path to rviz config file'),
        robot_state_publisher_node,
        spawn_entity,
        declare_dft_training_model_cmd,
        declare_dft_model_data_cmd,
        declare_dft_controller_data_cmd,
        robot_control_node,

        launch.actions.ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'start', 'joint_state_broadcaster'],
        output='screen'
        ),
        launch.actions.ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'start', 'motion_controller'],
        output='screen'
        ),
        launch.actions.ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'start', 'manipulator_controller'],
        output='screen'
        ),

        robot_localization_node
    ])
