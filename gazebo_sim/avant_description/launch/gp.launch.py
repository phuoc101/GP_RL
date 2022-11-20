"""
This launch file can be run when user wants to launch gazebo, rviz, gazebo control interface and motion control nodes at the
same time.
"""

import launch
from launch.substitutions import Command, LaunchConfiguration
import launch_ros
from launch_ros.parameter_descriptions import ParameterValue
import os
import sys

def generate_launch_description():
    # get paths to the file locations on the system
    pkg_share = launch_ros.substitutions.FindPackageShare(package='avant_description').find('avant_description')
    default_model_path = os.path.join(pkg_share, 'urdf/avant_bucket.urdf')
    default_rviz_config_path = os.path.join(pkg_share, 'rviz/urdf_config.rviz')

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
        executable='control_client_gp_node',
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
        #robot_control_node,

        # set timer action for launching rviz node to avoid unnecessery amount of warning messages
        #launch.actions.TimerAction(
        #    period=4.0,
        #    actions = [rviz_node]
        #),

        # active the controllers for the ros2 control
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
        #launch.actions.ExecuteProcess(
        #cmd=['ros2', 'control', 'load_controller', '--set-state', 'start', 'joint_trajectory_controller'],
        #output='screen'
        #),
        robot_localization_node
    ])
