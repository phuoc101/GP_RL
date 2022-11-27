from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="avant_bagreader",
                executable="bagreader_node",
                name="bagreader",
            ),
        ]
    )
