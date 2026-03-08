from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cashier_workflow",
            executable="demo_backend_node",
            name="demo_backend_node",
            output="screen",
        ),
        Node(
            package="cashier_workflow",
            executable="workflow_node",
            name="workflow_node",
            output="screen",
        ),
    ])