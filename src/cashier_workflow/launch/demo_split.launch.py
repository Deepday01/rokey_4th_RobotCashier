from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="cashier_workflow",
            executable="demo_voice_node",
            name="demo_voice_node",
            output="screen",
        ),
        Node(
            package="cashier_workflow",
            executable="demo_vision_node",
            name="demo_vision_node",
            output="screen",
        ),
        # Node(
        #     package="cashier_workflow",
        #     executable="demo_plan_packing_node",
        #     name="demo_plan_packing_node",
        #     output="screen",
        # ),
        Node(
            package="cashier_workflow",
            executable="demo_execute_packing_node",
            name="demo_execute_packing_node",
            output="screen",
        ),
        Node(
            package="cashier_workflow",
            executable="workflow_node",
            name="workflow_node",
            output="screen",
        ),
    ])
