from launch import LaunchDescription
from launch_ros.actions import Node

from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    debug_mode = LaunchConfiguration("debug_mode")

    debug_mode_arg = DeclareLaunchArgument(
        "debug_mode",
        default_value="false",
        description="Enable debug logs in workflow_node"
    )



    return LaunchDescription([
        debug_mode_arg,

        # Node(
        #     package="cashier_workflow",
        #     executable="demo_voice_node",
        #     name="demo_voice_node",
        #     output="screen",
        # ),
        # Node(
        #     package="cashier_workflow",
        #     executable="demo_vision_node",
        #     name="demo_vision_node",
        #     output="screen",
        # ),
        Node(
            package="cashier_workflow",
            executable="demo_plan_packing_node",
            name="demo_plan_packing_node",
            output="screen",
        ),
        Node(
            package="cashier_workflow",
            executable="demo_execute_packing_node",
            name="demo_execute_packing_node",
            output="screen",
        ),
        Node(
            package="cashier_workflow",
            executable="workflow_node_dev",
            name="workflow_node",
            output="screen",
            parameters=[
                 {"debug_mode": debug_mode}
            ],
        ),
    ])
