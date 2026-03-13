# 4개의 데모 노드와 1개의 실제 메인노드를 실행.
# 하나씩 주석 처리해서 사용


from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
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
            executable="workflow_node",
            name="workflow_node",
            output="screen",
        ),
    ])
