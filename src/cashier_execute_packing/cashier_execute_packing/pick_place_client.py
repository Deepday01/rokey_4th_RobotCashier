#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from robot_msgs.action import ExecutePacking
from robot_msgs.msg import Item, Placement


class ExecutePackingClient(Node):
    def __init__(self):
        super().__init__("execute_packing_client")
        self._client = ActionClient(self, ExecutePacking, "/execute_packing/placing")

    def send_goal(self):
        goal_msg = ExecutePacking.Goal()

        # -------------------------
        # pick_items
        # -------------------------
        item1 = Item()
        item1.item_id = "item_001"
        item1.name = "물병"
        item1.width = 60.0
        item1.depth = 60.0
        item1.height = 200.0
        item1.durability = 4
        item1.x = 200.0
        item1.y = 200.0
        item1.z = 200.0
        item1.roll = 62.496
        item1.pitch = 179.976
        item1.yaw = 62.158

        item2 = Item()
        item2.item_id = "item_002"
        item2.name = "박스"
        item2.width = 120.0
        item2.depth = 80.0
        item2.height = 100.0
        item2.durability = 3
        item1.x = 250.0
        item1.y = 250.0
        item1.z = 200.0
        item2.roll = 62.496
        item2.pitch = 179.976
        item2.yaw = 62.158

        # -------------------------
        # place_items
        # object_index는 pick_items의 index와 매칭
        # -------------------------
        place1 = Placement()
        place1.object_index = 0
        place1.x = 200.0
        place1.y = 100.0
        place1.z = 200.0
        place1.roll = 62.496
        place1.pitch = 179.976
        place1.yaw = 62.158

        place2 = Placement()
        place2.object_index = 1
        place1.x = 200.0
        place1.y = 100.0
        place1.z = 200.0
        place2.roll = 62.496
        place2.pitch = 179.976
        place2.yaw = 62.158

        goal_msg.pick_items = [item1, item2]
        goal_msg.place_items = [place1, place2]

        self.get_logger().info("Waiting for action server...")
        self._client.wait_for_server()

        self.get_logger().info("Sending goal...")
        future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().warn("Goal rejected")
            rclpy.shutdown()
            return

        self.get_logger().info("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        fb = feedback_msg.feedback
        self.get_logger().info(
            f"Feedback - progress: {fb.progress}, runtime: {fb.runtime}s"
        )

    def result_callback(self, future):
        result = future.result().result
        status = future.result().status

        self.get_logger().info(f"Result status: {status}")
        self.get_logger().info(f"Success: {result.success}")

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ExecutePackingClient()
    node.send_goal()
    rclpy.spin(node)


if __name__ == "__main__":
    main()


