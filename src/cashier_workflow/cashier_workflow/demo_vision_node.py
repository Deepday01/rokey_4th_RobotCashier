#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from cashier_interfaces.action import ScanItems
from cashier_interfaces.msg import Item


class DemoVisionNode(Node):
    def __init__(self):
        super().__init__("demo_vision_node")

        self.vision_server = ActionServer(
            self,
            ScanItems,
            "/vision/scan_items",
            execute_callback=self.execute_vision,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("demo_vision_node ready: /vision/scan_items")

    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_vision(self, goal_handle):
        req = goal_handle.request

        result = ScanItems.Result()
        result.success = False
        result.items_scan = []

        if not req.start_vision:
            self.get_logger().warn("[DEMO VISION] start_vision=False")
            goal_handle.succeed()
            return result

        for sec in range(1, 3):
            feedback = ScanItems.Feedback()
            feedback.runtime = sec
            goal_handle.publish_feedback(feedback)
            time.sleep(0.2)

        result.success = True
        result.items_scan = [
            self.make_item("item_001", "milk", 0.07, 0.07, 0.20, 2),
            self.make_item("item_002", "snack_box", 0.15, 0.10, 0.05, 3),
        ]

        self.get_logger().info(f"[DEMO VISION] success, scanned={len(result.items_scan)}")
        goal_handle.succeed()
        return result

    def make_item(self, item_id, name, width, depth, height, durability):
        item = Item()
        item.item_id = item_id
        item.name = name
        item.width = width
        item.depth = depth
        item.height = height
        item.durability = durability
        item.x = 0.10
        item.y = 0.20
        item.z = 0.00
        item.roll = 0.0
        item.pitch = 0.0
        item.yaw = 0.0
        return item


def main(args=None):
    rclpy.init(args=args)
    node = DemoVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
