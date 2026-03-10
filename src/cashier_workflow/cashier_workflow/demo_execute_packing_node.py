#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from cashier_interfaces.action import ExecutePacking


class DemoExecutePackingNode(Node):
    def __init__(self):
        super().__init__("demo_execute_packing_node")

        self.execute_server = ActionServer(
            self,
            ExecutePacking,
            "/execute_packing/placing",
            execute_callback=self.execute_packing,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("demo_execute_packing_node ready: /execute_packing/placing")

    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    def execute_packing(self, goal_handle):
        req = goal_handle.request
        total = min(len(req.pick_items), len(req.place_items))

        result = ExecutePacking.Result()
        result.success = False

        if total == 0:
            self.get_logger().warn("[DEMO EXECUTE] no work to do")
            goal_handle.succeed()
            return result

        for i in range(total):
            feedback = ExecutePacking.Feedback()
            feedback.progress = f"{i + 1}/{total}"
            feedback.runtime = i + 1
            goal_handle.publish_feedback(feedback)
            self.get_logger().debug(f"[DEMO EXECUTE] packing {i + 1}/{total}")
            time.sleep(0.3)

        result.success = True
        self.get_logger().info("[DEMO EXECUTE] success")
        goal_handle.succeed()
        return result


def main(args=None):
    rclpy.init(args=args)
    node = DemoExecutePackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
