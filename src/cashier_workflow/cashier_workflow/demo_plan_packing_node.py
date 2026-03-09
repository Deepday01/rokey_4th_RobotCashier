#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from cashier_interfaces.srv import ComputePackingPlan
from cashier_interfaces.msg import Placement


class DemoPlanPackingNode(Node):
    def __init__(self):
        super().__init__("demo_plan_packing_node")

        self.plan_service = self.create_service(
            ComputePackingPlan,
            "/plan_packing/compute_plan",
            self.handle_plan,
        )

        self.get_logger().info("demo_plan_packing_node ready: /plan_packing/compute_plan")

    def handle_plan(self, request, response):
        items = list(request.items)

        response.success = len(items) > 0
        response.placements = []

        base_x = 0.50
        base_y = 0.00
        gap_y = 0.12

        for i, _ in enumerate(items):
            p = Placement()
            p.object_index = i
            p.x = base_x
            p.y = base_y + i * gap_y
            p.z = 0.10
            p.roll = 0.0
            p.pitch = 0.0
            p.yaw = 0.0
            response.placements.append(p)

        self.get_logger().info(
            f"[DEMO PLAN] success={response.success}, placements={len(response.placements)}"
        )
        return response


def main(args=None):
    rclpy.init(args=args)
    node = DemoPlanPackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
