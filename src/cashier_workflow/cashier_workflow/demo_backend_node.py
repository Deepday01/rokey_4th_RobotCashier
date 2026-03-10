#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from cashier_interfaces.action import VoiceSession, ScanItems, ExecutePacking
from cashier_interfaces.srv import ComputePackingPlan
from cashier_interfaces.msg import Item, Placement


class DemoBackendNode(Node):
    def __init__(self):
        super().__init__("demo_backend_node")

        # 1) Voice action server
        self.voice_server = ActionServer(
            self,
            VoiceSession,
            "/voice/run_session",
            execute_callback=self.execute_voice,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        # 2) Vision action server
        self.vision_server = ActionServer(
            self,
            ScanItems,
            "/vision/scan_items",
            execute_callback=self.execute_vision,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        # 3) Plan service server
        self.plan_service = self.create_service(
            ComputePackingPlan,
            "/plan_packing/compute_plan",
            self.handle_plan,
        )

        # 4) Execute action server
        self.execute_server = ActionServer(
            self,
            ExecutePacking,
            "/execute_packing/placing",
            execute_callback=self.execute_packing,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("Demo backend node ready")

    # --------------------------------------------------
    # Common callbacks
    # --------------------------------------------------
    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

    # --------------------------------------------------
    # VoiceSession action
    # --------------------------------------------------
    def execute_voice(self, goal_handle):
        req = goal_handle.request
        mode = req.mode.strip().upper()

        feedback = VoiceSession.Feedback()
        feedback.runtime = 1
        goal_handle.publish_feedback(feedback)
        time.sleep(0.2)

        result = VoiceSession.Result()
        result.success = True
        result.command = 1
        result.items_out = []

        if mode == "WAKEUP":
            self.get_logger().info("[DEMO VOICE] WAKEUP success")

        elif mode == "EDIT":
            items = list(req.items_in)

            # vision 결과가 있으면 그대로 반환
            if not items:
                items = [self.make_item("item_999", "fallback_item", 0.10, 0.08, 0.12, 3)]

            result.items_out = items
            self.get_logger().info(f"[DEMO VOICE] EDIT success, items_out={len(items)}")

        else:
            result.success = False
            result.command = -1
            self.get_logger().warn(f"[DEMO VOICE] unknown mode: {req.mode}")

        goal_handle.succeed()
        return result

    # --------------------------------------------------
    # ScanItems action
    # --------------------------------------------------
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

    # --------------------------------------------------
    # ComputePackingPlan service
    # --------------------------------------------------
    def handle_plan(self, request, response):
        items = list(request.items)

        response.success = len(items) > 0
        response.placements = []

        base_x = 0.50
        base_y = 0.00
        gap_y = 0.12

        for i, _ in enumerate(items):
            p = Placement()
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

    # --------------------------------------------------
    # ExecutePacking action
    # --------------------------------------------------
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
            self.get_logger().info(f"[DEMO EXECUTE] packing {i + 1}/{total}")
            time.sleep(0.3)

        result.success = True
        self.get_logger().info("[DEMO EXECUTE] success")
        goal_handle.succeed()
        return result

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
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
    node = DemoBackendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()