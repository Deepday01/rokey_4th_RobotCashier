#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from cashier_interfaces.action import VoiceSession
from cashier_interfaces.msg import Item


class DemoVoiceNode(Node):
    def __init__(self):
        super().__init__("demo_voice_node")

        self.voice_server = ActionServer(
            self,
            VoiceSession,
            "/voice/run_session",
            execute_callback=self.execute_voice,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("demo_voice_node ready: /voice/run_session")

    def goal_callback(self, goal_request):
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        return CancelResponse.ACCEPT

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
    node = DemoVoiceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
