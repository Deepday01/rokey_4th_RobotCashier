#!/usr/bin/env python3

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from cashier_interfaces.action import ExecutePacking
from cashier_interfaces.msg import Item, Placement


class ExecutePackingClient(Node):

    def __init__(self):
        super().__init__("execute_packing_client")

        self.get_logger().info("1️⃣ ExecutePackingClient Node 시작")

        self._client = ActionClient(
            self,
            ExecutePacking,
            "/execute_packing/placing",
        )

    def send_goal(self):

        self.get_logger().info("2️⃣ Action Server 연결 대기")
        self._client.wait_for_server()
        self.get_logger().info("   Action Server 연결 완료")

        self.get_logger().info("3️⃣ Goal 메시지 생성")
        goal_msg = ExecutePacking.Goal()

        # -----------------------------
        # Item 생성
        # -----------------------------

        self.get_logger().info("4️⃣ Pick Item 생성")

        item1 = Item()
        item1.item_id = "item_001"
        item1.name = "홀스"
        item1.width = 74
        item1.depth = 44
        item1.height = 13
        item1.durability = 4
        item1.x = 300.853
        item1.y = 130.238
        item1.z = 200.0
        item1.roll = 78.647
        item1.pitch = 179.995
        item1.yaw = -13.647

        self.get_logger().info(
            f"   item1 생성 완료 | id={item1.item_id} "
            f"pose=({item1.x},{item1.y},{item1.z})"
        )

        self.get_logger().info("5️⃣ Placement 생성")

        place1 = Placement()
        place1.object_index = 0
        place1.x = 300.795
        place1.y = -130.951
        place1.z = 200.0
        place1.roll = 130.759
        place1.pitch = -179.998
        place1.yaw = 38.473

        self.get_logger().info(
            f"   place1 생성 | object_index={place1.object_index} "
            f"pose=({place1.x},{place1.y},{place1.z})"
        )

        # item2 = Item()
        # item2.item_id = "item_002"
        # item2.name = "홀스"
        # item2.width = 74
        # item2.depth = 44
        # item2.height = 13
        # item2.durability = 4
        # item2.x = 403.853
        # item2.y = 162.238
        # item2.z = 294.505
        # item2.roll = 78.647
        # item2.pitch = 179.995
        # item2.yaw = -13.647

        # self.get_logger().info(
        #     f"   item2 생성 완료 | id={item2.item_id} "
        #     f"pose=({item2.x},{item2.y},{item2.z})"
        # )

        # -----------------------------
        # Placement 생성
        # -----------------------------

        # place2 = Placement()
        # place2.object_index = 1
        # place2.x = 403.795
        # place2.y = -149.951
        # place2.z = 294.518
        # place2.roll = 130.759
        # place2.pitch = -179.998
        # place2.yaw = 38.473

        # self.get_logger().info(
        #     f"   place2 생성 | object_index={place2.object_index} "
        #     f"pose=({place2.x},{place2.y},{place2.z})"
        # )

        # goal_msg.pick_items = [item1, item2]
        # goal_msg.place_items = [place1, place2]

        goal_msg.pick_items = [item1]
        goal_msg.place_items = [place1]

        self.get_logger().info(
            f"6️⃣ Goal 전송 | pick_items={len(goal_msg.pick_items)} "
            f"place_items={len(goal_msg.place_items)}"
        )

        future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback,
        )

        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):

        self.get_logger().info("7️⃣ Goal 응답 수신")

        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().error("   Goal 거부됨")
            rclpy.shutdown()
            return

        self.get_logger().info("   Goal 수락됨")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):

        feedback = feedback_msg.feedback

        self.get_logger().info(
            f"8️⃣ 실행 중 Feedback 수신 | {feedback}"
        )

    def result_callback(self, future):

        result = future.result().result

        self.get_logger().info(
            f"9️⃣ 작업 완료 | result={result}"
        )

        self.get_logger().info("🔟 Node 종료")

        rclpy.shutdown()


def main(args=None):

    rclpy.init(args=args)

    node = ExecutePackingClient()

    node.send_goal()

    rclpy.spin(node)


if __name__ == "__main__":
    main()


