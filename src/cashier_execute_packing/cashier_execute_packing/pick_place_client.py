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
        item1 = Item()
        item1.item_id = "item_001"
        item1.name = "에크론"
        item1.width = 125
        item1.depth = 34   
        item1.height = 44
        item1.durability = 4
        item1.x = 403.853
        item1.y = 162.238
        item1.z = 0.0
        item1.roll = 0.0
        item1.pitch = 0.0
        item1.yaw = 0.0

        # -----------------------------
        # Placement 생성
        # -----------------------------
        place1 = Placement()
        place1.object_index = 0
        place1.x = 220.0 + 50
        place1.y = 20.0 + 50
        place1.z = 5.0 + 150
        place1.roll = 0.0
        place1.pitch = 0.0
        place1.yaw = 0.0


  # -----------------------------
        # Item 생성
        # -----------------------------
        item2 = Item()
        item2.item_id = "item_002"
        item2.name = "이클립스"
        item2.width = 85
        item2.depth = 31
        item2.height = 48
        item2.durability = 5
        item2.x = 250.6
        item2.y = 277.1
        item2.z = 0.0
        item2.roll = 0.0
        item2.pitch = 0.0
        item2.yaw = 0.0
   

        # -----------------------------
        # Placement 생성
        # -----------------------------
        place2 = Placement()
        place2.object_index = 1
        place2.x = 220.0 + 50
        place2.y = 20.0 + 50
        place2.z = 5.0 + 150
        place2.roll = 90.0
        place2.pitch = 0.0
        place2.yaw = 0.0

      

        # -----------------------------
        # Item 생성
        # -----------------------------
        item3 = Item()
        item3.item_id = "item_003"
        item3.name = "아이셔"
        item3.width = 120
        item3.depth = 52
        item3.height = 85
        item3.durability = 4
        item3.x = 403.853
        item3.y = 162.238
        item3.z = 0.0
        item3.roll = 0.0
        item3.pitch = 0.0
        item3.yaw = 0.0

    
        # -----------------------------
        # Placement 생성
        # -----------------------------
        place3 = Placement()
        place3.object_index = 2 
        place3.x = 220.0 + 50
        place3.y = 20.0 + 50
        place3.z = 5.0 + 150
        place3.roll = 90.0
        place3.pitch = 90.0
        place3.yaw = 0.0


        # -----------------------------
        # Item 생성
        # -----------------------------
        item4 = Item()
        item4.item_id = "item_004"
        item4.name = "카라멜"
        item4.width = 82
        item4.depth = 32
        item4.height = 54
        item4.durability = 4
        item4.x = 403.853
        item4.y = 162.238
        item4.z = 0.0
        item4.roll = 0.0
        item4.pitch = 0.0
        item4.yaw = 0.0

       

        # -----------------------------
        # Placement 생성
        # -----------------------------
        place4 = Placement()
        place4.object_index = 3
        place4.x = 220.0 + 50
        place4.y = 20.0 + 50
        place4.z = 5.0 + 150
        place4.roll = 90.0
        place4.pitch = 90.0
        place4.yaw = 90.0


        # -----------------------------
        # Item 생성
        # -----------------------------
        item5 = Item()
        item5.item_id = "item_005"
        item5.name = "나비"
        item5.width = 91
        item5.depth = 51
        item5.height = 65
        item5.durability = 4
        item5.x = 403.853
        item5.y = 162.238
        item5.z = 0.0
        item5.roll = 0.0
        item5.pitch = 0.0
        item5.yaw = 0.0

      
        # -----------------------------
        # Placement 생성
        # -----------------------------
        place5 = Placement()
        place5.object_index = 4
        place5.x = 220.0 + 50
        place5.y = 20.0 + 50
        place5.z = 5.0 + 150
        place5.roll = 0.0
        place5.pitch = 90.0
        place5.yaw = 0.0


        # -----------------------------
        # Item 생성
        # -----------------------------
        item6 = Item()
        item6.item_id = "item_006"
        item6.name = "홀스"
        item6.width = 81
        item6.depth = 25
        item6.height = 53
        item6.durability = 4
        item6.x = 403.853
        item6.y = 162.238
        item6.z = 0.0
        item6.roll = 0.0
        item6.pitch = 0.0
        item6.yaw = 0.0

       

        # -----------------------------
        # Placement 생성
        # -----------------------------
        place6 = Placement()
        place6.object_index = 5
        place6.x = 220.0 + 50
        place6.y = 20.0 + 50
        place6.z = 5.0 + 150
        place6.roll = 0.0
        place6.pitch = 90.0
        place6.yaw = 90.0


        goal_msg.pick_items = [
            item1, 
            item2,
            item3,
            item4,
            item5,
            item6,
        ]
        goal_msg.place_items = [
            place1, 
            place2,
            place3,
            place4,
            place5,
            place6,
        ]

        # goal_msg.pick_items = [item1]
        # goal_msg.place_items = [place1]
        
        # goal_msg.pick_items = [item2]
        # goal_msg.place_items = [place2]

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


