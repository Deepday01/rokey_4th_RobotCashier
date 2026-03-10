#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cashier_interfaces.action import VoiceSession
from cashier_interfaces.msg import Item


class DummyVoiceClient(Node):
    def __init__(self):
        super().__init__('dummy_voice_client')
        self._client = ActionClient(self, VoiceSession, '/voice/run_session')

        self.item_map = {
            1: "halls",
            2: "insect",
            3: "caramel",
            4: "candy",
            5: "cream",
            6: "eclipse_red",
            7: "eclipse_green",
        }

        self.pending_edit = False

    def make_item_msg(self, item_name: str) -> Item:
        msg = Item()
        msg.item_id = item_name
        msg.name = item_name

        # 더미값
        msg.width = 0
        msg.depth = 0
        msg.height = 0
        msg.durability = 0
        msg.x = 0.0
        msg.y = 0.0
        msg.z = 0.0
        msg.roll = 0.0
        msg.pitch = 0.0
        msg.yaw = 0.0
        return msg

    def input_items(self):
        items = []

        print("\n추가할 상품 번호를 입력하세요.")
        print("1: halls")
        print("2: insect")
        print("3: caramel")
        print("4: candy")
        print("5: cream")
        print("6: eclipse_red")
        print("7: eclipse_green")
        print("0: 입력 종료 후 EDIT goal 전송\n")

        while True:
            raw = input("번호 입력: ").strip()

            if not raw.isdigit():
                print("숫자만 입력하세요.")
                continue

            num = int(raw)

            if num == 0:
                break

            if num not in self.item_map:
                print("존재하지 않는 번호입니다.")
                continue

            item_name = self.item_map[num]
            item_msg = self.make_item_msg(item_name)
            items.append(item_msg)

            print(f"추가됨: {item_name} | 현재 {len(items)}개")

        return items

    def send_wakeup_goal(self):
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action server not available')
            rclpy.shutdown()
            return

        goal_msg = VoiceSession.Goal()
        goal_msg.mode = "WAKEUP"
        goal_msg.items_in = []

        self.get_logger().info("Sending WAKEUP goal...")

        future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def send_edit_goal(self, items):
        goal_msg = VoiceSession.Goal()
        goal_msg.mode = "EDIT"
        goal_msg.items_in = items

        self.get_logger().info(
            f"Sending EDIT goal with {len(goal_msg.items_in)} items..."
        )

        future = self._client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback
        )
        future.add_done_callback(self.goal_response_callback)

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f"Feedback runtime: {feedback.runtime}")

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected")
            rclpy.shutdown()
            return

        self.get_logger().info("Goal accepted")
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result

        self.get_logger().info(f"success: {result.success}")
        self.get_logger().info(f"command: {result.command}")
        self.get_logger().info(f"items_out count: {len(result.items_out)}")

        for i, item in enumerate(result.items_out):
            self.get_logger().info(
                f"[{i}] item_id={item.item_id}, name={item.name}"
            )

        # 1차 WAKEUP 결과 처리
        if not self.pending_edit:
            if result.command == 1:   # scan
                self.pending_edit = True

                items = self.input_items()

                if len(items) == 0:
                    print("입력된 상품이 없습니다. 빈 리스트로 EDIT goal을 전송합니다.")

                self.send_edit_goal(items)
                return
            else:
                self.get_logger().info("scan 명령이 아니므로 다시 WAKEUP으로 대기합니다.")
                self.pending_edit = False
                self.send_wakeup_goal()
                return

        # EDIT 결과 처리
        if result.command == 2:   # CMD_PACK
            self.get_logger().info("포장 명령이 확인되었습니다.")
        else:
            self.get_logger().info("포장 없이 세션을 종료합니다.")

        # 다음 손님/다음 명령을 위해 다시 WAKEUP
        self.pending_edit = False
        self.get_logger().info("다시 WAKEUP 모드로 돌아갑니다.")
        self.send_wakeup_goal()


def main(args=None):
    rclpy.init(args=args)

    node = DummyVoiceClient()
    node.send_wakeup_goal()

    rclpy.spin(node)


if __name__ == '__main__':
    main()