#!/usr/bin/env python3
# workflow_node.py

import time
import threading
from enum import Enum, auto
from typing import List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from cashier_interfaces.action import VoiceSession, ScanItems, ExecutePacking
from cashier_interfaces.srv import ComputePackingPlan
from cashier_interfaces.msg import Item, Placement


class WorkflowState(Enum):
    IDLE = auto()
    VOICE_WAKEUP = auto()
    VISION = auto()
    VOICE_EDIT = auto()
    PLAN_PACKING = auto()
    EXECUTE_PACKING = auto()
    DONE = auto()
    ERROR = auto()


class WorkflowNode(Node):
    def __init__(self):
        super().__init__("workflow_node")
        
        # ---- 각 노드와 통신할 clients ----
        self.voice_client = ActionClient(
            self,
            VoiceSession,
            "/voice/run_session"
        )
        self.vision_client = ActionClient(
            self,
            ScanItems,
            "/vision/scan_items"
        )
        self.plan_client = self.create_client(
            ComputePackingPlan,
            "/plan_packing/compute_plan"
        )
        self.execute_client = ActionClient(
            self,
            ExecutePacking,
            "/execute_packing/placing"
        )

        # ---- workflow state ----
        self.state = WorkflowState.IDLE
        self.is_running = False
        self.last_success_step = ""
        self.last_error = ""

        # ---- per-order context ----
        self.items_scanned: List[Item] = []
        self.items_final: List[Item] = []
        self.packing_plan: List[Placement] = []

        # IDLE 상태일 때 자동으로 한 사이클 시작
        # WAKEUP 액션이 실제로는 음성 대기를 담당하므로,
        # workflow는 계속 IDLE -> WAKEUP 사이클로 두어도 괜찮음
        self.create_timer(1.0, self._tick)

        self.get_logger().info("workflow_node ready. state=IDLE")

    def _item_names(self, items):
      return [item.name for item in items]

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def _tick(self): # 시동버튼
        if self.is_running:
            return
        # 1초에 한번씩 상태 체크 후 IDLE일때 이전 작업 초기화 후 새 쓰레드로 실행
        if self.state == WorkflowState.IDLE:
            self.is_running = True
            self._reset_order_context()
            threading.Thread(target=self._run_workflow, daemon=True).start()
    
    # 본체
    def _run_workflow(self):
        try:
            # 1) VOICE_WAKEUP
            self._set_state(WorkflowState.VOICE_WAKEUP)
            success, _ = self.call_voice(mode="WAKEUP")
            if not success:
                raise RuntimeError("voice wakeup failed")
            self.last_success_step = "VOICE_WAKEUP"

            # 2) VISION
            self._set_state(WorkflowState.VISION)
            success, scanned_items = self.call_vision()
            if not success or not scanned_items:
                raise RuntimeError("vision scan failed or returned no items")
            self.items_scanned = scanned_items
            self.last_success_step = "VISION"

            # 3) VOICE_EDIT
            self._set_state(WorkflowState.VOICE_EDIT)
            success, final_items = self.call_voice(
                mode="EDIT",
                items=self.items_scanned
            )
            if not success or not final_items:
                raise RuntimeError("voice edit failed or returned no items")
            self.items_final = final_items
            self.last_success_step = "VOICE_EDIT"

            # 4) PLAN_PACKING
            self._set_state(WorkflowState.PLAN_PACKING)
            success, placements = self.call_plan_packing(self.items_final)
            if not success or not placements:
                raise RuntimeError("packing plan failed or returned no placements")

            if len(placements) != len(self.items_final):
                raise RuntimeError("items_final and packing_plan size mismatch")

            self.packing_plan = placements
            self.last_success_step = "PLAN_PACKING"

            # 5) EXECUTE_PACKING
            self._set_state(WorkflowState.EXECUTE_PACKING)
            success = self.call_execute_packing(
                self.items_final,
                self.packing_plan
            )
            if not success:
                raise RuntimeError("execute packing failed")
            self.last_success_step = "EXECUTE_PACKING"

            # 6) DONE
            self._set_state(WorkflowState.DONE)
            self.get_logger().info("workflow completed successfully")

        except Exception as e:
            self.last_error = str(e)
            self._set_state(WorkflowState.ERROR)
            self.get_logger().error(f"workflow failed: {self.last_error}")

        finally:
            # MVP에서는 recovery 없이 다시 IDLE로 복귀
            time.sleep(0.5)
            # self._set_state(WorkflowState.IDLE)
            self.is_running = False

    # ------------------------------------------------------------------
    # Context / state helpers
    # ------------------------------------------------------------------
    def _reset_order_context(self):
        self.items_scanned = []
        self.items_final = []
        self.packing_plan = []
        self.last_success_step = ""
        self.last_error = ""

    def _set_state(self, new_state: WorkflowState):
        if self.state != new_state:
            self.get_logger().info(f"[STATE] {self.state.name} -> {new_state.name}")
            self.state = new_state

    # 액션, 서비스는 즉시 결과가 오는게 아니라 future로 돌아온다. future.done이 될때까지 기다리는 공통 헬퍼
    def _wait_future(self, future, timeout_sec: Optional[float] = None):
        start = time.time()

        while rclpy.ok() and not future.done():
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                raise TimeoutError("future wait timeout")
            time.sleep(0.05)

        return future.result()

    # ------------------------------------------------------------------
    # Interface wrappers. 통신 주고 받은것에서 필요한것만 명확히 정의하는 함수로 변화.
    # ------------------------------------------------------------------
    def call_voice(
        self,
        mode: str,
        items: Optional[List[Item]] = None
    ) -> Tuple[bool, List[Item]]:
        if not self.voice_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("voice action server not available: /voice/run_session")

        goal = VoiceSession.Goal()
        goal.mode = mode
        goal.items_in = items if items is not None else []

        send_future = self.voice_client.send_goal_async(
            goal,
            feedback_callback=self._voice_feedback_callback
        )
        goal_handle = self._wait_future(send_future)

        if not goal_handle.accepted:
            raise RuntimeError(f"voice goal rejected: mode={mode}")

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future).result

        # 제품확인용
        self.get_logger().info(
        f"[VOICE RES] success={result.success}, command={result.command}, items_out={self._item_names(result.items_out)}"
        )   

        return result.success, list(result.items_out)

    def call_vision(self) -> Tuple[bool, List[Item]]:
        if not self.vision_client.wait_for_server(timeout_sec=200.0):
            raise RuntimeError("vision action server not available: /vision/scan_items")

        goal = ScanItems.Goal()
        goal.start_vision = True

        send_future = self.vision_client.send_goal_async(
            goal,
            feedback_callback=self._vision_feedback_callback
        )
        goal_handle = self._wait_future(send_future)

        if not goal_handle.accepted:
            raise RuntimeError("vision goal rejected")

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future).result
        
        self.get_logger().info(
        f"[VISION RES] success={result.success}, items_scan={self._item_names(result.items_scan)}"
        )   

        return result.success, list(result.items_scan)

    def call_plan_packing(self, items: List[Item]) -> Tuple[bool, List[Placement]]:
        if not self.plan_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("plan service not available: /plan_packing/compute_plan")

        request = ComputePackingPlan.Request()
        request.items = items

        future = self.plan_client.call_async(request)
        response = self._wait_future(future, timeout_sec=30.0)

        self.get_logger().info(
            f"[PLAN RES] success={response.success}, placements={len(response.placements)}"
        )    

        return response.success, list(response.placements)

    def call_execute_packing(
        self,
        items: List[Item],
        plan: List[Placement]
    ) -> bool:
        if not self.execute_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("execute action server not available: /execute_packing/placing")

        goal = ExecutePacking.Goal()
        goal.pick_items = items
        goal.place_items = plan

        send_future = self.execute_client.send_goal_async(
            goal,
            feedback_callback=self._execute_feedback_callback
        )
        goal_handle = self._wait_future(send_future)

        if not goal_handle.accepted:
            raise RuntimeError("execute packing goal rejected")

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future).result

        return result.success

    # ------------------------------------------------------------------
    # Feedback callbacks 진행상황을 로그로 보기 좋게 하기 위해. (필수 아님)
    # ------------------------------------------------------------------
    def _voice_feedback_callback(self, feedback_msg):
        self.get_logger().debug(
            f"[VOICE] runtime={feedback_msg.feedback.runtime}s"
        )

    def _vision_feedback_callback(self, feedback_msg):
        self.get_logger().debug(
            f"[VISION] runtime={feedback_msg.feedback.runtime}s"
        )

    def _execute_feedback_callback(self, feedback_msg):
        self.get_logger().debug(
            f"[EXECUTE] progress={feedback_msg.feedback.progress}, "
            f"runtime={feedback_msg.feedback.runtime}s"
        )


def main(args=None):
    rclpy.init(args=args)
    node = WorkflowNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("workflow_node stopped by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()