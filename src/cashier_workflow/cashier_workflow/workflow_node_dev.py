#!/usr/bin/env python3
# workflow_node_dev.py


# 260311 : 복구 모드 구성중


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
    RECOVERY = auto()


class WorkflowNode(Node):
    def __init__(self):
        super().__init__("workflow_node")

        # ---- parameters ----
        self.declare_parameter("debug_mode", False)
        self.declare_parameter("max_recovery_attempts", 100)
        self.debug_mode = self.get_parameter("debug_mode").value
        self.max_recovery_attempts = int(
            self.get_parameter("max_recovery_attempts").value
        )

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
        self.recovery_attempts = 0
        self.recovery_exhausted_logged = False

        # ---- per-order context ----
        self.items_scanned: List[Item] = []
        self.items_final: List[Item] = []
        self.packing_plan: List[Placement] = []

        # IDLE -> 새 주문 1회 시작
        # ERROR -> RECOVERY 1회 시도
        # DONE -> 문맥 정리 후 IDLE 복귀
        self.create_timer(1.0, self._tick)
        self.get_logger().info(
            f"workflow_node ready. state=IDLE, max_recovery_attempts={self.max_recovery_attempts}"
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _item_names(self, items):
        return [item.name for item in items]

    def _dbg(self, msg: str):
        if self.debug_mode:
            self.get_logger().info(f"[DEBUG] {msg}")

    def _log_context_summary(self, prefix: str):
        self.get_logger().info(
            f"{prefix} scanned={len(self.items_scanned)}, "
            f"final={len(self.items_final)}, "
            f"plan={len(self.packing_plan)}, "
            f"last_success_step={self.last_success_step}"
        )

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------
    def _tick(self):
        if self.is_running:
            return

        if self.state == WorkflowState.IDLE:
            self.is_running = True
            self.recovery_attempts = 0
            self.recovery_exhausted_logged = False
            self._reset_order_context()
            threading.Thread(
                target=self._run_from_state,
                args=(WorkflowState.VOICE_WAKEUP, False),
                daemon=True,
            ).start()
            return

        if self.state == WorkflowState.ERROR:
            if self.recovery_attempts < self.max_recovery_attempts:
                self.is_running = True
                threading.Thread(target=self._run_recovery, daemon=True).start()
            elif not self.recovery_exhausted_logged:
                self.recovery_exhausted_logged = True
                self.get_logger().error(
                    "recovery attempts exhausted. keep ERROR state for operator check"
                )
            return

        if self.state == WorkflowState.DONE:
            self._log_context_summary("[DONE] clearing context:")
            self._reset_order_context()
            self._set_state(WorkflowState.IDLE)
            return

    # ------------------------------------------------------------------
    # Workflow runners
    # ------------------------------------------------------------------
    def _run_recovery(self):
        self._set_state(WorkflowState.RECOVERY)
        self.recovery_attempts += 1

        resume_state = self._get_resume_state()
        self.get_logger().warn(
            f"[RECOVERY] attempt={self.recovery_attempts}/{self.max_recovery_attempts}, "
            f"resume={resume_state.name}, last_error={self.last_error}"
        )
        self._log_context_summary("[RECOVERY] context:")

        self._run_from_state(resume_state, is_recovery=True)

    def _run_from_state(self, start_state: WorkflowState, is_recovery: bool = False):
        step_order = [
            WorkflowState.VOICE_WAKEUP,
            WorkflowState.VISION,
            WorkflowState.VOICE_EDIT,
            WorkflowState.PLAN_PACKING,
            WorkflowState.EXECUTE_PACKING,
        ]

        try:
            if start_state == WorkflowState.DONE:
                self._set_state(WorkflowState.DONE)
                self.get_logger().info("workflow already completed. move to DONE")
                return

            start_index = step_order.index(start_state)
            self._dbg(
                f"[RUN] start_state={start_state.name}, is_recovery={is_recovery}, "
                f"last_success_step={self.last_success_step}"
            )

            for step in step_order[start_index:]:
                if step == WorkflowState.VOICE_WAKEUP:
                    self._run_voice_wakeup()
                elif step == WorkflowState.VISION:
                    self._run_vision()
                elif step == WorkflowState.VOICE_EDIT:
                    self._run_voice_edit()
                elif step == WorkflowState.PLAN_PACKING:
                    self._run_plan_packing()
                elif step == WorkflowState.EXECUTE_PACKING:
                    self._run_execute_packing()

            self._set_state(WorkflowState.DONE)
            self.get_logger().info("workflow completed successfully")

        except Exception as e:
            self.last_error = str(e)
            self._set_state(WorkflowState.ERROR)
            self.get_logger().error(f"workflow failed: {self.last_error}")

        finally:
            time.sleep(0.5)
            self.is_running = False

    # ------------------------------------------------------------------
    # Step runners
    # ------------------------------------------------------------------
    def _run_voice_wakeup(self):
        self._set_state(WorkflowState.VOICE_WAKEUP)
        success, _ = self.call_voice(mode="WAKEUP")
        if not success:
            raise RuntimeError("voice wakeup failed")
        self.last_success_step = "VOICE_WAKEUP"

    def _run_vision(self):
        self._set_state(WorkflowState.VISION)
        success, scanned_items = self.call_vision()
        if not success or not scanned_items:
            raise RuntimeError("vision scan failed or returned no items")
        self.items_scanned = scanned_items
        self.last_success_step = "VISION"

    def _run_voice_edit(self):
        self._set_state(WorkflowState.VOICE_EDIT)
        success, final_items = self.call_voice(
            mode="EDIT",
            items=self.items_scanned
        )
        if not success or not final_items:
            raise RuntimeError("voice edit failed or returned no items")
        self.items_final = final_items
        self.last_success_step = "VOICE_EDIT"

    def _run_plan_packing(self):
        self._set_state(WorkflowState.PLAN_PACKING)
        success, placements = self.call_plan_packing(self.items_final)
        if not success or not placements:
            raise RuntimeError("packing plan failed or returned no placements")

        if len(placements) != len(self.items_final):
            raise RuntimeError("items_final and packing_plan size mismatch")

        self.packing_plan = placements
        self.last_success_step = "PLAN_PACKING"

    def _run_execute_packing(self):
        self._set_state(WorkflowState.EXECUTE_PACKING)
        success = self.call_execute_packing(
            self.items_final,
            self.packing_plan
        )
        if not success:
            raise RuntimeError("execute packing failed")
        self.last_success_step = "EXECUTE_PACKING"

    # ------------------------------------------------------------------
    # Recovery / context helpers
    # ------------------------------------------------------------------
    def _reset_order_context(self):
        self.items_scanned = []
        self.items_final = []
        self.packing_plan = []
        self.last_success_step = ""
        self.last_error = ""

    def _get_resume_state(self) -> WorkflowState:
        mapping = {
            "": WorkflowState.VOICE_WAKEUP,
            "VOICE_WAKEUP": WorkflowState.VISION,
            "VISION": WorkflowState.VOICE_EDIT,
            "VOICE_EDIT": WorkflowState.PLAN_PACKING,
            "PLAN_PACKING": WorkflowState.EXECUTE_PACKING,
            "EXECUTE_PACKING": WorkflowState.DONE,
        }
        candidate = mapping.get(self.last_success_step, WorkflowState.VOICE_WAKEUP)

        # 문맥 검증 후 안전하게 복귀 단계 보정
        if candidate == WorkflowState.VOICE_EDIT and not self.items_scanned:
            return WorkflowState.VISION

        if candidate == WorkflowState.PLAN_PACKING:
            if not self.items_final:
                return WorkflowState.VOICE_EDIT if self.items_scanned else WorkflowState.VISION

        if candidate == WorkflowState.EXECUTE_PACKING:
            if not self.items_final:
                return WorkflowState.VOICE_EDIT if self.items_scanned else WorkflowState.VISION
            if not self.packing_plan or len(self.packing_plan) != len(self.items_final):
                return WorkflowState.PLAN_PACKING

        return candidate

    def _set_state(self, new_state: WorkflowState):
        if self.state != new_state:
            self.get_logger().info(f"[STATE] {self.state.name} -> {new_state.name}")
            self.state = new_state

    # 액션, 서비스는 즉시 결과가 오는게 아니라 future로 돌아온다.
    # future.done()이 될 때까지 기다리는 공통 헬퍼
    def _wait_future(self, future, timeout_sec: Optional[float] = None):
        start = time.time()

        while rclpy.ok() and not future.done():
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                raise TimeoutError("future wait timeout")
            time.sleep(0.05)

        return future.result()

    # ------------------------------------------------------------------
    # Interface wrappers
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

        self._dbg(f"[VOICE TX] mode={goal.mode}, items_in={self._item_names(goal.items_in)}")

        send_future = self.voice_client.send_goal_async(
            goal,
            feedback_callback=self._voice_feedback_callback
        )
        goal_handle = self._wait_future(send_future)

        if not goal_handle.accepted:
            raise RuntimeError(f"voice goal rejected: mode={mode}")

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future).result

        self._dbg(f"[VOICE RX] success={result.success}, command={result.command}")
        return result.success, list(result.items_out)

    def call_vision(self) -> Tuple[bool, List[Item]]:
        if not self.vision_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError("vision action server not available: /vision/scan_items")

        goal = ScanItems.Goal()
        goal.start_vision = True

        self._dbg(f"[VISION TX] start_vision={goal.start_vision}")

        send_future = self.vision_client.send_goal_async(
            goal,
            feedback_callback=self._vision_feedback_callback
        )
        goal_handle = self._wait_future(send_future)

        if not goal_handle.accepted:
            raise RuntimeError("vision goal rejected")

        result_future = goal_handle.get_result_async()
        self._dbg("[VISION] waiting result future...")
        result = self._wait_future(result_future).result

        self._dbg(f"[VISION RX] success={result.success}, items_scan={result.items_scan}")
        return result.success, list(result.items_scan)

    def call_plan_packing(self, items: List[Item]) -> Tuple[bool, List[Placement]]:
        if not self.plan_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("plan service not available: /plan_packing/compute_plan")

        request = ComputePackingPlan.Request()
        request.items = items

        self._dbg(f"[PLAN TX] items={self._item_names(request.items)}")

        future = self.plan_client.call_async(request)
        response = self._wait_future(future, timeout_sec=30.0)

        self._dbg(f"[PLAN RX] success={response.success}, placement={response.placements}")
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

        self._dbg(
            f"[EXECUTE TX] pick_items={self._item_names(goal.pick_items)}, "
            f"place={goal.place_items}"
        )

        send_future = self.execute_client.send_goal_async(
            goal,
            feedback_callback=self._execute_feedback_callback
        )
        goal_handle = self._wait_future(send_future)

        if not goal_handle.accepted:
            raise RuntimeError("execute packing goal rejected")

        result_future = goal_handle.get_result_async()
        result = self._wait_future(result_future).result

        self._dbg(
            f"[EXECUTE RX] success={result.success}, "
            f"pick_count={len(goal.pick_items)}, place_count={len(goal.place_items)}"
        )
        return result.success

    # ------------------------------------------------------------------
    # Feedback callbacks
    # ------------------------------------------------------------------
    def _voice_feedback_callback(self, feedback_msg):
        self._dbg(f"[VOICE] runtime={feedback_msg.feedback.runtime}s")

    def _vision_feedback_callback(self, feedback_msg):
        self._dbg(f"[VISION] runtime={feedback_msg.feedback.runtime}s")

    def _execute_feedback_callback(self, feedback_msg):
        self._dbg(
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