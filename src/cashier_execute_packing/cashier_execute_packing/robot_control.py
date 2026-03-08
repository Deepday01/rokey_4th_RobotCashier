#!/usr/bin/env python3

import time
import rclpy
import DR_init

from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from robot_msgs.action import ExecutePacking
from robot_control.onrobot import RG

from cashier_execute_packing.packing_executor import (
    build_execution_plan_from_request,
    execute_plan_with_callbacks,
)

# ==============================
# Robot config
# ==============================

ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

VELOCITY = 100
ACC = 100

SAFE_Z = 300.0
READY_J = [0, 0, 90, 0, 90, 0]

# ==============================
# Gripper
# ==============================

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


# ==============================
# Robot helpers
# ==============================

def initialize_robot():
    from DSR_ROBOT2 import (
        set_tool,
        set_tcp,
        set_robot_mode,
        ROBOT_MODE_MANUAL,
        ROBOT_MODE_AUTONOMOUS,
    )

    set_robot_mode(ROBOT_MODE_MANUAL)
    set_tool(ROBOT_TOOL)
    set_tcp(ROBOT_TCP)
    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    time.sleep(1)


def move_ready():
    from DSR_ROBOT2 import movej
    movej(READY_J, vel=100, acc=100)


def move_to_pose(pose):
    from DSR_ROBOT2 import movel, posx, mwait

    movel(
        posx(
            [
                pose.x,
                pose.y,
                pose.z,
                pose.roll,
                pose.pitch,
                pose.yaw,
            ]
        ),
        vel=VELOCITY,
        acc=ACC,
    )

    mwait()


def safe_rise():
    from DSR_ROBOT2 import get_current_posx

    cur, _ = get_current_posx()

    move_to_pose(
        type("Pose", (), {
            "x": cur[0],
            "y": cur[1],
            "z": SAFE_Z,
            "roll": cur[3],
            "pitch": cur[4],
            "yaw": cur[5],
        })()
    )


def open_gripper(node):
    from DSR_ROBOT2 import mwait

    node.get_logger().info("[GRIPPER] open")

    gripper.open_gripper()

    t0 = time.time()
    while gripper.get_status()[0]:
        if time.time() - t0 > 5:
            raise RuntimeError("gripper open timeout")
        time.sleep(0.1)

    mwait()


def close_gripper(node):
    from DSR_ROBOT2 import mwait

    node.get_logger().info("[GRIPPER] close")

    gripper.close_gripper()

    t0 = time.time()
    while gripper.get_status()[0]:
        if time.time() - t0 > 5:
            raise RuntimeError("gripper close timeout")
        time.sleep(0.1)

    mwait()


# ==============================
# Executor callbacks
# ==============================

def execute_pick_and_stage_on_rotation_station(executor, task, stage_plan):
    logger = executor.get_logger()

    # 1. 원위치 접근
    executor.robot.move_to_pose(stage_plan.pick_approach_pose)

    # 2. pick 위치 이동
    executor.robot.move_to_pose(stage_plan.pick_pose)

    # 3. 그리퍼로 물체 grasp
    executor.robot.gripper_open()
    executor.robot.gripper_close(task.grip_width)

    # 4. pick 후 상승
    executor.robot.move_to_pose(stage_plan.pick_retreat_pose)

    # 5. 회전 스테이션 접근
    executor.robot.move_to_pose(stage_plan.station_approach_pose)

    # 6. 회전 스테이션 위에 place
    executor.robot.move_to_pose(stage_plan.station_place_pose)
    executor.robot.gripper_open()

    # 7. 회전 스테이션에서 이탈
    executor.robot.move_to_pose(stage_plan.station_retreat_pose)


def execute_align_object_on_rotation_station(executor, task, align_plan):
    logger = executor.get_logger()
    logger.info(
        f"[TASK {task.task_index}] "
        "execute_align_object_on_rotation_station start"
    )

    if not getattr(align_plan, "required", False):
        logger.info(
            f"[TASK {task.task_index}] "
            "align skip: alignment not required"
        )
        return

    # 예시:
    # align_plan.steps = [
    #     {"rx": 90.0, "ry": 0.0, "rz": 0.0},
    #     {"rx": 0.0, "ry": 0.0, "rz": 90.0},
    # ]

    for step_index, step in enumerate(align_plan.steps, start=1):
        rx = step.get("rx", 0.0)
        ry = step.get("ry", 0.0)
        rz = step.get("rz", 0.0)

        logger.info(
            f"[TASK {task.task_index}] "
            f"[ALIGN STEP {step_index}] "
            f"rx={rx}, ry={ry}, rz={rz}"
        )

        executor.rotate_object_on_rotation_station(
            rx_deg=rx,
            ry_deg=ry,
            rz_deg=rz,
        )

    logger.info(
        f"[TASK {task.task_index}] "
        "execute_align_object_on_rotation_station done"
    )

def execute_pick_and_place_to_box(executor, task, box_plan):
    logger = executor.get_logger()
    logger.info(
        f"[TASK {task.task_index}] "
        "execute_pick_and_place_to_box start"
    )

    # 1. 회전 스테이션 위 물체 다시 pick
    executor.robot.move_to_pose(box_plan.station_pick_approach_pose)
    executor.robot.move_to_pose(box_plan.station_pick_pose)
    executor.robot.gripper_close(task.grip_width)
    executor.robot.move_to_pose(box_plan.station_pick_retreat_pose)

    # 2. 박스 접근
    executor.robot.move_to_pose(box_plan.box_approach_pose)

    # 3. 박스 내 최종 배치 위치로 이동
    executor.robot.move_to_pose(box_plan.box_place_pose)

    # 4. 물체 release
    executor.robot.gripper_open()

    # 5. 박스에서 안전 이탈
    executor.robot.move_to_pose(box_plan.box_retreat_pose)

    logger.info(
        f"[TASK {task.task_index}] "
        "execute_pick_and_place_to_box done"
    )

# def execute_rotation_step(node, task, step_index, step):
#     """
#     회전 단계 실행
#     """
#     node.get_logger().info(
#         f"[ROTATION] task={task.task_index} step={step_index} axis={step.axis}"
#     )

#     item_after = step.item_after

#     safe_rise()

#     move_to_pose(item_after.pose)


# def execute_pick_plan(node, task, pick_plan):
#     node.get_logger().info(
#         f"[PICK] task={task.task_index} grip_width={pick_plan.grip_width}"
#     )

#     safe_rise()

#     move_to_pose(pick_plan.approach_pose)

#     open_gripper(node)

#     move_to_pose(pick_plan.pick_pose)

#     close_gripper(node)

#     safe_rise()




# def execute_place_plan(node, task, place_plan):
#     node.get_logger().info(f"[PLACE] task={task.task_index}")

#     safe_rise()

#     move_to_pose(place_plan.approach_pose)

#     move_to_pose(place_plan.place_pose)

#     open_gripper(node)

#     safe_rise()

# ==============================
# Action server
# ==============================

class ExecutePackingServer(Node):

    def __init__(self):
        super().__init__("execute_packing_server", namespace=ROBOT_ID)

        self._busy = False

        self._server = ActionServer(
            self,
            ExecutePacking,
            "/execute_packing/placing",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("ExecutePacking ActionServer ready")

    # ------------------------------

    def goal_callback(self, goal_request):

        if self._busy:
            self.get_logger().warn("Reject goal: robot busy")
            return GoalResponse.REJECT

        return GoalResponse.ACCEPT

    # ------------------------------

    def cancel_callback(self, goal_handle):

        self.get_logger().warn("Cancel requested")

        return CancelResponse.ACCEPT

    # ------------------------------

    def execute_callback(self, goal_handle):

        self._busy = True

        result = ExecutePacking.Result()

        try:

            req = goal_handle.request

            self.get_logger().info("Build execution plan")

            # 플랜 요청
            plan = build_execution_plan_from_request(
                pick_items=req.pick_items,
                place_items=req.place_items,
                logger=self.get_logger(),
            )

            move_ready()

            def _on_pick_and_stage_on_rotation_station(task, stage_plan):
                execute_pick_and_stage_on_rotation_station(self, task, stage_plan)


            def _on_align_object_on_rotation_station(task, align_plan):
                execute_align_object_on_rotation_station(self, task, align_plan)


            def _on_pick_and_place_to_box(task, box_plan):
                execute_pick_and_place_to_box(self, task, box_plan)


            execute_plan_with_callbacks(
                plan=plan,
                on_pick_and_stage_on_rotation_station=_on_pick_and_stage_on_rotation_station,
                on_align_object_on_rotation_station=_on_align_object_on_rotation_station,
                on_pick_and_place_to_box=_on_pick_and_place_to_box,
                logger=self.get_logger(),
            )

            # execute_plan_with_callbacks(
            #     plan=plan,
            #     on_pick_plan=lambda task, pick_plan: execute_pick_plan(
            #         self, task, pick_plan
            #     ),
            #     on_rotation_step=lambda task, i, step: execute_rotation_step(
            #         self, task, i, step
            #     ),
            #     on_place_plan=lambda task, place_plan: execute_place_plan(
            #         self, task, place_plan
            #     ),
            #     logger=self.get_logger(),
            # )

            goal_handle.succeed()

            result.success = True

            return result

        except Exception as e:

            self.get_logger().error(f"[TASK FAILED] {e}")

            goal_handle.abort()

            result.success = False

            return result

        finally:

            self._busy = False


# ==============================
# main
# ==============================

def main(args=None):

    rclpy.init(args=args)

    node = ExecutePackingServer()

    DR_init.__dsr__node = node

    try:

        initialize_robot()

        rclpy.spin(node)

    except KeyboardInterrupt:

        print("Interrupted")

    finally:

        node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":
    main()