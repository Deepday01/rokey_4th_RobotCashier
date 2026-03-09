#!/usr/bin/env python3

import time

import DR_init
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node

from robot_msgs.action import ExecutePacking

from cashier_execute_packing.config import (
    ACC,
    GRIPPER_NAME,
    GRIPPER_TIMEOUT_SEC,
    POLL_INTERVAL_SEC,
    READY_J,
    ROBOT_ID,
    ROBOT_MODEL,
    ROBOT_TCP,
    ROBOT_TOOL,
    SAFE_Z,
    TOOLCHARGER_IP,
    TOOLCHARGER_PORT,
    VELOCITY,
)
from cashier_execute_packing.packing_executor import (
    build_execution_plan_from_request,
    execute_plan_with_callbacks,
)

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

from robot_control.onrobot import RG


gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


class RobotController:
    def initialize(self) -> None:
        from DSR_ROBOT2 import (
            ROBOT_MODE_AUTONOMOUS,
            ROBOT_MODE_MANUAL,
            set_robot_mode,
            set_tcp,
            set_tool,
        )

        set_robot_mode(ROBOT_MODE_MANUAL)
        set_tool(ROBOT_TOOL)
        set_tcp(ROBOT_TCP)
        set_robot_mode(ROBOT_MODE_AUTONOMOUS)
        time.sleep(1)

    def move_ready(self) -> None:
        from DSR_ROBOT2 import movej

        movej(READY_J, vel=VELOCITY, acc=ACC)

    def move_to_pose(self, pose) -> None:
        from DSR_ROBOT2 import movel, mwait, posx

        movel(
            posx([pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]),
            vel=VELOCITY,
            acc=ACC,
        )
        mwait()

    def safe_rise(self) -> None:
        from DSR_ROBOT2 import get_current_posx

        current, _ = get_current_posx()
        self.move_to_pose(
            type("Pose", (), {
                "x": current[0],
                "y": current[1],
                "z": SAFE_Z,
                "roll": current[3],
                "pitch": current[4],
                "yaw": current[5],
            })()
        )

    def open_gripper(self) -> None:
        from DSR_ROBOT2 import mwait

        gripper.open_gripper()
        started = time.time()
        while gripper.get_status()[0]:
            if time.time() - started > GRIPPER_TIMEOUT_SEC:
                raise RuntimeError("gripper open timeout")
            time.sleep(POLL_INTERVAL_SEC)
        mwait()

    def close_gripper(self) -> None:
        from DSR_ROBOT2 import mwait

        gripper.close_gripper()
        started = time.time()
        while gripper.get_status()[0]:
            if time.time() - started > GRIPPER_TIMEOUT_SEC:
                raise RuntimeError("gripper close timeout")
            time.sleep(POLL_INTERVAL_SEC)
        mwait()


class ExecutePackingServer(Node):
    def __init__(self):
        super().__init__("execute_packing_server", namespace=ROBOT_ID)
        self._busy = False
        self.robot = RobotController()
        self._server = ActionServer(
            self,
            ExecutePacking,
            "/execute_packing/placing",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

    def goal_callback(self, goal_request):
        self.get_logger().info("goal_callback()!!!")

        if self._busy:
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("cancel_callback()!!!")

        return CancelResponse.ACCEPT

    def rotate_object_on_rotation_station(self, rx_deg: float = 0.0, ry_deg: float = 0.0, rz_deg: float = 0.0) -> None:
        return None

    def execute_pick_and_stage_on_rotation_station(self, task, stage_plan) -> None:
        self.robot.move_to_pose(stage_plan.pick_approach_pose)
        self.robot.open_gripper()
        self.robot.move_to_pose(stage_plan.pick_pose)
        self.robot.close_gripper()
        self.robot.move_to_pose(stage_plan.pick_retreat_pose)
        self.robot.move_to_pose(stage_plan.station_approach_pose)
        self.robot.move_to_pose(stage_plan.station_place_pose)
        self.robot.open_gripper()
        self.robot.move_to_pose(stage_plan.station_retreat_pose)

    def execute_align_object_on_rotation_station(self, task, align_plan) -> None:
        if not align_plan.required:
            return
        for step in align_plan.steps:
            self.rotate_object_on_rotation_station(
                rx_deg=step.rx_deg,
                ry_deg=step.ry_deg,
                rz_deg=step.rz_deg,
            )

    def execute_pick_and_place_to_box(self, task, box_plan) -> None:
        self.robot.move_to_pose(box_plan.station_pick_approach_pose)
        self.robot.move_to_pose(box_plan.station_pick_pose)
        self.robot.close_gripper()
        self.robot.move_to_pose(box_plan.station_pick_retreat_pose)
        self.robot.move_to_pose(box_plan.box_approach_pose)
        self.robot.move_to_pose(box_plan.box_place_pose)
        self.robot.open_gripper()
        self.robot.move_to_pose(box_plan.box_retreat_pose)

    def execute_callback(self, goal_handle):
        self.get_logger().info("execute_callback()!!!")

        self._busy = True
        result = ExecutePacking.Result()
        try:

            request = goal_handle.request
            plan = build_execution_plan_from_request(
                pick_items=request.pick_items,
                place_items=request.place_items,
            )




            # 디버깅 시작

            self.get_logger().info("===== Execution Plan Debug =====")

            self.get_logger().info(f"task_count={len(plan.tasks)}")

            for task in plan.tasks:

                item_pose = task.item.pose
                place_pose = task.placement.pose

                station_place = task.stage_plan.station_place_pose
                station_pick = task.box_plan.station_pick_pose
                box_place = task.box_plan.box_place_pose

                self.get_logger().info(
                    f"[TASK {task.task_index}] item={task.item.name} "
                    f"grip_width={task.grip_width}"
                )

                self.get_logger().info(
                    f"  pick_pose=({item_pose.x:.3f}, {item_pose.y:.3f}, {item_pose.z:.3f}) "
                    f"rpy=({item_pose.roll:.3f}, {item_pose.pitch:.3f}, {item_pose.yaw:.3f})"
                )

                self.get_logger().info(
                    f"  place_pose=({place_pose.x:.3f}, {place_pose.y:.3f}, {place_pose.z:.3f}) "
                    f"rpy=({place_pose.roll:.3f}, {place_pose.pitch:.3f}, {place_pose.yaw:.3f})"
                )

                self.get_logger().info(
                    f"  station_place=({station_place.x:.3f}, {station_place.y:.3f}, {station_place.z:.3f})"
                )

                self.get_logger().info(
                    f"  station_pick=({station_pick.x:.3f}, {station_pick.y:.3f}, {station_pick.z:.3f})"
                )

                self.get_logger().info(
                    f"  box_place=({box_place.x:.3f}, {box_place.y:.3f}, {box_place.z:.3f})"
                )

                self.get_logger().info(
                    f"  align_required={task.align_plan.required}"
                )

                if task.align_plan.required:
                    for i, step in enumerate(task.align_plan.steps, start=1):
                        self.get_logger().info(
                            f"    align_step {i} "
                            f"rx={step.rx_deg} "
                            f"ry={step.ry_deg} "
                            f"rz={step.rz_deg}"
                        )

            self.get_logger().info("===== Execution Plan Debug End =====")


            # 디버깅 종료








            self.robot.move_ready()
            execute_plan_with_callbacks(
                plan=plan,
                on_pick_and_stage_on_rotation_station=self.execute_pick_and_stage_on_rotation_station,
                on_align_object_on_rotation_station=self.execute_align_object_on_rotation_station,
                on_pick_and_place_to_box=self.execute_pick_and_place_to_box,
            )

            goal_handle.succeed()
            result.success = True
            return result
        except Exception as e:

            # 디버깅
            self.get_logger().error(f"[TASK FAILED] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


            goal_handle.abort()
            result.success = False
            return result
        finally:
            self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = ExecutePackingServer()
    DR_init.__dsr__node = node
    try:
        node.robot.initialize()
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
