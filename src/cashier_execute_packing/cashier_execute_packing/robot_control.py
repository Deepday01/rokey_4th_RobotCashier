#!/usr/bin/env python3

import time
import rclpy
import DR_init

from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse

from robot_msgs.action import ExecutePacking
from robot_msgs.msg import Item, Placement
from robot_control.onrobot import RG


GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


# ==============================
# Project / Robot config
# ==============================
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

DEFAULT_YAW = 62.158
DEFAULT_PITCH = 179.976
DEFAULT_ROLL = 62.496

VELOCITY = 100
ACC = 100

InitJReady = [0, 0, 90, 0, 90, 0]
SAFE_RISE_HEIGHT = 300.0


# ==============================
# Robot hooks
# ==============================
def initialize_robot():
    from DSR_ROBOT2 import (
        set_tool,
        set_tcp,
        ROBOT_MODE_MANUAL,
        ROBOT_MODE_AUTONOMOUS,
        set_robot_mode,
    )

    set_robot_mode(ROBOT_MODE_MANUAL)
    set_tool(ROBOT_TOOL)
    set_tcp(ROBOT_TCP)
    set_robot_mode(ROBOT_MODE_AUTONOMOUS)
    time.sleep(1)


def move_ready():
    from DSR_ROBOT2 import movej
    movej(InitJReady, vel=100, acc=100)


def move_to_pose(x, y, z, yaw=DEFAULT_YAW, pitch=DEFAULT_PITCH, roll=DEFAULT_ROLL):
    from DSR_ROBOT2 import movel, posx
    movel(posx([x, y, z, yaw, pitch, roll]), vel=VELOCITY, acc=ACC)


def open_gripper(self, timeout=5.0):
    from DSR_ROBOT2 import mwait

    gripper.open_gripper()

    t0 = time.time()
    while gripper.get_status()[0]:
        if time.time() - t0 > timeout:
            raise RuntimeError("open_gripper timeout")
        time.sleep(0.1)

    mwait()
    self.get_logger().info("[GRIPPER] open complete")


def close_gripper(self, timeout=5.0):
    from DSR_ROBOT2 import mwait

    gripper.close_gripper()

    t0 = time.time()
    while gripper.get_status()[0]:
        if time.time() - t0 > timeout:
            raise RuntimeError("close_gripper timeout")
        time.sleep(0.1)

    mwait()
    self.get_logger().info("[GRIPPER] close complete")


def pick_object(self, item: Item):
    from DSR_ROBOT2 import get_current_posx, mwait

    cur_pos, _ = get_current_posx()
    move_to_pose(
        cur_pos[0],
        cur_pos[1],
        SAFE_RISE_HEIGHT,
        cur_pos[3],
        cur_pos[4],
        cur_pos[5],
    )
    mwait()

    cur_pos, _ = get_current_posx()
    move_to_pose(
        item.x,
        item.y,
        cur_pos[2],
        cur_pos[3],
        cur_pos[4],
        cur_pos[5],
    )
    mwait()

    open_gripper(self)

    move_to_pose(
        item.x,
        item.y,
        item.z,
        item.yaw,
        item.pitch,
        item.roll,
    )
    mwait()

    close_gripper(self)

    cur_pos, _ = get_current_posx()
    move_to_pose(
        cur_pos[0],
        cur_pos[1],
        SAFE_RISE_HEIGHT,
        cur_pos[3],
        cur_pos[4],
        cur_pos[5],
    )
    mwait()

    self.get_logger().info(f"[PICK] complete: {item.name}")


def place_object(self, placement: Placement):
    from DSR_ROBOT2 import get_current_posx, mwait

    cur_pos, _ = get_current_posx()
    move_to_pose(
        cur_pos[0],
        cur_pos[1],
        SAFE_RISE_HEIGHT,
        cur_pos[3],
        cur_pos[4],
        cur_pos[5],
    )
    mwait()

    cur_pos, _ = get_current_posx()
    move_to_pose(
        placement.x,
        placement.y,
        SAFE_RISE_HEIGHT,
        cur_pos[3],
        cur_pos[4],
        cur_pos[5],
    )
    mwait()

    move_to_pose(
        placement.x,
        placement.y,
        placement.z,
        placement.yaw,
        placement.pitch,
        placement.roll,
    )
    mwait()

    open_gripper(self)

    move_to_pose(
        placement.x,
        placement.y,
        SAFE_RISE_HEIGHT,
        placement.yaw,
        placement.pitch,
        placement.roll,
    )
    mwait()

    self.get_logger().info(
        f"[PLACE] complete: index={placement.object_index}"
    )


def finish_task(self):
    from DSR_ROBOT2 import get_current_posx, movej, mwait

    cur_pos, _ = get_current_posx()
    move_to_pose(
        cur_pos[0],
        cur_pos[1],
        SAFE_RISE_HEIGHT,
        cur_pos[3],
        cur_pos[4],
        cur_pos[5],
    )
    mwait()

    movej(InitJReady, vel=100, acc=100)
    mwait()

    self.get_logger().info("[TASK] robot returned to ready position")


# ==============================
# Action Server
# ==============================
class ExecutePackingActionServer(Node):
    def __init__(self):
        super().__init__("execute_packing_action_server", namespace=ROBOT_ID)

        self._busy = False

        self._action_server = ActionServer(
            self,
            ExecutePacking,
            "/execute_packing/placing",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info("ActionServer ready: /execute_packing/placing")

    def goal_callback(self, goal_request):
        if self._busy:
            self.get_logger().warn("Reject goal: busy")
            return GoalResponse.REJECT

        if goal_request is None:
            self.get_logger().warn("Reject goal: empty request")
            return GoalResponse.REJECT

        if len(goal_request.pick_items) == 0:
            self.get_logger().warn("Reject goal: no pick_items")
            return GoalResponse.REJECT

        if len(goal_request.place_items) == 0:
            self.get_logger().warn("Reject goal: no place_items")
            return GoalResponse.REJECT

        self.get_logger().info(
            f"Goal accepted: pick_items={len(goal_request.pick_items)}, "
            f"place_items={len(goal_request.place_items)}"
        )
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().warn("Cancel requested")
        return CancelResponse.ACCEPT

    def publish_feedback(self, goal_handle, current_index: int, total: int, start_time: float):
        fb = ExecutePacking.Feedback()
        fb.progress = f"{current_index}/{total}"
        fb.runtime = int(time.time() - start_time)
        goal_handle.publish_feedback(fb)

    def execute_callback(self, goal_handle):
        self._busy = True
        start_time = time.time()

        result = ExecutePacking.Result()

        try:
            req = goal_handle.request
            pick_items = req.pick_items
            place_items = req.place_items

            total = len(place_items)
            self.get_logger().info(f"[TASK] start: total={total}")

            move_ready()
            self.publish_feedback(goal_handle, 0, total, start_time)

            for i, placement in enumerate(place_items):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    self.get_logger().warn(f"Canceled before task {i}")
                    return result

                obj_index = placement.object_index
                if obj_index < 0 or obj_index >= len(pick_items):
                    raise RuntimeError(f"Invalid object_index: {obj_index}")

                item = pick_items[obj_index]

                self.get_logger().info(
                    f"[TASK {i+1}/{total}] start: item={item.name}, object_index={obj_index}"
                )

                pick_object(self, item)

                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.success = False
                    self.get_logger().warn(f"Canceled after pick of task {i}")
                    return result

                place_object(self, placement)

                self.publish_feedback(goal_handle, i + 1, total, start_time)

                self.get_logger().info(
                    f"[TASK {i+1}/{total}] done: item={item.name}, object_index={obj_index}"
                )

            finish_task(self)

            elapsed = time.time() - start_time
            self.get_logger().info(f"[TASK] success: elapsed={elapsed:.2f}s")

            goal_handle.succeed()
            result.success = True
            return result

        except Exception as e:
            self.get_logger().error(f"[TASK] failed: {e}")
            goal_handle.abort()
            result.success = False
            return result

        finally:
            self._busy = False


def main(args=None):
    rclpy.init(args=args)

    node = ExecutePackingActionServer()

    DR_init.__dsr__node = node
    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    try:
        initialize_robot()
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()






