#!/usr/bin/env python3
import time
import sys

import rclpy
import DR_init

from robot_control.onrobot import RG

# -----------------------------
# Basic config
# -----------------------------
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
VELOCITY = 60
ACC = 60

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# virtual launch일 때 True
SIMULATION_MODE = False

# robot_control.py의 init_robot()와 같은 준비 자세
JREADY = [0, 0, 90, 0, 90, 0]

# 취소 물건을 내려놓을 고정 위치
CANCEL_DROP_POSE = [445.5, -242.6, 174.4, 156.4, 180.0, -112.5]
JDROP = [90, 0, 90, 0, 90, 0]

# 그리퍼 너비 구하는 딕셔너리
GRIPPER_WIDTH = {
    "halls": 530,
    "insect": 630,
    "caramel": 540,
    "candy": 850,
    "cream": 435,
    "eclipse_red": 480,
    "eclipse_gre": 480,
}

_initialized = False
gripper = None
movej = None
movel = None
mwait = None
get_current_posj = None
get_current_posx = None

class DummyGripper:
    def open_gripper(self):
        print("[SIM] open gripper")
    
    def move_gripper(self, width_val, force_val=400):
        print(f"[SIM] move gripper to {width_val}")

    def close_gripper(self):
        print("[SIM] close gripper")

    def get_status(self):
        # [busy, grip_detected] 처럼 사용 중인 코드에 맞춤
        return [0, 1]


def _ensure_initialized():
    global _initialized, gripper, movej, movel, mwait, get_current_posj, get_current_posx

    if _initialized:
        return

    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    if not rclpy.ok():
        rclpy.init(args=None)

    dsr_node = rclpy.create_node("robot_cancel_helper_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
        from DSR_ROBOT2 import movej as _movej, movel as _movel, mwait as _mwait, get_current_posj as _get_current_posj, get_current_posx as _get_current_posx
    except ImportError as e:
        print(f"[robot_cancel_helper] DSR_ROBOT2 import error: {e}")
        sys.exit(1)

    movej = _movej
    movel = _movel
    mwait = _mwait

    if SIMULATION_MODE:
        gripper = DummyGripper()
        print("[robot_cancel_helper] simulation mode: using DummyGripper")
    else:
        gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

    _initialized = True
    get_current_posj = _get_current_posj
    get_current_posx = _get_current_posx


def wait_gripper_done(timeout_sec=5.0):
    if SIMULATION_MODE:
        time.sleep(0.2)
        return True

    start = time.time()
    while True:
        status = gripper.get_status()
        busy = status[0]

        if not busy:
            return True

        if time.time() - start > timeout_sec:
            return False

        time.sleep(0.1)


def init_robot_pose():
    _ensure_initialized()

    movej(JREADY, vel=VELOCITY, acc=ACC)
    gripper.open_gripper()
    wait_gripper_done()
    mwait()


def go_ready_only():
    _ensure_initialized()
    movej(JREADY, vel=VELOCITY, acc=ACC)
    mwait()


def remove_item_by_pose(target_pose, target_name, drop_pose=None):
    """
    target_pose: [x, y, z, rx, ry, rz]
    drop_pose  : [x, y, z, rx, ry, rz]
    """
    _ensure_initialized()

    if drop_pose is None:
        #drop_pose = CANCEL_DROP_POSE
        drop_pose = JDROP

    try:
        print(f"[robot_cancel_helper] target_pose={target_pose}")
        print(f"[robot_cancel_helper] drop_pose={drop_pose}")

        init_robot_pose()

        ## 구분동작으로 지어줌 ##
        x, y, z, rx, ry, rz = target_pose
        align_pose = [x, y, 300, 90, 180, 90]

        # 1. 물체 위로 먼저 이동
        print(f"{align_pose}로 이동 시작합니다.")
        movel(align_pose, vel=VELOCITY, acc=ACC)
        mwait()

        cur_x, sol = get_current_posx()
        print(f"현재 좌표는 {cur_x}")

        # 2. 물체의 방향만큼 그리퍼 회전
        cur_j = get_current_posj()
        cur_j[5] = cur_j[5] + rz - 90
        movej(cur_j, vel=VELOCITY, acc=ACC)
        mwait()

        # 3. 그리퍼 크기 조정 (물체마다 다르게 적용)
        gripper.move_gripper(GRIPPER_WIDTH[target_name] + 200)
        if not wait_gripper_done():
            print("[robot_cancel_helper] gripper close timeout")
            return False
        mwait()

        # 4. 그리퍼 내려가기 (물체마다 다르게 적용)
        cur_x, sol = get_current_posx()
        cur_x[2] = 220  # z 좌표만 타겟으로 변경
        movel(cur_x, vel=VELOCITY, acc=ACC)
        mwait()

        # 5. 물체 집기 (물체마다 다르게 적용)
        #gripper.close_gripper()
        gripper.move_gripper(GRIPPER_WIDTH[target_name])
        if not wait_gripper_done():
            print("[robot_cancel_helper] gripper close timeout")
            return False
        mwait()

        # 6. 물체 들어올리기 (충돌 방지 위해)
        cur_x, sol = get_current_posx()
        cur_x[2] = align_pose[2]  # z 좌표만 타겟으로 변경
        movel(cur_x, vel=VELOCITY, acc=ACC)
        mwait()

        # 7. drop 위치 이동
        #movel(drop_pose, vel=VELOCITY, acc=ACC)
        movej(drop_pose, vel=VELOCITY, acc=ACC)
        mwait()

        # 8. 그리퍼 열어서 물체 놓기
        gripper.open_gripper()
        if not wait_gripper_done():
            print("[robot_cancel_helper] gripper open timeout")
            return False
        mwait()

        # 9. 준비 자세로 복귀
        go_ready_only()
        return True

    except Exception as e:
        print(f"[robot_cancel_helper] remove_item_by_pose failed: {e}")
        return False