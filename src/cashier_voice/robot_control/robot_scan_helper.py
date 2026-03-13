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
VELOCITY = 100
ACC = 100

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = "502"

# virtual launch일 때 True
SIMULATION_MODE = True

# robot_control.py의 init_robot()와 같은 준비 자세
JREADY = [0, 0, 90, 0, 90, 0]

# 스캔 자세
SCAN_POSE = [260, 50, 545, 90, 180, 90]

_initialized = False
gripper = None
movej = None
movel = None
mwait = None


class DummyGripper:
    def open_gripper(self):
        print("[SIM] open gripper")

    def close_gripper(self):
        print("[SIM] close gripper")

    def get_status(self):
        # [busy, grip_detected] 형식처럼 사용
        return [0, 1]


def _ensure_initialized():
    global _initialized, gripper, movej, movel, mwait

    if _initialized:
        return

    DR_init.__dsr__id = ROBOT_ID
    DR_init.__dsr__model = ROBOT_MODEL

    if not rclpy.ok():
        rclpy.init(args=None)

    dsr_node = rclpy.create_node("robot_scan_helper_node", namespace=ROBOT_ID)
    DR_init.__dsr__node = dsr_node

    try:
        from DSR_ROBOT2 import movej as _movej, movel as _movel, mwait as _mwait
    except ImportError as e:
        print(f"[robot_scan_helper] import error: {e}")
        sys.exit(1)

    movej = _movej
    movel = _movel
    mwait = _mwait

    if SIMULATION_MODE:
        gripper = DummyGripper()
        print("[robot_scan_helper] simulation mode: using DummyGripper")
    else:
        gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)

    _initialized = True


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

    print(f"[robot_scan_helper] move to ready pose: {JREADY}")
    movej(JREADY, vel=VELOCITY, acc=ACC)
    mwait()


def move_to_scan_pose():
    _ensure_initialized()

    try:
        # 1. 먼저 안전한 준비 자세로 이동
        init_robot_pose()

        # 2. 스캔 전 그리퍼 닫기
        gripper.close_gripper()
        if not wait_gripper_done():
            print("[robot_scan_helper] gripper close timeout")
            return False

        # 3. 스캔 자세로 이동
        print(f"[robot_scan_helper] move to scan pose: {SCAN_POSE}")
        movel(SCAN_POSE, vel=VELOCITY, acc=ACC)
        mwait()

        return True

    except Exception as e:
        print(f"[robot_scan_helper] failed: {e}")
        return False