#!/usr/bin/env python3

import time

import DR_init
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.node import Node
from cashier_execute_packing.config import *

from cashier_interfaces.action import ExecutePacking

from cashier_execute_packing.packing_executor import (
    make_packing_plan_list,
    execute_plan_with_callbacks,
)
from cashier_execute_packing.util import(
    get_gripper_depth_offset
)


DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

from robot_control.onrobot import RG


gripper = RG(GRIPPER_NAME, TOOLCHARGER_IP, TOOLCHARGER_PORT)


class RobotController:
    def __init__(self, logger=None):
        self.logger = logger
        self.last_grip_width_mm = None
        
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
        from DSR_ROBOT2 import movej, mwait

        movej(READY_J, vel=VELOCITY, acc=ACC)
        mwait()

    # movel 절대좌표 이동
    def move_to_pose(self, pose: Pose3D, ref=None, log: bool = False) -> None:
        from DSR_ROBOT2 import movel, mwait, posx, DR_BASE, DR_TOOL

        if ref is None:
            ref = DR_BASE

        ref_name = "DR_TOOL" if ref == DR_TOOL else "DR_BASE"

        if log and self.logger is not None:
            self.logger.info(
                "[ROBOT_MOVE] movel(abs) -> "
                f"ref={ref_name}, "
                f"x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}, "
                f"roll={pose.roll:.3f}, pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}"
            )

        movel(
            posx([pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]),
            vel=VELOCITY,
            acc=ACC,
            ref=ref,
        )
        mwait()


    # movel 상대좌표 이동
    def move_to_relative_pose(self, pose:Pose3D, ref=None, log: bool = False) -> None:
        from DSR_ROBOT2 import movel, mwait, posx, DR_BASE, DR_TOOL, DR_MV_MOD_REL

        if ref is None:
            ref = DR_BASE

        ref_name = "DR_TOOL" if ref == DR_TOOL else "DR_BASE"

        if log and self.logger is not None:
            self.logger.info(
                "[ROBOT_MOVE] movel(rel) -> "
                f"ref={ref_name}, "
                f"x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}, "
                f"roll={pose.roll:.3f}, pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}"
            )

        movel(
            posx([pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]),
            vel=VELOCITY,
            acc=ACC,
            ref=ref,
            mod=DR_MV_MOD_REL,
        )
        mwait()

    # movej 상대좌표 이동
    def movej_to_relative_pose(self, movej_list) -> None:
        from DSR_ROBOT2 import movej, mwait, DR_MV_MOD_REL

        movej(movej_list, vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
        mwait()


    def open_gripper(self) -> None:
        from DSR_ROBOT2 import mwait

        gripper.open_gripper()
        started = time.time()
        while gripper.get_status()[0]:
            if time.time() - started > GRIPPER_TIMEOUT_SEC:
                raise RuntimeError("gripper open timeout")
            time.sleep(POLL_INTERVAL_SEC)
        mwait()
        
    # def close_gripper(self, width_mm: int | None = None) -> None:
    def close_gripper(self, width_mm: float | None = None) -> None:
        from DSR_ROBOT2 import mwait

        self.logger.info(f'close_gripper 호출 > width_mm={width_mm}')

        if width_mm is None:
            gripper.close_gripper()
        else:
            gripper.move_gripper(width_mm * 10)

        # width 기록
        self.last_grip_width_mm = width_mm

        started = time.time()
        while gripper.get_status()[0]:
            if time.time() - started > GRIPPER_TIMEOUT_SEC:
                raise RuntimeError("gripper open timeout")
            time.sleep(POLL_INTERVAL_SEC)

        mwait()
    def rotate_object(self, step: AlignStep, item: ItemState):
        from DSR_ROBOT2 import get_current_posx

        #====================================테스트 시작====================================

        #====================================테스트 종료====================================



        if step.ry_deg == 90:
            self.logger.info(f"====================ry 90도 회전 수행====================")

            # ====================ry 90도 회전 수행====================
            # 45도 잡기 수행====================
            self.open_gripper()
            self.move_to_pose(Pose3D(91.723, -582.756, 182.977 + 200, 176.985, 135.902, 88.212)) # y축 회전 45도 잡기전 경유 지점 

            # 잡는 위치 1차 이동('바닥 높이 + 물체 뎁스/2)
            object_grap_z = 182.977 + item.size.depth/2
            self.logger.info(f"step.ry_deg > 45도 잡기 >  item.size.depth/2 확인: {item.size.depth/2}")
            
            #======잡는 높이값 보상======
            # 그리퍼 잡는 중심점 더 아래로 보상(단 바닥과 충돌되는 위치보다 위쪽으로 설정) # 15라는 수치는 그리퍼가 45도로 기울어져 바닥에 닿고 있을때 부터 그리퍼 중심점까지 높이 
            if object_grap_z - 15 >= 182.977:
                # 182.977 보다 큰값이면 사용가능
                object_grap_z = object_grap_z - 15
            else:
                # 더 아래로 가면 바닥과충돌하기에 이수치 밑으로는 못감
                object_grap_z = 182.977 


            self.move_to_pose(Pose3D(91.723, -582.756, object_grap_z, 176.985, 135.902, 88.212)) # y축 회전 45도 잡기

            # 잡는 위치 2차 이동(그리퍼 좌표계로 그리퍼 옵셋만큼 이동) # 그리퍼 옵셋: 그리퍼 완전 닫힌 높이 - 그리퍼 물체 잡은 높이
            griper_offset = get_gripper_depth_offset(item.size.height)

            self.move_to_relative_pose(Pose3D(0, 0, griper_offset, 0,0,0), ref=1)

            # 잡기 수행
            width_mm = item.size.height
            self.close_gripper(width_mm = width_mm)
            self.logger.info(f"step.ry_deg > 45도 잡기 > width_mm = item.size.height 확인: {width_mm}")


            # 올라오기
            self.move_to_relative_pose(Pose3D(0, 0, 200, 0,0,0))


            # 135도 놓기 시작====================
            self.move_to_pose(Pose3D(-148.29, -579.322, 8.418 + 200, 178.483, -132.87, 87.051)) # y축 회전 135도 놓기전 경우 지점
            # 아이템 size 스왑=========
            width = item.size.width
            depth = item.size.depth
            item.size.width = depth
            item.size.depth = width
            self.logger.info(f"<길이 스왑> \n기존 width: {width}, depth: {depth}\n 변경 width: {item.size.width}, depth: {item.size.depth}")
            # 아이템 ry 값 싱크 맞추기
            item.pose.pitch = abs(item.pose.pitch - 90) 

            # 1차 놓는 위치 이동('바닥 높이 + 물체 뎁스/2)
            object_lay_approch_z = 8.418 + item.size.depth/2
            self.logger.info(f"step.ry_deg  > 135도 놓기  > item.size.depth/2 확인: {item.size.depth/2}")
            self.move_to_pose(Pose3D(-148.29, -579.322, object_lay_approch_z, 178.483, -132.87, 87.051)) # y축 회전 135도 접근

            # 그리퍼 옵셋 이동
            griper_offset = get_gripper_depth_offset(item.size.height)
            self.move_to_relative_pose(Pose3D(0, 0, griper_offset, 0,0,0), ref=1)
            self.logger.info(f"step.ry_deg  > 135도 놓기  > griper_offset 확인: {griper_offset}")


            # 그리퍼 오픈
            self.open_gripper()

            # 올라오기
            self.move_to_relative_pose(Pose3D(0, 0, 200, 0,0,0)) 
            # self.move_to_relative_pose(Pose3D(0, 0, 150, 0,0,0)) # 물체 간섭 방지 목적(경유 지점)
            # self.move_to_pose(Pose3D(-62.725, -581.752, 143.232 - 5 + 200, 101.222, 178.691, 11.537)) # 경유 지점


        elif step.rz_deg == 90:
            self.logger.info(f" ====================rz 90도 회전 수행====================")

            # ====================rz 90도 회전 수행====================
            self.open_gripper()
            self.move_to_pose(Pose3D(-62.725, -581.752, 143.232 - 5 + 200, 101.222, 178.691, 11.537)) # 경유 지점

            # 물체 잡기('바닥 높이 + 물체 뎁스/2 - 그리퍼 옵셋)
            griper_offset = get_gripper_depth_offset(item.size.height)
            object_grap_z = 143.232 - 5 + item.size.depth/2 - griper_offset
            self.logger.info(f"step.rz_deg  > 잡기 > item.size.depth/2 확인: {item.size.depth/2}")
            self.logger.info(f"step.rz_deg  > 잡기 > griper_offset 확인: {griper_offset}")

            self.move_to_pose(Pose3D(-62.725, -581.752, object_grap_z, 101.222, 178.691, 11.537)) # 물체 중심 높이까지 이동

            # 물체 잡기
            self.close_gripper(width_mm = item.size.height)

            # 올라오기 > rz 90도 회전 
            # object_grap_z = 145.745 + item.size.depth/2 - griper_offset
            # self.logger.info(f"step.rz_deg  > 올라오기 > rz 90도 회전  > item.size.depth/2 확인: {item.size.depth/2}")
            # self.logger.info(f"step.rz_deg  > 올라오기 > rz 90도 회전 > griper_offset 확인: {griper_offset}")
            # self.move_to_pose(Pose3D(-45.907, -345.922, object_grap_z +10, 125.195, 179.579, 125.739)) # 10은 물체를 살짝들어 회전시키기 위함
            
            # 올라오기
            self.move_to_relative_pose(Pose3D(0,0,10,0,0,0))
            # rz 90도 회전 
            self.movej_to_relative_pose([0,0,0,0,0,90])




            # 아이템 size 스왑=========
            width = item.size.width
            height = item.size.height
            item.size.width = height
            item.size.height = width
            self.logger.info(f"<길이 스왑> \n기존 width: {width}, height: {height}\n 변경 width: {item.size.width}, height: {item.size.height}")
            # 아이템 rz 값 싱크 맞추기
            item.pose.yaw = abs(item.pose.yaw - 90) 
            

            # 물체 내려 놓기
            # self.move_to_pose(Pose3D(-45.907, -345.922, object_grap_z, 125.195, 179.579, 125.739)) # 물체 중심 높이까지
            self.move_to_relative_pose(Pose3D(0,0,-10,0,0,0))

            # 그리퍼 오픈
            self.open_gripper()

            # 올라오기
            self.move_to_relative_pose(Pose3D(0, 0, 200, 0,0,0)) # 물체 간섭 방지 목적(경유 지점)
            # self.move_to_relative_pose(Pose3D(0, 0, 150, 0,0,0)) # 물체 간섭 방지 목적(경유 지점)
            # self.move_to_pose(Pose3D(-62.725, -581.752, 143.232 - 5 + 200, 101.222, 178.691, 11.537)) # 경유 지점


class ExecutePackingServer(Node):
    def __init__(self):
        super().__init__("execute_packing_server", namespace=ROBOT_ID)
        self._busy = False
        self.robot = RobotController(logger=self.get_logger())
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
    

    # 초기 오브젝트 픽
    def excute_init_object_pick(self, packingPlan: PackingPlan) -> None:
        from DSR_ROBOT2 import get_current_posx

        item = packingPlan.item

        self.get_logger().info(f'물체 최초 pick 시작 > 물체: {item.name}')

        # 잡는 높이 조정(물체 중심 z - 그리퍼 너비에 따른 뎁스 오프셋값)
        griper_depth_offset = get_gripper_depth_offset(packingPlan.item.size.height)
        self.get_logger().info(f'물체 최초 pick 단계 > griper_depth_offset: {griper_depth_offset}')

        pick_z = item.pose.z - griper_depth_offset

        
        # 로봇 물체 접근
        self.robot.move_to_pose(
            Pose3D(
                x=item.pose.x,
                y=item.pose.y,
                z=pick_z + 200,
                roll = 90,
                pitch = 180,
                yaw = 90,
            )
        )

        # yaw 만큼 회전
        self.robot.movej_to_relative_pose([0, 0, 0, 0, 0, item.pose.yaw -90])
        self.get_logger().info(f'물체 회전: {item.pose.yaw}')


        self.robot.open_gripper()
   
        # 그대로 아래로 내려가기
        self.robot.move_to_relative_pose(Pose3D(0,0,-200,0,0,0))


        self.get_logger().info(f'물체 최초 pick 단계 > item.pose.yaw: {item.pose.yaw}')

        # # 잡기
        self.robot.close_gripper(width_mm=item.size.height)

        # 그대로 올라오기
        self.robot.move_to_relative_pose(Pose3D(0,0,200,0,0,0))

        # 물체 정렬 
        self.robot.movej_to_relative_pose([0, 0, 0, 0, 0, -item.pose.yaw])
        self.get_logger().info(f'물체 정렬: {-item.pose.yaw}')

        # 물체 상태 변경
        item.pose.yaw -= item.pose.yaw
        self.get_logger().info(f'물체 최초 pick 단계 > 정렬된 물체 yaw 확인 : {item.pose.yaw}')


    # 회전 스테이지
    def align_object(self, packingPlan: PackingPlan) -> None:
        from DSR_ROBOT2 import get_current_posx 

        align_plan = packingPlan.align_plan
        item = packingPlan.item

        if not align_plan.required:
            self.get_logger().info(f'회전 스테이지 스킵')
            return
        else:
            # ============회전 스테이지에 내려놓기============
            self.get_logger().info(f'회전 스테이지에 내려놓기 시작 ')
            # 경유 지점 이동
            self.robot.move_to_pose(Pose3D(-62.725, -581.752, 143.232 - 5 + 200, 101.222, 178.691, 11.537))
            # 물체 좌표에 내려 놓기 
            # - 내려 놓는 높이 계산 = 바닥 높이 + 물체 뎁스/2 - 그리퍼 옵셋(그리퍼 완전 닫힌 높이 - 그리퍼 물체 잡은 높이)
            griper_offset = get_gripper_depth_offset(item.size.height)
            self.get_logger().info(f'회전 스테이지에 내려놓기 시작 > griper_offset: {griper_offset}')

            pick_z = 143.232 - 5 + item.size.depth/2 - griper_offset
            self.get_logger().info(f'회전 스테이지에 내려놓기 시작 > item_size.depth/2: {item.size.depth/2}')
            # 물체 내려놓기
            self.robot.move_to_pose(Pose3D(-62.725, -581.752, pick_z, 101.222, 178.691, 11.537))

            # 그리퍼 열기
            self.robot.open_gripper()

            # 올라오기
            self.robot.move_to_relative_pose(Pose3D(0,0,200,0,0,0))

            # ============회전 수행============
            for step in align_plan.steps:
                self.get_logger().info(f'회전 수행 > 현재 회전 : {step.rx_deg}, {step.ry_deg}, {step.rz_deg}')
                self.robot.rotate_object(step , packingPlan.item)

            # # ============회전 종료 > 잡기 및 들어올리기 ============
            self.get_logger().info(f'회전 종료 > 잡기 및 들어올리기')
            # 그리퍼 물체 잡는 방향으로 회전
            self.robot.move_to_pose(Pose3D(-62.725, -581.752, 143.232 - 5 + 200, 101.222, 178.691, 11.537)) 
            # 그리퍼 오픈
            self.robot.open_gripper()
            # 하강 및 물체 접근
            # - 잡는 높이 계산 = 바닥 높이 + 물체 뎁스/2 - 그리퍼 옵셋(그리퍼 완전 닫힌 높이 - 그리퍼 물체 잡은 높이)
            griper_offset = get_gripper_depth_offset(item.size.height)
            self.get_logger().info(f'회전 종료 > 잡기 및 들어올리기 > griper_offset: {griper_offset}')
            pick_z = 143.232 - 5 + item.size.depth/2 - griper_offset
            self.get_logger().info(f'회전 종료 > 잡기 및 들어올리기 > item_size.depth/2: {item.size.depth/2}')
            self.robot.move_to_pose(Pose3D(-62.725, -581.752, pick_z, 101.222, 178.691, 11.537))
            # 잡기
            self.get_logger().info(f'회전 종료 > 잡기 및 들어올리기 > item.size.height: {item.size.height}')
            self.robot.close_gripper(width_mm=item.size.height)
            # 다음 이동을 위한 정위치로 올라오기(다음 작업부터 상대좌표로 이동함)
            self.robot.move_to_pose(Pose3D(-62.725, -581.752, 143.232 - 5 + 200, 101.222, 178.691, 11.537)) 


    # 박스에 놓기
    def pick_and_place_to_box(self, packingPlan: PackingPlan) -> None:
        from DSR_ROBOT2 import get_current_posx 

        item = packingPlan.item
        placement = packingPlan.placement


        # 회전스테이션을 거쳤다면
        if packingPlan.align_plan.required:
            # 90도 회전이 필요하다면 회전
            if packingPlan.placement.pose.yaw != item.pose.yaw:
                self.robot.movej_to_relative_pose([0,0,0,0,0,90])
                self.get_logger().info(f'플레이싱시 필요시 90도 회전')

                # 스왑
                width = item.size.width
                height = item.size.height
                item.size.width = height
                item.size.height = width

            # 회전스테이션을 거쳤다면
            # self.robot.move_to_relative_pose(Pose3D(placement.pose.x, placement.pose.y, 0,0,0,0)) # origin 
            self.robot.move_to_relative_pose(Pose3D(placement.pose.x, placement.pose.y-30, 0,0,0,0)) # y값 -30 보정됨
            self.robot.move_to_relative_pose(Pose3D(0,0,placement.pose.z -200,0,0,0)) # 바닥에서 올라온 높이가 200임
            
            
            # 테스트 로그
            self.get_logger().info(
                f"[ITEM] id={item.item_id}, name={item.name}, "
                f"pose=({item.pose.x:.3f}, {item.pose.y:.3f}, {item.pose.z:.3f}, "
                f"{item.pose.roll:.3f}, {item.pose.pitch:.3f}, {item.pose.yaw:.3f}), "
                f"size=({item.size.width}, {item.size.depth}, {item.size.height}), "
                f"durability={item.durability}"
            )




            self.robot.close_gripper(width_mm = self.robot.last_grip_width_mm +4) # 잡고 있는 상태에서 2mm열림

            
            self.robot.move_to_relative_pose(Pose3D(0,0,200,0,0,0)) 
            self.robot.move_to_relative_pose(Pose3D(0,0,0,0,0,0)) # 마지막 작업 스킵되는 문제로 빈 move 작성
        else:
            # 회전스테이션을 거쳤치지 않았다면
            # self.robot.move_to_pose(Pose3D(-62.725 + placement.pose.x, -581.752 + placement.pose.y, 143.232 - 5 + 200, 101.222, 178.691, 11.537)) # 경유 지점 # origin
            
            cur_pose, _ = get_current_posx()
            self.robot.move_to_pose(Pose3D(-62.725 + placement.pose.x, -581.752 + placement.pose.y -30, 143.232 - 5 + 200, 101.222, 178.691, 11.537)) # 경유 지점 -30 y값 보정함 
            # self.robot.move_to_pose(Pose3D(-62.725 + placement.pose.x, -581.752 + placement.pose.y, 143.232 - 5 + 200, 101.222, 178.691, 11.537)) # 경유 지점 origin
            

            if packingPlan.placement.pose.yaw != item.pose.yaw:
                self.robot.movej_to_relative_pose([0,0,0,0,0,90])
                self.get_logger().info(f'플레이싱시 필요시 90도 회전')

                # 스왑
                width = item.size.width
                height = item.size.height
                item.size.width = height
                item.size.height = width


            # self.robot.move_to_relative_pose(Pose3D(placement.pose.x -62.725, placement.pose.y -581.752, 143.232 - 5+200, 101.222, 178.691, 11.537)) 
            # self.move_to_pose(Pose3D(-62.725, -581.752, 143.232 - 5 + 200, 101.222, 178.691, 11.537)) # 경유 지점
            self.robot.move_to_relative_pose(Pose3D(0,0,placement.pose.z -200,0,0,0)) # 바닥에서 올라온 높이가 200임



            # 테스트 로그
            self.get_logger().info(
                f"[ITEM] id={item.item_id}, name={item.name}, "
                f"pose=({item.pose.x:.3f}, {item.pose.y:.3f}, {item.pose.z:.3f}, "
                f"{item.pose.roll:.3f}, {item.pose.pitch:.3f}, {item.pose.yaw:.3f}), "
                f"size=({item.size.width}, {item.size.depth}, {item.size.height}), "
                f"durability={item.durability}"
            )


            self.robot.close_gripper(width_mm = self.robot.last_grip_width_mm +4) # 잡고 있는 상태에서 1mm열림
            self.robot.move_to_relative_pose(Pose3D(0,0,200,0,0,0)) 
            self.robot.move_to_relative_pose(Pose3D(0,0,0,0,0,0)) # 마지막 작업 스킵되는 문제로 빈 move 작성


    # 피드백
    def _publish_item_completed_feedback(
        self,
        goal_handle,
        plan: PackingPlan,
        completed_count: int,
        total_count: int,
    ) -> None:
        feedback = ExecutePacking.Feedback()

        message = (
            f"[{completed_count}/{total_count}] "
            f"박스 적재 완료 | item_name={plan.item.name}, "
            f"item_id={plan.item.item_id}, "
            f"task_index={plan.task_index}"
        )

        # action feedback 정의를 아직 못 본 상태라
        # 존재하는 필드만 안전하게 채운다.
        if hasattr(feedback, "message"):
            feedback.message = message
        if hasattr(feedback, "item_name"):
            feedback.item_name = plan.item.name
        if hasattr(feedback, "item_id"):
            feedback.item_id = plan.item.item_id
        if hasattr(feedback, "task_index"):
            feedback.task_index = plan.task_index
        if hasattr(feedback, "completed_count"):
            feedback.completed_count = completed_count
        if hasattr(feedback, "total_count"):
            feedback.total_count = total_count
        if hasattr(feedback, "success"):
            feedback.success = True

        goal_handle.publish_feedback(feedback)

        self.get_logger().info(
            f"feedback 전송 완료 | "
            f"{completed_count}/{total_count} | {plan.item.name}"
        )



    def publish_progress_feedback(
        self,
        goal_handle,
        completed_count: int,
        total_count: int,
        start_time: float,
    ) -> None:
        feedback = ExecutePacking.Feedback()
        feedback.progress = f"{completed_count}/{total_count}"
        feedback.runtime = int(time.time() - start_time)
        goal_handle.publish_feedback(feedback)
        








    # 실행 콜백 
    def execute_callback(self, goal_handle):
        self.get_logger().info("execute_callback()!!!")

        self._busy = True
        result = ExecutePacking.Result()

        try:
            start_time = time.time()

            request = goal_handle.request
            self.get_logger().info(f"클라이언트 데이터 수신 완료")
            self.get_logger().info(f"계획 작성 시작")

            packingPlanList: PackingPlanList = make_packing_plan_list(
                pick_items=request.pick_items,
                place_items=request.place_items,
                logger=self.get_logger()
            )
            self.get_logger().info(f"계획 작성 완료")

            self.get_logger().info(f"초기 위치로 이동")
            self.robot.move_to_relative_pose(Pose3D(0,0,150,0,0,0))
            self.robot.move_ready()




            # #=================== 테스트 진행 ===============
            # 스캔위치
            # self.robot.move_to_relative_pose(Pose3D(0,0,150,0,0,0))
            # self.robot.move_to_pose(Pose3D(260 ,50 , 535 + 10, 90, 180, 90)) 
            # self.robot.close_gripper(width_mm=50)
            # self.robot.close_gripper(width_mm=50 +1)
       
            # return

      

            # item = packingPlanList.planList[0].item

            # self.get_logger().info(f'물체 최초 pick 시작 > 물체: {item.name}')

            # # 잡는 높이 조정(물체 중심 z - 그리퍼 너비에 따른 뎁스 오프셋값)
            # griper_depth_offset = get_gripper_depth_offset(item.size.height)
            # self.get_logger().info(f'물체 최초 pick 단계 > griper_depth_offset: {griper_depth_offset}')

            # pick_z = item.pose.z - griper_depth_offset

            
            # # 로봇 물체 접근
            # self.robot.move_to_pose(
            #     Pose3D(
            #         x=item.pose.x,
            #         y=item.pose.y,
            #         z=pick_z + 200,
            #         roll = 90,
            #         pitch = 180,
            #         yaw = 90,
            #     )
            # )

            # self.robot.open_gripper()
            
            # # 그대로 아래로 내려가기
            # self.robot.move_to_relative_pose(Pose3D(0,0,-200,0,0,0))

            # # yaw 만큼 회전
            # self.robot.movej_to_relative_pose([0, 0, 0, 0, 0, item.pose.yaw -90])
            # self.get_logger().info(f'물체 최초 pick 단계 > item.pose.yaw: {item.pose.yaw}')

            # # # 잡기
            # self.robot.close_gripper(width_mm=item.size.height)

            # # 그대로 올라오기
            # self.robot.move_to_relative_pose(Pose3D(0,0,200,0,0,0))

            # # 물체 정렬 
            # self.robot.movej_to_relative_pose([0, 0, 0, 0, 0, -item.pose.yaw])

            # # 물체 상태 변경
            # item.pose.yaw -= item.pose.yaw
            # self.get_logger().info(f'물체 최초 pick 단계 > 정렬된 물체 yaw 확인 : {item.pose.yaw}')

            # return

        
            # #=================== 테스트 완료 ===============


            isComplet = execute_plan_with_callbacks(
                planList=packingPlanList,
                excute_init_object_pick=self.excute_init_object_pick,
                excute_align_object=self.align_object,
                execute_pick_and_place_to_box=self.pick_and_place_to_box,
                on_item_completed=lambda completed_count, total_count, plan: (
                    self.publish_progress_feedback(
                        goal_handle=goal_handle,
                        completed_count=completed_count,
                        total_count=total_count,
                        start_time=start_time,
                    )
                ),
            )

            if isComplet:
                self.get_logger().info(f"패킹 작업 완료")
                self.robot.move_ready()

            goal_handle.succeed()
            result.success = True
            return result

        except Exception as e:
            self.get_logger().error(f"[익셉션 발생] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

            self.get_logger().info(f"패킹 작업 실패")
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
