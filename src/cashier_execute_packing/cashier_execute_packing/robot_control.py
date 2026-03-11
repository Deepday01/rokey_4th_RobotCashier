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
    def move_to_pose(self, pose) -> None:
        from DSR_ROBOT2 import movel, mwait, posx

        movel(
            posx([pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]),
            vel=VELOCITY,
            acc=ACC,
        )
        mwait()

    # movel 상대좌표 이동
    def move_to_relative_pose(self, pose) -> None:
        from DSR_ROBOT2 import movel, mwait, posx,  DR_MV_MOD_REL

        movel(
            posx([pose.x, pose.y, pose.z, pose.roll, pose.pitch, pose.yaw]),
            vel=VELOCITY,
            acc=ACC,
            mod=DR_MV_MOD_REL
        )
        mwait()

    # movej 상대좌표 이동
    def movej_to_relative_pose(self, movej_list) -> None:
        from DSR_ROBOT2 import movej, mwait, DR_MV_MOD_REL

        movej(movej_list, vel=VELOCITY, acc=ACC, mod=DR_MV_MOD_REL)
        mwait()

    
    def safe_rise(self) -> None:
        from DSR_ROBOT2 import get_current_posx, mwait

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
        

    def close_gripper(self, width_mm: int | None = None) -> None:
        from DSR_ROBOT2 import mwait

        if width_mm is None:
            gripper.close_gripper()
        else:
            gripper.move_gripper(width_mm *10) # 기본 단위가 1/10 mm 이기에 10을 곱해줌
        started = time.time()
        while gripper.get_status()[0]:
            if time.time() - started > GRIPPER_TIMEOUT_SEC:
                raise RuntimeError("gripper open timeout")
            time.sleep(POLL_INTERVAL_SEC)
        mwait()

    # def rotate_object(self, object_pose, rotate_direction):
    def rotate_object(self):
        from DSR_ROBOT2 import get_current_posx
        # # # TODO : 여기서부터
        # # if rotate_direction == 'x_axis':
        # #     self.move_to_pose()
        # # if rotate_direction == 'y_axis':
        # #     self.move_to_pose()
        # # if rotate_direction == 'z_axis':
        # #     pass

        # # 45도 잡기 시작
        # current, _ = get_current_posx()
        # self.open_gripper()
        # self.move_to_pose(Pose3D(91.712, -582.737, 384.288, 177.012, 135.921, 88.244)) # y축 회전 45도 잡기전 경유 지점 
        # self.move_to_pose(Pose3D(91.723, -582.756, 182.977 + 10, 176.985, 135.902, 88.212)) # y축 회전 45도 잡기
        # # 10
        # self.close_gripper(width_mm = 40+15) # 기존 + 부착된 그리퍼 너비길이가 합쳐서 15mm 이라 보정
        # # self.close_gripper() 
        # self.safe_rise() # 안전을 위해 추가


        # # # 135도 놓기 시작
        # self.move_to_pose(Pose3D(-152.673, -579.061, 207.891, 179.251, -132.47, 87.477)) # y축 회전 135도 놓기전 경우 지점
        # self.move_to_pose(Pose3D(-152.678, -579.069, 7.86 + 37.5, 179.252, -132.471, 87.476)) # y축 회전 135도 놓기
        # # 37.5

        # self.open_gripper()
        # self.safe_rise()

        # current, _ = get_current_posx()
        # self.move_to_pose(Pose3D(286.7, 117, 546-60 +100,current[3],current[4], 41)) 
        # # tcp 끝단에서 추가 그리퍼 길이 60 // 지금 테스트에서는 z값만 추가하면됨


        # current, _ = get_current_posx()
        # self.move_to_pose(Pose3D(430.0, 78.0, current[2],current[3],current[4], 83.0 +90)) 
        



        ########################################### 테스트 시작 ###########################################
        # self.move_ready()


        
        # 스캔위치
        self.move_to_pose(Pose3D(260 ,50 , 535 + 10, 90, 180, 90)) 


        # 우측 상단
        # self.move_to_pose(Pose3D(493.0, -75, 240, 90, 180, 90)) 
        # 좌측 상단
        # self.move_to_pose(Pose3D(493.0, -75 + 400, 240, 90, 180, 90)) 
        # 좌측 하단
        # self.move_to_pose(Pose3D(493.0 -400, -75 + 400, 240, 90, 180, 90)) 
        # 우측 하단
        # self.move_to_pose(Pose3D(493.0 -400, -75, 240, 90, 180, 90)) 


        # 에크론
        # self.move_to_pose(Pose3D(287.4, 53.2, 280, 90, 180, 90)) 
        # 카라멜
        # self.move_to_pose(Pose3D(431.6, 140.7, 280, 90, 180, 90)) 
        # 인섹트
        # self.move_to_pose(Pose3D(158.9, 152.5, 280, 90, 180, 90)) 
        # 이클립스
        # self.move_to_pose(Pose3D(250.6, 277.1, 280, 90, 180, 90)) 





        # self.open_gripper()
        # self.close_gripper(width_mm = 32+10+2)
        # self.safe_rise()


        

        # 베이스 z 높이
        # self.move_to_pose(Pose3D(301.6, 140.7, 220, 90, 180, 90)) 


        ########################################### 테스트 종료 ###########################################
        ########################################### 그리퍼 너비에 따른 depth값 추출 테스트 종료 ###########################################
        
        # self.move_to_pose(Pose3D(301.6, 140.7, 222 + 100, 90, 180, 90)) # 베스 좌표
        # self.movej_to_relative_pose([30, 0, 0, 0, 0, 0]  )
        
        # self.open_gripper()

        # self.close_gripper()
        # self.close_gripper(width_mm=10)
        # self.close_gripper(width_mm=20)
        # self.close_gripper(width_mm=30)
        # self.close_gripper(width_mm=40)
        # self.close_gripper(width_mm=50)
        # self.close_gripper(width_mm=60)
        # self.close_gripper(width_mm=70)
        # self.close_gripper(width_mm=80)
        # self.close_gripper(width_mm=90)
        # self.close_gripper(width_mm=100)
        # self.close_gripper(width_mm=110)

        


        # 0~1100mm
        # width(mm) : depth(mm)
        # 0         :   26
        # 100       :   26
        # 200       :   26
        # 300       :   25.9
        # 400       :   25.8
        # 500       :   25.6
        # 600       :   25.3
        # 700       :   24.9
        # 800       :   24.6
        # 900       :   24.1
        # 1000      :   23.4
        # 1100      :   22.3



        ########################################### 테스트 종료 ###########################################

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
    

    # 초기 오브젝트 픽
    def excute_init_object_pick(self, packingPlan: PackingPlan) -> None:
        from DSR_ROBOT2 import get_current_posx

        pick_plan = packingPlan.init_object_pick_plan

        griper_depth_offset = get_gripper_depth_offset(packingPlan.item.size.height)
        # 잡는 높이 조정(물체 중심 z - 그리퍼 너비에 따른 뎁스 오프셋값)
        pick_z = pick_plan.pick_pose.z - griper_depth_offset 
        
        pick_z = pick_plan.pick_pose.z - griper_depth_offset

        self.get_logger().info(
            f"잡는 높이 조정(물체 중심 z - 그리퍼 너비에 따른 뎁스 오프셋값) "
            f"{pick_plan.pick_pose.z:.3f} - {griper_depth_offset:.3f} = {pick_z:.3f}"
        )
            
        
        # 로봇 시작
        self.robot.move_to_pose(pick_plan.pick_approach_pose)
        self.robot.open_gripper()

        # yaw 만큼 회전
        self.robot.movej_to_relative_pose([0, 0, 0, 0, 0, pick_plan.pick_pose.yaw])

        # 그대로 아래로 내려가기
        cur_pos, _ = get_current_posx()
        griper_pick_pose = Pose3D(
            x=pick_plan.pick_pose.x,
            y=pick_plan.pick_pose.y,
            z=pick_z,
            roll = cur_pos[3],
            pitch = cur_pos[4],
            yaw = cur_pos[5],
        )
        self.robot.move_to_pose(griper_pick_pose)

        # 잡기
        self.robot.close_gripper()
        # 그대로 올라오기
        self.robot.move_to_relative_pose(Pose3D(x=0,y=0,z=200,roll=0,pitch=0,yaw=0))


  
    # 회전 스테이션
    def align_object(self, plan, align_plan) -> None:
       
        # 테스트 코드 
        self.robot.rotate_object()
            

        # if not align_plan.required:
        #     return
        # for step in align_plan.steps:
        #     self.robot.rotate_object(
        #         rx_deg=step.rx_deg,
        #         ry_deg=step.ry_deg,
        #         rz_deg=step.rz_deg,
        #     )

    # 박스에 놓기
    def pick_and_place_to_box(self, task, box_plan : BoxPlan) -> None:
        self.robot.move_to_pose(box_plan.station_pick_approach_pose)
        self.robot.move_to_pose(box_plan.station_pick_pose)
        self.robot.close_gripper()
        self.robot.move_to_pose(box_plan.station_pick_retreat_pose)
        self.robot.move_to_pose(box_plan.box_approach_pose)
        self.robot.move_to_pose(box_plan.box_place_pose)
        self.robot.open_gripper()
        self.robot.move_to_pose(box_plan.box_retreat_pose)


    # 실행 콜백 
    def execute_callback(self, goal_handle):
        self.get_logger().info("execute_callback()!!!")

        # 로봇 실행중
        self._busy = True
        result = ExecutePacking.Result()
        try:
            request = goal_handle.request
            PackingPlanList = make_packing_plan_list(
                pick_items=request.pick_items, # 클라이언트에서 정의된 접근 변수값 // pick_items은 인터페이스의 item 이다
                place_items=request.place_items,
            )

            # relase 코드
            # self.robot.move_ready()
            execute_plan_with_callbacks(
                planList=PackingPlanList,
                excute_init_object_pick=self.excute_init_object_pick,
                align_object=self.align_object,
                execute_pick_and_place_to_box=self.pick_and_place_to_box,
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
