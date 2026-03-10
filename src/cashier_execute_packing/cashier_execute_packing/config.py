from dataclasses import dataclass
from cashier_execute_packing.packing_models import Pose3D

# ================================
# 로봇 기본 설정
# ================================

ROBOT_ID = "dsr01"           # ROS에서 사용하는 로봇 네임스페이스
ROBOT_MODEL = "m0609"        # 두산 로봇 모델명
ROBOT_TOOL = "Tool Weight"   # 로봇에 장착된 Tool 설정 이름
ROBOT_TCP = "GripperDA_v1"   # 로봇 TCP (Tool Center Point) 이름


# ================================
# 로봇 모션 설정
# ================================

VELOCITY = 100               # 로봇 이동 속도
ACC = 100                    # 로봇 이동 가속도

SAFE_Z = 300.0               # 충돌 방지를 위한 안전 상승 높이

READY_J = [0, 0, 90, 0, 90, 0]  
# 로봇 시작 대기 자세 (Joint 좌표)


# ================================
# Pick / Place 계산 파라미터
# ================================

APPROACH_OFFSET_Z = 100.0  
# 목표 위치에서 위쪽으로 얼마나 떨어진 위치에서 접근할지
# ex) pick 전에 위에서 접근

GRIP_MARGIN = 2.0            
# 그리퍼 폭 계산 시 여유 공간

MIN_GRIP_WIDTH = 5.0         
# 그리퍼 최소 닫힘 폭

GRIPPER_TIMEOUT_SEC = 5.0    
# 그리퍼 동작 완료 대기 최대 시간

POLL_INTERVAL_SEC = 0.1      
# 그리퍼 상태 확인 주기


# ================================
# 그리퍼 장비 설정
# ================================

GRIPPER_NAME = "rg2"         
# 사용 중인 OnRobot 그리퍼 모델

TOOLCHARGER_IP = "192.168.1.1"  
# Tool Charger 장비 IP 주소

TOOLCHARGER_PORT = 502        
# Tool Charger 통신 포트



# ================================
# 3D Pose 모델
# ================================
# 로봇 TCP 위치, 스테이션 위치, 접근 위치 등을 표현하는
# 3차원 위치 + 자세 데이터 구조
@dataclass(frozen=True)
class Pose:
    x: float      # X 좌표 (로봇 기준 또는 월드 좌표)
    y: float      # Y 좌표
    z: float      # Z 좌표 (높이)
    roll: float   # X축 회전
    pitch: float  # Y축 회전
    yaw: float    # Z축 회전


# ================================
# 회전 스테이션 위치
# ================================



# 회전 스테이션 접근 위치
# 실제 place 위치보다 위쪽에 위치
ROTATION_STATION_APPROACH_POSE = Pose3D(
    x=91.712,
    y=-582.737,
    z=384.288,
    roll=177.012,
    pitch=135.921,
    yaw=88.244,
)
ROTATION_STATION_APPROACH_POSE = Pose3D(
    x=91.712,
    y=-582.737,
    z=384.288,
    roll=177.012,
    pitch=135.921,
    yaw=88.244,
)



# 회전 스테이션 위에 물체를 내려놓는 실제 위치
ROTATION_STATION_PLACE_BASE_POSE = Pose3D(
    x=91.723,
    y=-582.756,
    z=182.977, # 런타임에서 pick z값이 재할당됨
    roll=176.985,
    pitch=135.902,
    yaw=88.212,
)

# 회전 스테이션에서 물체를 내려놓은 후
# 로봇이 안전하게 빠져나오는 위치
ROTATION_STATION_RETREAT_POSE = Pose3D(
    x=300.0,
    y=0.0,
    z=200.0 + APPROACH_OFFSET_Z, 
    roll=78.647,
    pitch=179.995,
    yaw=-13.647,
)