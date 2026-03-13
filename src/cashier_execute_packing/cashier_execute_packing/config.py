from dataclasses import dataclass, field
from typing import List, Tuple            # 타입 힌트를 위한 typing 모듈

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

SAFE_Z = 150.0               # 충돌 방지를 위한 안전 상승 높이

BASE_Z = 222.0 # 베이스 z 높이
PLACE_Z = 210.0 # 베이스 z 높이 # todo 테스트할때 조정하기 # 추후 사용할듯


# READY_J = [0, 0, 90, 0, 90, 0]
READY_J = [0, 0, 90, 0, 90, -90]
# 로봇 시작 대기 자세 (Joint 좌표)


# ================================
# Pick / Place 계산 파라미터
# ================================

APPROACH_OFFSET_Z = 150.0  
# 목표 위치에서 위쪽으로 얼마나 떨어진 위치에서 접근할지
# ex) pick 전에 위에서 접근


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
# 3D Pose 모델
# ================================
# 로봇 TCP, 물체 위치, 박스 위치 등
# 모든 3D 좌표와 자세를 표현하는 기본 모델
@dataclass
class Pose3D:
    x: float      # X 위치
    y: float      # Y 위치
    z: float      # Z 위치
    roll: float   # X축 회전
    pitch: float  # Y축 회전
    yaw: float    # Z축 회전

    # roll, pitch, yaw 값을 튜플 형태로 반환
    def rpy(self) -> Tuple[float, float, float]:
        return (self.roll, self.pitch, self.yaw)


# ================================
# 물체 크기 모델
# ================================
# 물체의 실제 크기를 표현하는 모델
@dataclass
class Size3D:
    width: float   # x 방향 길이
    depth: float   # y 방향 길이
    height: float  # z 방향 길이

    # 크기 정보를 튜플로 반환
    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.width, self.depth, self.height)


# ================================
# 물체 상태 모델
# ================================
# 실제 작업 대상 물체를 표현
@dataclass
class ItemState:
    item_id: str   # 물체 고유 ID
    name: str      # 물체 이름
    pose: Pose3D   # 물체 위치 및 자세
    size: Size3D   # 물체 크기
    durability: int = 0  # 내구도 (fragile 판단 등에 활용 가능)

    # ItemState 객체를 깊은 복사
    # 회전 계산이나 상태 변경 시 원본 데이터를 보존하기 위해 사용
    def copy(self) -> "ItemState":
        return ItemState(
            item_id=self.item_id,
            name=self.name,
            pose=Pose3D(
                x=self.pose.x,
                y=self.pose.y,
                z=self.pose.z,
                roll=self.pose.roll,
                pitch=self.pose.pitch,
                yaw=self.pose.yaw,
            ),
            size=Size3D(
                width=self.size.width,
                depth=self.size.depth,
                height=self.size.height,
            ),
            durability=self.durability,
        )


# ================================
# 물체 배치 위치 모델
# ================================
# 특정 물체를 어디에 놓을지 표현
@dataclass
class PlacementState:
    object_index: int  # pick_items 리스트에서의 인덱스
    pose: Pose3D       # 목표 배치 위치

    # PlacementState 객체 복사
    def copy(self) -> "PlacementState":
        return PlacementState(
            object_index=self.object_index,
            pose=Pose3D(
                x=self.pose.x,
                y=self.pose.y,
                z=self.pose.z,
                roll=self.pose.roll,
                pitch=self.pose.pitch,
                yaw=self.pose.yaw,
            ),
        )


# ================================
# 회전 스텝 모델
# ================================
# 회전 스테이션에서 물체를 얼마나 회전할지 정의
@dataclass
class AlignStep:
    rx_deg: float = 0.0  # x축 회전 각도
    ry_deg: float = 0.0  # y축 회전 각도
    rz_deg: float = 0.0  # z축 회전 각도


# ================================
# StagePlan
# ================================
# 물체를 집어서 회전 스테이션에 올리는 단계
@dataclass
class InitObjectPickPlan:
    pick_approach_pose: Pose3D      # pick 접근 위치
    pick_pose: Pose3D               # 실제 pick 위치
    pick_retreat_pose: Pose3D       # pick 후 상승 위치 # todo 안쓸듯 아마도
  

# ================================
# AlignPlan
# ================================
# 회전 스테이션 위에서 물체를 정렬하는 계획
@dataclass
class AlignPlan:
    required: bool                          # 회전 정렬이 필요한지 여부
    steps: List[AlignStep] = field(default_factory=list)  # 회전 스텝 리스트


# ================================
# BoxPlan
# ================================
# 회전 스테이션에서 물체를 다시 집어 박스에 넣는 단계
@dataclass
class BoxPlan:
    station_pick_approach_pose: Pose3D   # 스테이션 pick 접근 위치
    station_pick_pose: Pose3D            # 스테이션 pick 위치
    station_pick_retreat_pose: Pose3D    # pick 후 상승 위치
    box_approach_pose: Pose3D            # 박스 접근 위치
    box_place_pose: Pose3D               # 박스 내부 최종 배치 위치
    box_retreat_pose: Pose3D             # 박스에서 이탈 위치


# ================================
# PackingPlan
# ================================
# 물체 하나에 대한 전체 작업 계획
@dataclass
class PackingPlan:
    task_index: int          # 작업 번호
    item: ItemState          # 작업 대상 물체
    placement: PlacementState  # 최종 목표 배치 정보
    init_object_pick_plan: InitObjectPickPlan    
    align_plan: AlignPlan    # Align 단계 계획
    box_plan: BoxPlan        # Box 단계 계획


# ================================
# 전체 실행 계획
# ================================
# 여러 물체 작업을 하나로 묶은 실행 계획
@dataclass
class PackingPlanList:
    planList: List[PackingPlan] = field(default_factory=list)


# ================================
# ROS 메시지 → Pose3D 변환
# ================================
def pick_pose_from_ros_fields(msg) -> Pose3D:
    return Pose3D(
        x=msg.x,
        y=msg.y,
        z=BASE_Z + msg.depth/2, # 베이스 높이 + depth/2 = 물체 중심 높이
        roll=msg.roll,
        pitch=msg.pitch,
        yaw=msg.yaw,
    )

def place_pose_from_ros_fields(msg) -> Pose3D:
    return Pose3D(
        x=msg.x,
        y=msg.y,
        z=msg.z,
        roll=msg.roll,
        pitch=msg.pitch,
        yaw=msg.yaw,
    )

# ================================
# ROS Item 메시지 → ItemState 변환
# ================================
def item_from_ros_msg(item_msg) -> ItemState:
    return ItemState(
        item_id=item_msg.item_id,
        name=item_msg.name,
        pose=pick_pose_from_ros_fields(item_msg),
        size=Size3D(
            width=item_msg.width,
            height=item_msg.height,
            depth=item_msg.depth,
        ),
        durability=item_msg.durability,
    )


# ================================
# ROS Placement 메시지 → PlacementState 변환
# ================================
def placement_from_ros_msg(placement_msg) -> PlacementState:
    return PlacementState(
        object_index=placement_msg.object_index,
        pose=place_pose_from_ros_fields(placement_msg),
    )


# ================================
# Pose Z 오프셋 함수
# ================================
# 특정 pose의 z 값을 dz 만큼 올리거나 내리는 함수
# 주로 approach pose 생성 시 사용
def offset_pose_z(pose: Pose3D, dz: float) -> Pose3D:
    return Pose3D(
        x=pose.x,
        y=pose.y,
        z=pose.z + dz,
        roll=pose.roll,
        pitch=pose.pitch,
        yaw=pose.yaw,
    )












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

