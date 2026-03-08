from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TOOL = "Tool Weight"
ROBOT_TCP = "GripperDA_v1"

VELOCITY = 100
ACC = 100
SAFE_Z = 300.0
READY_J = [0, 0, 90, 0, 90, 0]

APPROACH_OFFSET_Z = 100.0
GRIP_MARGIN = 2.0
MIN_GRIP_WIDTH = 5.0
GRIPPER_TIMEOUT_SEC = 5.0
POLL_INTERVAL_SEC = 0.1

GRIPPER_NAME = "rg2"
TOOLCHARGER_IP = "192.168.1.1"
TOOLCHARGER_PORT = 502

ROTATION_STATION_PLACE_POSE = Pose(
    x=400.0,
    y=0.0,
    z=120.0,
    roll=180.0,
    pitch=0.0,
    yaw=180.0,
)

ROTATION_STATION_APPROACH_POSE = Pose(
    x=400.0,
    y=0.0,
    z=200.0,
    roll=180.0,
    pitch=0.0,
    yaw=180.0,
)

ROTATION_STATION_RETREAT_POSE = Pose(
    x=400.0,
    y=0.0,
    z=250.0,
    roll=180.0,
    pitch=0.0,
    yaw=180.0,
)
