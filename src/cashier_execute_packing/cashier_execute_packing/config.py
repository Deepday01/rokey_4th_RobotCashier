from dataclasses import dataclass


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float


ROTATION_STATION_POSE = Pose(
    x=400.0,
    y=0.0,
    z=120.0,
    rx=180.0,
    ry=0.0,
    rz=180.0,
)

ROTATION_STATION_APPROACH_POSE = Pose(
    x=400.0,
    y=0.0,
    z=200.0,
    rx=180.0,
    ry=0.0,
    rz=180.0,
)

ROTATION_STATION_RETREAT_POSE = Pose(
    x=400.0,
    y=0.0,
    z=250.0,
    rx=180.0,
    ry=0.0,
    rz=180.0,
)