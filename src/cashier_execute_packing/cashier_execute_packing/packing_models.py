#!/usr/bin/env python3

from dataclasses import dataclass, field
from typing import Tuple, Any, List


@dataclass
class Pose3D:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def rpy(self) -> Tuple[float, float, float]:
        return (self.roll, self.pitch, self.yaw)


@dataclass
class Size3D:
    width: float
    depth: float
    height: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.width, self.depth, self.height)


@dataclass
class ItemState:
    item_id: str
    name: str
    pose: Pose3D
    size: Size3D
    durability: int = 0

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


@dataclass
class PlacementState:
    object_index: int
    pose: Pose3D

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


@dataclass
class RelativeRotation:
    roll: int
    pitch: int
    yaw: int

    def as_tuple(self) -> Tuple[int, int, int]:
        return (self.roll, self.pitch, self.yaw)

    def is_zero(self) -> bool:
        return self.roll == 0 and self.pitch == 0 and self.yaw == 0


@dataclass
class PickPlan:
    approach_pose: Pose3D
    pick_pose: Pose3D
    grip_width: float


@dataclass
class PlacePlan:
    approach_pose: Pose3D
    place_pose: Pose3D


@dataclass
class RotationStep:
    axis: str          # "roll" / "pitch" / "yaw"
    angle_deg: int     # 현재는 90 고정 가정
    item_before: ItemState
    item_after: ItemState


def item_from_ros_msg(item_msg) -> ItemState:
    return ItemState(
        item_id=item_msg.item_id,
        name=item_msg.name,
        pose=Pose3D(
            x=item_msg.x,
            y=item_msg.y,
            z=item_msg.z,
            roll=item_msg.roll,
            pitch=item_msg.pitch,
            yaw=item_msg.yaw,
        ),
        size=Size3D(
            width=item_msg.width,
            depth=item_msg.depth,
            height=item_msg.height,
        ),
        durability=item_msg.durability,
    )


def placement_from_ros_msg(placement_msg) -> PlacementState:
    return PlacementState(
        object_index=placement_msg.object_index,
        pose=Pose3D(
            x=placement_msg.x,
            y=placement_msg.y,
            z=placement_msg.z,
            roll=placement_msg.roll,
            pitch=placement_msg.pitch,
            yaw=placement_msg.yaw,
        ),
    )



@dataclass
class StagePlan:
    pick_approach_pose: Any
    pick_pose: Any
    pick_retreat_pose: Any
    station_approach_pose: Any
    station_place_pose: Any
    station_retreat_pose: Any


@dataclass
class AlignStep:
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0


@dataclass
class AlignPlan:
    required: bool
    steps: List[AlignStep] = field(default_factory=list)


@dataclass
class BoxPlan:
    station_pick_approach_pose: Any
    station_pick_pose: Any
    station_pick_retreat_pose: Any
    box_approach_pose: Any
    box_place_pose: Any
    box_retreat_pose: Any


@dataclass
class PackingTaskPlan:
    task_index: int
    item: Any
    stage_plan: StagePlan
    align_plan: AlignPlan
    box_plan: BoxPlan
    grip_width: float = 0.0


@dataclass
class PackingExecutionPlan:
    tasks: List[PackingTaskPlan]


def _offset_pose_z(pose, dz: float):
    return type(pose)(
        x=pose.x,
        y=pose.y,
        z=pose.z + dz,
        rx=pose.rx,
        ry=pose.ry,
        rz=pose.rz,
    )