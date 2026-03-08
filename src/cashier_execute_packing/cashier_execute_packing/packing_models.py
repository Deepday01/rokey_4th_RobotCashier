from dataclasses import dataclass, field
from typing import List, Tuple


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
class AlignStep:
    rx_deg: float = 0.0
    ry_deg: float = 0.0
    rz_deg: float = 0.0


@dataclass
class StagePlan:
    pick_approach_pose: Pose3D
    pick_pose: Pose3D
    pick_retreat_pose: Pose3D
    station_approach_pose: Pose3D
    station_place_pose: Pose3D
    station_retreat_pose: Pose3D


@dataclass
class AlignPlan:
    required: bool
    steps: List[AlignStep] = field(default_factory=list)


@dataclass
class BoxPlan:
    station_pick_approach_pose: Pose3D
    station_pick_pose: Pose3D
    station_pick_retreat_pose: Pose3D
    box_approach_pose: Pose3D
    box_place_pose: Pose3D
    box_retreat_pose: Pose3D


@dataclass
class PackingTaskPlan:
    task_index: int
    item: ItemState
    placement: PlacementState
    stage_plan: StagePlan
    align_plan: AlignPlan
    box_plan: BoxPlan
    grip_width: float


@dataclass
class PackingExecutionPlan:
    tasks: List[PackingTaskPlan] = field(default_factory=list)


def pose_from_ros_fields(msg) -> Pose3D:
    return Pose3D(
        x=msg.x,
        y=msg.y,
        z=msg.z,
        roll=msg.roll,
        pitch=msg.pitch,
        yaw=msg.yaw,
    )


def item_from_ros_msg(item_msg) -> ItemState:
    return ItemState(
        item_id=item_msg.item_id,
        name=item_msg.name,
        pose=pose_from_ros_fields(item_msg),
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
        pose=pose_from_ros_fields(placement_msg),
    )


def offset_pose_z(pose: Pose3D, dz: float) -> Pose3D:
    return Pose3D(
        x=pose.x,
        y=pose.y,
        z=pose.z + dz,
        roll=pose.roll,
        pitch=pose.pitch,
        yaw=pose.yaw,
    )
