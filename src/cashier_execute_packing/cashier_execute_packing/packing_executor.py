from typing import List, Tuple

from .config import *
from .packing_models import (
    AlignPlan,
    BoxPlan,
    ItemState,
    PackingPlanList,
    PackingPlan,
    PlacementState,
    Pose3D,
    StagePlan,
    item_from_ros_msg,
    placement_from_ros_msg,
)
from .rotation_planner import build_align_plan


def compute_grip_width(item: ItemState) -> float:
    return max(MIN_GRIP_WIDTH, min(item.size.width, item.size.depth) - GRIP_MARGIN)



def build_approach_pose(target_pose: Pose3D, offset_z: float = APPROACH_OFFSET_Z) -> Pose3D:
    return Pose3D(
        x=target_pose.x,
        y=target_pose.y,
        z=target_pose.z + offset_z,
        roll=target_pose.roll,
        pitch=target_pose.pitch,
        yaw=target_pose.yaw,
    )

def build_stage_plan(item) -> StagePlan:
    pick_pose = item.pose

    pick_approach_pose = build_approach_pose(pick_pose)
    pick_retreat_pose = build_approach_pose(pick_pose)

    station_place_pose = Pose3D(
        x=ROTATION_STATION_PLACE_BASE_POSE.x,
        y=ROTATION_STATION_PLACE_BASE_POSE.y,
        z=pick_pose.z,
        roll=ROTATION_STATION_PLACE_BASE_POSE.roll,
        pitch=ROTATION_STATION_PLACE_BASE_POSE.pitch,
        yaw=ROTATION_STATION_PLACE_BASE_POSE.yaw,
    )

    return StagePlan(
        pick_approach_pose=pick_approach_pose,
        pick_pose=pick_pose,
        pick_retreat_pose=pick_retreat_pose,
        station_approach_pose=ROTATION_STATION_APPROACH_POSE,
        station_place_pose=station_place_pose,
        station_retreat_pose=ROTATION_STATION_RETREAT_POSE,
    )

def build_box_plan(
    placement: PlacementState,
    station_pick_pose: Pose3D,
) -> BoxPlan:
    box_place_pose = placement.pose
    box_approach_pose = build_approach_pose(box_place_pose)
    box_retreat_pose = build_approach_pose(box_place_pose)

    station_pick_approach_pose = build_approach_pose(station_pick_pose)
    station_pick_retreat_pose = build_approach_pose(station_pick_pose)

    return BoxPlan(
        station_pick_approach_pose=station_pick_approach_pose,
        station_pick_pose=station_pick_pose,
        station_pick_retreat_pose=station_pick_retreat_pose,
        box_approach_pose=box_approach_pose,
        box_place_pose=box_place_pose,
        box_retreat_pose=box_retreat_pose,
    )


def validate_request_items_and_places(pick_items: List, place_items: List) -> None:
    if not pick_items:
        raise ValueError("pick_items is empty")
    if not place_items:
        raise ValueError("place_items is empty")
    used_indices = set()
    for index, placement in enumerate(place_items):
        object_index = placement.object_index
        if object_index < 0 or object_index >= len(pick_items):
            raise ValueError(f"Invalid object_index at place_items[{index}]: {object_index}")
        if object_index in used_indices:
            raise ValueError(f"Duplicated object_index at place_items[{index}]: {object_index}")
        used_indices.add(object_index)


def convert_request_to_internal_models(pick_items: List, place_items: List) -> Tuple[List[ItemState], List[PlacementState]]:
    items = [item_from_ros_msg(item_msg) for item_msg in pick_items]
    placements = [placement_from_ros_msg(placement_msg) for placement_msg in place_items]
    return items, placements


def build_single_plan(task_index: int, item, placement, grip_width: float) -> PackingPlan:
    stage_plan = build_stage_plan(item)
    align_plan = build_align_plan(item, placement)

    final_station_pose = stage_plan.station_place_pose

    box_plan = build_box_plan(
        placement=placement,
        station_pick_pose=final_station_pose,
    )

    return PackingPlan(
        task_index=task_index,
        item=item,
        placement=placement,
        stage_plan=stage_plan,
        align_plan=align_plan,
        box_plan=box_plan,
        grip_width=grip_width,
    )

def build_execution_plan_from_request(pick_items: List, place_items: List, logger=None) -> PackingPlanList:
    validate_request_items_and_places(pick_items=pick_items, place_items=place_items)
    items, placements = convert_request_to_internal_models(pick_items=pick_items, place_items=place_items)
    tasks: List[PackingPlan] = []
    for task_index, placement in enumerate(placements, start=1):
        item = items[placement.object_index]
        tasks.append(build_single_plan(task_index=task_index, item=item, placement=placement, grip_width = compute_grip_width(item)))
    return PackingPlanList(planList=tasks)


def execute_plan_with_callbacks(
    planList: PackingPlanList,
    on_pick_and_stage_on_rotation_station,
    on_align_object_on_rotation_station,
    on_pick_and_place_to_box,
    logger=None,
) -> None:
    for plan in planList.planList:
        # on_pick_and_stage_on_rotation_station(plan, plan.stage_plan)
        on_align_object_on_rotation_station(plan, plan.align_plan)
        # on_pick_and_place_to_box(plan, plan.box_plan)


def build_station_place_pose_from_item_z(item) -> Pose3D:
    return Pose3D(
        x=ROTATION_STATION_PLACE_BASE_POSE.x,
        y=ROTATION_STATION_PLACE_BASE_POSE.y,
        z=item.pose.z,
        roll=ROTATION_STATION_PLACE_BASE_POSE.roll,
        pitch=ROTATION_STATION_PLACE_BASE_POSE.pitch,
        yaw=ROTATION_STATION_PLACE_BASE_POSE.yaw,
    )