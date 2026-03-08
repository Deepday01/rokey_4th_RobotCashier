#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Optional

from .packing_models import (
    ItemState,
    PlacementState,
    PickPlan,
    PlacePlan,
    RotationStep,
    item_from_ros_msg,
    placement_from_ros_msg,
    StagePlan,
    AlignStep,
    AlignPlan,
    BoxPlan,
)
from .pose_normalizer import (
    normalize_pose_rpy,
    relative_rotation_from_pick_to_place,
)
from .rotation_planner import (
    build_rotation_steps_from_pick_and_place,
    summarize_rotation_steps,
)
from .pick_place_planner import (
    build_pick_plan,
    build_place_plan,
)




# ============================
# Logger helper
# ============================

def _log(logger, level: str, msg: str):
    if logger is None:
        print(f"[PACKING_EXECUTOR][{level}] {msg}")
        return

    if level == "debug":
        logger.debug(msg)
    elif level == "info":
        logger.info(msg)
    elif level == "warn":
        logger.warn(msg)
    else:
        logger.info(msg)


# ============================
# Execution models
# ============================

from dataclasses import dataclass
from typing import Any, List


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
    """
    전체 포장 작업 실행 계획
    """
    tasks: List[PackingTaskPlan]

def build_stage_plan(item, placement):
    pick_pose = item.pick_pose

    return StagePlan(
        pick_approach_pose=pick_pose.approach_pose,
        pick_pose=pick_pose.target_pose,
        pick_retreat_pose=pick_pose.retreat_pose,
        station_approach_pose=placement.rotation_station.approach_pose,
        station_place_pose=placement.rotation_station.place_pose,
        station_retreat_pose=placement.rotation_station.retreat_pose,
    )


def build_align_plan(item, placement):
    delta_rx = placement.place_pose.rx - item.pick_pose.target_pose.rx
    delta_ry = placement.place_pose.ry - item.pick_pose.target_pose.ry
    delta_rz = placement.place_pose.rz - item.pick_pose.target_pose.rz

    steps = []

    if abs(delta_rx) > 1e-3:
        steps.append(AlignStep(rx_deg=delta_rx))

    if abs(delta_ry) > 1e-3:
        steps.append(AlignStep(ry_deg=delta_ry))

    if abs(delta_rz) > 1e-3:
        steps.append(AlignStep(rz_deg=delta_rz))

    return AlignPlan(
        required=len(steps) > 0,
        steps=steps,
    )


def build_box_plan(item, placement):
    return BoxPlan(
        station_pick_approach_pose=placement.rotation_station.pick_approach_pose,
        station_pick_pose=placement.rotation_station.pick_pose,
        station_pick_retreat_pose=placement.rotation_station.pick_retreat_pose,
        box_approach_pose=placement.place_pose.approach_pose,
        box_place_pose=placement.place_pose.target_pose,
        box_retreat_pose=placement.place_pose.retreat_pose,
    )




# ============================
# Validation
# ============================

def validate_request_items_and_places(
    pick_items: List,
    place_items: List,
    logger=None,
) -> None:
    """
    ROS action request 수준 검증
    """
    _log(
        logger,
        "info",
        f"validate_request_items_and_places: pick_items={len(pick_items)}, place_items={len(place_items)}",
    )

    if len(pick_items) == 0:
        raise ValueError("pick_items is empty")

    if len(place_items) == 0:
        raise ValueError("place_items is empty")

    used_indices = set()

    for i, placement in enumerate(place_items):
        obj_index = placement.object_index

        if obj_index < 0 or obj_index >= len(pick_items):
            raise ValueError(
                f"Invalid object_index at place_items[{i}]: {obj_index}"
            )

        if obj_index in used_indices:
            raise ValueError(
                f"Duplicated object_index at place_items[{i}]: {obj_index}"
            )

        used_indices.add(obj_index)

    _log(logger, "info", "validate_request_items_and_places: success")


# ============================
# ROS msg -> internal model
# ============================

def convert_request_to_internal_models(
    pick_items: List,
    place_items: List,
    logger=None,
) -> (List[ItemState], List[PlacementState]):
    """
    ROS msg 리스트를 내부 모델 리스트로 변환
    """
    _log(logger, "info", "convert_request_to_internal_models start")

    items: List[ItemState] = []
    placements: List[PlacementState] = []

    for idx, item_msg in enumerate(pick_items):
        item = item_from_ros_msg(item_msg)
        items.append(item)

        _log(
            logger,
            "debug",
            f"pick_item[{idx}] -> ItemState("
            f"id={item.item_id}, name={item.name}, "
            f"pose=({item.pose.x},{item.pose.y},{item.pose.z},{item.pose.roll},{item.pose.pitch},{item.pose.yaw}), "
            f"size=({item.size.width},{item.size.depth},{item.size.height}))",
        )

    for idx, placement_msg in enumerate(place_items):
        placement = placement_from_ros_msg(placement_msg)
        placements.append(placement)

        _log(
            logger,
            "debug",
            f"place_item[{idx}] -> PlacementState("
            f"object_index={placement.object_index}, "
            f"pose=({placement.pose.x},{placement.pose.y},{placement.pose.z},{placement.pose.roll},{placement.pose.pitch},{placement.pose.yaw}))",
        )

    _log(
        logger,
        "info",
        f"convert_request_to_internal_models done: items={len(items)}, placements={len(placements)}",
    )

    return items, placements


# ============================
# Single task planner
# ============================
def build_single_task_plan(task_index, item, placement, logger=None):
    stage_plan = build_stage_plan(item, placement)
    align_plan = build_align_plan(item, placement)
    box_plan = build_box_plan(item, placement)

    return PackingTaskPlan(
        task_index=task_index,
        item=item,
        stage_plan=stage_plan,
        align_plan=align_plan,
        box_plan=box_plan,
        grip_width=item.grip.width_mm,
    )

# ============================
# Full execution planner
# ============================

def build_execution_plan_from_request(
    pick_items: List,
    place_items: List,
    logger=None,
) -> PackingExecutionPlan:
    """
    action request 의 pick_items / place_items 로부터 전체 실행 계획 생성
    """
    _log(logger, "info", "build_execution_plan_from_request start")

    validate_request_items_and_places(
        pick_items=pick_items,
        place_items=place_items,
        logger=logger,
    )

    items, placements = convert_request_to_internal_models(
        pick_items=pick_items,
        place_items=place_items,
        logger=logger,
    )

    task_plans: List[PackingTaskPlan] = []

    for task_index, placement in enumerate(placements, start=1):
        item = items[placement.object_index]

        plan = build_single_task_plan(
            task_index=task_index,
            item=item,
            placement=placement,
            logger=logger,
        )
        task_plans.append(plan)

    execution_plan = PackingExecutionPlan(tasks=task_plans)

    _log(
        logger,
        "info",
        f"build_execution_plan_from_request done: total_tasks={len(execution_plan.tasks)}",
    )

    return execution_plan


# ============================
# Debug summary helpers
# ============================

def summarize_task_plan(task: PackingTaskPlan, logger=None) -> str:
    rotation_count = len(task.rotation_steps)

    summary = (
        f"[TASK {task.task_index}] "
        f"item={task.item.name}, "
        f"object_index={task.object_index}, "
        f"normalized_pick_rpy={task.normalized_pick_item.pose.rpy()}, "
        f"normalized_place_rpy={task.normalized_place.pose.rpy()}, "
        f"rotation_steps={rotation_count}, "
        f"final_pick_rpy={task.final_pick_item.pose.rpy()}, "
        f"final_pick_size=({task.final_pick_item.size.width},"
        f"{task.final_pick_item.size.depth},{task.final_pick_item.size.height}), "
        f"grip_width={task.pick_plan.grip_width}, "
        f"pick_pose=({task.pick_plan.pick_pose.x},"
        f"{task.pick_plan.pick_pose.y},{task.pick_plan.pick_pose.z},"
        f"{task.pick_plan.pick_pose.roll},{task.pick_plan.pick_pose.pitch},{task.pick_plan.pick_pose.yaw}), "
        f"place_pose=({task.place_plan.place_pose.x},"
        f"{task.place_plan.place_pose.y},{task.place_plan.place_pose.z},"
        f"{task.place_plan.place_pose.roll},{task.place_plan.place_pose.pitch},{task.place_plan.place_pose.yaw})"
    )

    _log(logger, "info", summary)
    return summary


def summarize_execution_plan(plan: PackingExecutionPlan, logger=None) -> str:
    if plan.is_empty():
        summary = "execution plan is empty"
        _log(logger, "warn", summary)
        return summary

    lines = [f"execution plan summary: total_tasks={len(plan.tasks)}"]

    for task in plan.tasks:
        lines.append(
            f"- task={task.task_index}, item={task.item.name}, "
            f"object_index={task.object_index}, rotation_steps={len(task.rotation_steps)}, "
            f"grip_width={task.pick_plan.grip_width}"
        )

    summary = "\n".join(lines)
    _log(logger, "info", summary)
    return summary


# ============================
# Optional execution bridge
# ============================
def execute_plan_with_callbacks(
    plan,
    on_pick_and_stage_on_rotation_station,
    on_align_object_on_rotation_station,
    on_pick_and_place_to_box,
    logger=None,
):
    """
    전체 execution plan 을 순회하며
    각 task를 3단계로 실행한다.

    1. pick_and_stage_on_rotation_station
    2. align_object_on_rotation_station
    3. pick_and_place_to_box
    """

    for task in plan.tasks:
        on_pick_and_stage_on_rotation_station(task, task.stage_plan)
        on_align_object_on_rotation_station(task, task.align_plan)
        on_pick_and_place_to_box(task, task.box_plan)
