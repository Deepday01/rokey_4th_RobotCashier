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

@dataclass
class PackingTaskPlan:
    task_index: int
    object_index: int
    item: ItemState
    placement: PlacementState
    normalized_pick_item: ItemState
    normalized_place: PlacementState
    rotation_steps: List[RotationStep]
    final_pick_item: ItemState
    pick_plan: PickPlan
    place_plan: PlacePlan


@dataclass
class PackingExecutionPlan:
    tasks: List[PackingTaskPlan]

    def is_empty(self) -> bool:
        return len(self.tasks) == 0


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

def build_single_task_plan(
    task_index: int,
    item: ItemState,
    placement: PlacementState,
    logger=None,
) -> PackingTaskPlan:
    """
    단일 item / placement 에 대한 전체 실행 계획 생성
    """
    _log(
        logger,
        "info",
        f"[TASK PLAN {task_index}] start: item={item.name}, object_index={placement.object_index}",
    )

    normalized_pick_item = item.copy()
    normalized_pick_item.pose = normalize_pose_rpy(item.pose, logger)

    normalized_place = placement.copy()
    normalized_place.pose = normalize_pose_rpy(placement.pose, logger)

    _log(
        logger,
        "info",
        f"[TASK PLAN {task_index}] normalized pick rpy={normalized_pick_item.pose.rpy()} "
        f"| normalized place rpy={normalized_place.pose.rpy()}",
    )

    relative_rotation = relative_rotation_from_pick_to_place(
        normalized_pick_item.pose,
        normalized_place.pose,
        logger,
    )

    _log(
        logger,
        "info",
        f"[TASK PLAN {task_index}] relative rotation="
        f"({relative_rotation.roll}, {relative_rotation.pitch}, {relative_rotation.yaw})",
    )

    rotation_steps = build_rotation_steps_from_pick_and_place(
        item=normalized_pick_item,
        place_pose=normalized_place.pose,
        logger=logger,
    )

    summarize_rotation_steps(rotation_steps, logger)

    final_pick_item = normalized_pick_item.copy()
    if rotation_steps:
        final_pick_item = rotation_steps[-1].item_after.copy()

    _log(
        logger,
        "info",
        f"[TASK PLAN {task_index}] final pick state after rotation: "
        f"rpy={final_pick_item.pose.rpy()}, "
        f"size=({final_pick_item.size.width}, {final_pick_item.size.depth}, {final_pick_item.size.height})",
    )

    pick_plan = build_pick_plan(
        item=final_pick_item,
        logger=logger,
    )

    place_plan = build_place_plan(
        placement=normalized_place,
        logger=logger,
    )

    plan = PackingTaskPlan(
        task_index=task_index,
        object_index=placement.object_index,
        item=item.copy(),
        placement=placement.copy(),
        normalized_pick_item=normalized_pick_item,
        normalized_place=normalized_place,
        rotation_steps=rotation_steps,
        final_pick_item=final_pick_item,
        pick_plan=pick_plan,
        place_plan=place_plan,
    )

    _log(
        logger,
        "info",
        f"[TASK PLAN {task_index}] done: item={item.name}, "
        f"rotation_steps={len(rotation_steps)}, grip_width={pick_plan.grip_width}",
    )

    return plan


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
    plan: PackingExecutionPlan,
    on_task_start=None,
    on_rotation_step=None,
    on_pick_plan=None,
    on_place_plan=None,
    on_task_done=None,
    logger=None,
) -> None:
    """
    실제 로봇 제어는 외부 callback 으로 연결하는 브리지 함수

    callback signature 예시
    - on_task_start(task_plan)
    - on_rotation_step(task_plan, step_index, step)
    - on_pick_plan(task_plan, pick_plan)
    - on_place_plan(task_plan, place_plan)
    - on_task_done(task_plan)
    """
    _log(logger, "info", "execute_plan_with_callbacks start")

    for task in plan.tasks:
        _log(
            logger,
            "info",
            f"execute task start: task={task.task_index}, item={task.item.name}",
        )

        if on_task_start is not None:
            on_task_start(task)

        for step_index, step in enumerate(task.rotation_steps, start=1):
            _log(
                logger,
                "info",
                f"rotation execute request: task={task.task_index}, "
                f"step={step_index}, axis={step.axis}, angle={step.angle_deg}",
            )
            if on_rotation_step is not None:
                on_rotation_step(task, step_index, step)

        _log(
            logger,
            "info",
            f"pick execute request: task={task.task_index}, grip_width={task.pick_plan.grip_width}",
        )
        if on_pick_plan is not None:
            on_pick_plan(task, task.pick_plan)

        _log(
            logger,
            "info",
            f"place execute request: task={task.task_index}",
        )
        if on_place_plan is not None:
            on_place_plan(task, task.place_plan)

        if on_task_done is not None:
            on_task_done(task)

        _log(
            logger,
            "info",
            f"execute task done: task={task.task_index}, item={task.item.name}",
        )

    _log(logger, "info", "execute_plan_with_callbacks done")