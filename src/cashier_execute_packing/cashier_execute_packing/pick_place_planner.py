#!/usr/bin/env python3

from typing import Tuple

from .packing_models import (
    ItemState,
    PlacementState,
    PickPlan,
    PlacePlan,
    Pose3D,
)
from .pose_normalizer import normalize_pose_rpy


# ============================
# Config
# ============================

DEFAULT_APPROACH_OFFSET_Z = 100.0
DEFAULT_MIN_GRIP_WIDTH = 5.0
DEFAULT_GRIP_MARGIN = 2.0


# ============================
# Logger helper
# ============================

def _log(logger, level: str, msg: str):
    if logger is None:
        print(f"[PICK_PLACE_PLANNER][{level}] {msg}")
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
# Basic helpers
# ============================

def _xy_grip_candidates(item: ItemState) -> Tuple[float, float]:
    """
    현재 물체의 상부에서 접근한다고 가정할 때,
    그리퍼가 닫아야 할 후보 길이는 width / depth 이다.
    """
    return item.size.width, item.size.depth


def compute_grip_width(
    item: ItemState,
    margin: float = DEFAULT_GRIP_MARGIN,
    min_width: float = DEFAULT_MIN_GRIP_WIDTH,
    logger=None,
) -> float:
    """
    pick 시 사용할 grip width 계산

    규칙
    - width, depth 중 더 작은 값을 선택
    - margin 을 뺀다
    - min_width 보다 작아지면 min_width 사용
    """
    width_candidate, depth_candidate = _xy_grip_candidates(item)
    base_width = min(width_candidate, depth_candidate)
    grip_width = max(min_width, base_width - margin)

    _log(
        logger,
        "info",
        f"compute_grip_width: item={item.name}, "
        f"width={item.size.width}, depth={item.size.depth}, "
        f"base={base_width}, margin={margin}, result={grip_width}",
    )

    return grip_width


def build_approach_pose(
    target_pose: Pose3D,
    offset_z: float = DEFAULT_APPROACH_OFFSET_Z,
    logger=None,
) -> Pose3D:
    """
    목표 pose 상부의 안전 접근 pose 생성
    """
    approach = Pose3D(
        x=target_pose.x,
        y=target_pose.y,
        z=target_pose.z + offset_z,
        roll=target_pose.roll,
        pitch=target_pose.pitch,
        yaw=target_pose.yaw,
    )

    _log(
        logger,
        "debug",
        f"build_approach_pose: target=({target_pose.x},{target_pose.y},{target_pose.z},"
        f"{target_pose.roll},{target_pose.pitch},{target_pose.yaw}) "
        f"-> approach=({approach.x},{approach.y},{approach.z},"
        f"{approach.roll},{approach.pitch},{approach.yaw})",
    )

    return approach


def normalize_item_for_pick(item: ItemState, logger=None) -> ItemState:
    """
    pick 계획용 item pose 정규화
    """
    normalized = item.copy()
    normalized.pose = normalize_pose_rpy(item.pose, logger)

    _log(
        logger,
        "info",
        f"normalize_item_for_pick: item={item.name}, "
        f"rpy {item.pose.rpy()} -> {normalized.pose.rpy()}",
    )

    return normalized


def normalize_placement_for_place(placement: PlacementState, logger=None) -> PlacementState:
    """
    place 계획용 placement pose 정규화
    """
    normalized = placement.copy()
    normalized.pose = normalize_pose_rpy(placement.pose, logger)

    _log(
        logger,
        "info",
        f"normalize_placement_for_place: object_index={placement.object_index}, "
        f"rpy {placement.pose.rpy()} -> {normalized.pose.rpy()}",
    )

    return normalized


# ============================
# Pick planner
# ============================

def build_pick_plan(
    item: ItemState,
    approach_offset_z: float = DEFAULT_APPROACH_OFFSET_Z,
    grip_margin: float = DEFAULT_GRIP_MARGIN,
    min_grip_width: float = DEFAULT_MIN_GRIP_WIDTH,
    logger=None,
) -> PickPlan:
    """
    pick 계획 생성

    전제
    - 회전 루프 수행 후 item 이 pick 가능한 상태(0,0,0 또는 협업 규칙상 정렬 상태)
    - top-down 접근
    - pick pose 는 item 중심 pose 사용
    """
    normalized_item = normalize_item_for_pick(item, logger)

    grip_width = compute_grip_width(
        normalized_item,
        margin=grip_margin,
        min_width=min_grip_width,
        logger=logger,
    )

    pick_pose = Pose3D(
        x=normalized_item.pose.x,
        y=normalized_item.pose.y,
        z=normalized_item.pose.z,
        roll=normalized_item.pose.roll,
        pitch=normalized_item.pose.pitch,
        yaw=normalized_item.pose.yaw,
    )

    approach_pose = build_approach_pose(
        pick_pose,
        offset_z=approach_offset_z,
        logger=logger,
    )

    plan = PickPlan(
        approach_pose=approach_pose,
        pick_pose=pick_pose,
        grip_width=grip_width,
    )

    _log(
        logger,
        "info",
        f"build_pick_plan done: item={item.name}, "
        f"approach_pose=({approach_pose.x},{approach_pose.y},{approach_pose.z},"
        f"{approach_pose.roll},{approach_pose.pitch},{approach_pose.yaw}), "
        f"pick_pose=({pick_pose.x},{pick_pose.y},{pick_pose.z},"
        f"{pick_pose.roll},{pick_pose.pitch},{pick_pose.yaw}), "
        f"grip_width={grip_width}",
    )

    return plan


# ============================
# Place planner
# ============================

def build_place_plan(
    placement: PlacementState,
    approach_offset_z: float = DEFAULT_APPROACH_OFFSET_Z,
    logger=None,
) -> PlacePlan:
    """
    place 계획 생성

    전제
    - placement.pose 를 기준으로 목표 위치/자세를 그대로 사용
    - 상부 접근 후 하강
    """
    normalized_placement = normalize_placement_for_place(placement, logger)

    place_pose = Pose3D(
        x=normalized_placement.pose.x,
        y=normalized_placement.pose.y,
        z=normalized_placement.pose.z,
        roll=normalized_placement.pose.roll,
        pitch=normalized_placement.pose.pitch,
        yaw=normalized_placement.pose.yaw,
    )

    approach_pose = build_approach_pose(
        place_pose,
        offset_z=approach_offset_z,
        logger=logger,
    )

    plan = PlacePlan(
        approach_pose=approach_pose,
        place_pose=place_pose,
    )

    _log(
        logger,
        "info",
        f"build_place_plan done: object_index={placement.object_index}, "
        f"approach_pose=({approach_pose.x},{approach_pose.y},{approach_pose.z},"
        f"{approach_pose.roll},{approach_pose.pitch},{approach_pose.yaw}), "
        f"place_pose=({place_pose.x},{place_pose.y},{place_pose.z},"
        f"{place_pose.roll},{place_pose.pitch},{place_pose.yaw})",
    )

    return plan


# ============================
# Combined planner
# ============================

def build_pick_and_place_plan(
    item: ItemState,
    placement: PlacementState,
    approach_offset_z: float = DEFAULT_APPROACH_OFFSET_Z,
    grip_margin: float = DEFAULT_GRIP_MARGIN,
    min_grip_width: float = DEFAULT_MIN_GRIP_WIDTH,
    logger=None,
) -> Tuple[PickPlan, PlacePlan]:
    """
    item / placement 기준으로 pick, place plan 을 함께 생성
    """
    _log(
        logger,
        "info",
        f"build_pick_and_place_plan start: item={item.name}, object_index={placement.object_index}",
    )

    pick_plan = build_pick_plan(
        item=item,
        approach_offset_z=approach_offset_z,
        grip_margin=grip_margin,
        min_grip_width=min_grip_width,
        logger=logger,
    )

    place_plan = build_place_plan(
        placement=placement,
        approach_offset_z=approach_offset_z,
        logger=logger,
    )

    _log(
        logger,
        "info",
        f"build_pick_and_place_plan done: item={item.name}, object_index={placement.object_index}",
    )

    return pick_plan, place_plan