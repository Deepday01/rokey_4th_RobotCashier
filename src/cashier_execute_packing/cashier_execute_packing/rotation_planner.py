#!/usr/bin/env python3

from typing import List, Tuple

from .packing_models import ItemState, RelativeRotation, RotationStep
from .pose_normalizer import (
    normalize_pose_rpy,
    relative_rotation_from_pick_to_place,
)


# ============================
# Logger helper
# ============================

def _log(logger, level: str, msg: str):
    if logger is None:
        print(f"[ROTATION_PLANNER][{level}] {msg}")
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

def _rotation_tuple(rotation: RelativeRotation) -> Tuple[int, int, int]:
    return (rotation.roll, rotation.pitch, rotation.yaw)


def is_zero_rotation(rotation: RelativeRotation) -> bool:
    return rotation.roll == 0 and rotation.pitch == 0 and rotation.yaw == 0


def rotation_key(rotation: RelativeRotation) -> Tuple[int, int, int]:
    return (rotation.roll, rotation.pitch, rotation.yaw)


# ============================
# Rotation state transition
# ============================

def next_rotation_state(rotation: RelativeRotation, logger=None) -> Tuple[str, RelativeRotation]:
    """
    현재 상대 회전 상태에서 다음 회전 축과 다음 상태를 반환한다.

    요구사항 기준 매핑:
    - 0 0 0 -> 종료
    - 90 0 0 -> roll  회전 후 0 0 0
    - 0 90 0 -> pitch 회전 후 0 0 0
    - 0 0 90 -> yaw   회전 후 0 0 0
    - 90 90 0 -> yaw   회전 후 0 0 90 -> yaw 회전 후 0 0 0
    - 90 0 90 -> pitch 회전 후 0 90 0 -> pitch 회전 후 0 0 0
    - 0 90 90 -> roll  회전 후 90 0 0 -> roll 회전 후 0 0 0
    - 90 90 90 은 canonical 단계에서 0 90 0 으로 들어온다고 가정
    """
    key = rotation_key(rotation)

    _log(logger, "debug", f"next_rotation_state input={key}")

    mapping = {
        (0, 0, 0): (None, RelativeRotation(0, 0, 0)),

        (90, 0, 0): ("roll", RelativeRotation(0, 0, 0)),
        (0, 90, 0): ("pitch", RelativeRotation(0, 0, 0)),
        (0, 0, 90): ("yaw", RelativeRotation(0, 0, 0)),

        (90, 90, 0): ("yaw", RelativeRotation(0, 0, 90)),
        (90, 0, 90): ("pitch", RelativeRotation(0, 90, 0)),
        (0, 90, 90): ("roll", RelativeRotation(90, 0, 0)),
    }

    if key not in mapping:
        raise ValueError(f"Unsupported rotation state: {key}")

    axis, next_state = mapping[key]

    _log(
        logger,
        "info",
        f"rotation transition: current={key}, axis={axis}, next={rotation_key(next_state)}",
    )

    return axis, next_state


# ============================
# Item transform by 90deg rotation
# ============================

def rotate_item_state_90(item: ItemState, axis: str, logger=None) -> ItemState:
    """
    물체를 현재 중심 기준으로 90도 회전시켰다고 가정하고 상태를 갱신한다.

    현재 규칙:
    - 중심점 자체 x,y,z는 유지
    - 회전에 따라 크기축 swap
    - 자세 rpy는 axis에 해당하는 값만 0으로 만들고
      남은 값들은 요구 플로우에 맞춰 swap
    - 입력 item.pose 는 normalize_pose_rpy 로 정규화되어 있다고 가정

    축별 크기 swap:
    - roll  회전: depth <-> height
    - pitch 회전: width <-> height
    - yaw   회전: width <-> depth
    """
    if axis not in ("roll", "pitch", "yaw"):
        raise ValueError(f"Unsupported axis: {axis}")

    before = item.copy()

    norm_pose = normalize_pose_rpy(before.pose, logger)

    r = int(norm_pose.roll)
    p = int(norm_pose.pitch)
    y = int(norm_pose.yaw)

    w = before.size.width
    d = before.size.depth
    h = before.size.height

    _log(
        logger,
        "info",
        f"rotate_item_state_90 start: item={before.name}, axis={axis}, "
        f"before_rpy=({r},{p},{y}), before_size=({w},{d},{h})",
    )

    if axis == "roll":
        # roll 축 회전 -> pitch <-> yaw 성격 swap
        new_r = 0
        new_p = y
        new_y = p

        new_w = w
        new_d = h
        new_h = d

    elif axis == "pitch":
        # pitch 축 회전 -> roll <-> yaw 성격 swap
        new_r = y
        new_p = 0
        new_y = r

        new_w = h
        new_d = d
        new_h = w

    else:  # yaw
        # yaw 축 회전 -> roll <-> pitch 성격 swap
        new_r = p
        new_p = r
        new_y = 0

        new_w = d
        new_d = w
        new_h = h

    after = before.copy()
    after.pose.roll = new_r
    after.pose.pitch = new_p
    after.pose.yaw = new_y

    after.size.width = new_w
    after.size.depth = new_d
    after.size.height = new_h

    _log(
        logger,
        "info",
        f"rotate_item_state_90 done: item={after.name}, axis={axis}, "
        f"after_rpy=({after.pose.roll},{after.pose.pitch},{after.pose.yaw}), "
        f"after_size=({after.size.width},{after.size.depth},{after.size.height})",
    )

    return after


# ============================
# Rotation step builder
# ============================


def build_rotation_steps_from_relative_rotation(
    item,
    relative_rotation,
    logger=None,
) -> List[RotationStep]:
    """
    현재는 raw rotation 기반 테스트 단계.
    90/0 정규화 없이 들어오므로,
    우선 회전 step 생성은 비활성화하고 빈 리스트 반환.
    """

    _log(
        logger,
        "info",
        f"build_rotation_steps_from_relative_rotation start: "
        f"roll={relative_rotation.roll}, "
        f"pitch={relative_rotation.pitch}, "
        f"yaw={relative_rotation.yaw}",
    )

    steps: List[RotationStep] = []

    _log(
        logger,
        "info",
        "rotation step generation disabled for raw-rpy test mode: total_steps=0",
    )

    return steps

# def build_rotation_steps_from_relative_rotation(
#     item: ItemState,
#     relative_rotation: RelativeRotation,
#     logger=None,
# ) -> List[RotationStep]:
#     """
#     상대 회전값을 기반으로 실제 회전 step 목록을 만든다.
#     """
#     _log(
#         logger,
#         "info",
#         f"build_rotation_steps_from_relative_rotation start: rel={_rotation_tuple(relative_rotation)}",
#     )

#     steps: List[RotationStep] = []
#     current_item = item.copy()
#     current_rotation = RelativeRotation(
#         relative_rotation.roll,
#         relative_rotation.pitch,
#         relative_rotation.yaw,
#     )

#     loop_guard = 0

#     while not is_zero_rotation(current_rotation):
#         loop_guard += 1
#         if loop_guard > 10:
#             raise RuntimeError("Rotation planning loop exceeded safe limit")

#         axis, next_rotation = next_rotation_state(current_rotation, logger)

#         if axis is None:
#             break

#         item_before = current_item.copy()
#         item_after = rotate_item_state_90(item_before, axis, logger)

#         step = RotationStep(
#             axis=axis,
#             angle_deg=90,
#             item_before=item_before,
#             item_after=item_after,
#         )
#         steps.append(step)

#         _log(
#             logger,
#             "info",
#             f"rotation step added: idx={len(steps)}, axis={axis}, "
#             f"state_before={_rotation_tuple(current_rotation)}, "
#             f"state_after={_rotation_tuple(next_rotation)}",
#         )

#         current_item = item_after
#         current_rotation = next_rotation

#     _log(
#         logger,
#         "info",
#         f"build_rotation_steps_from_relative_rotation done: total_steps={len(steps)}",
#     )

#     return steps


def build_rotation_steps_from_pick_and_place(
    item,
    place_pose,
    logger=None,
):
    from cashier_execute_packing.pose_normalizer import (
        normalize_pose_rpy,
        relative_rotation_from_pick_to_place,
    )

    _log(logger, "info", f"build_rotation_steps_from_pick_and_place start: item={item.name}")

    normalized_pick = normalize_pose_rpy(item.pose, logger)
    normalized_place = normalize_pose_rpy(place_pose, logger)

    rel = relative_rotation_from_pick_to_place(
        normalized_pick,
        normalized_place,
        logger,
    )

    _log(
        logger,
        "info",
        f"relative rotation for item={item.name}: {rel.roll}, {rel.pitch}, {rel.yaw}",
    )

    temp_item = item.copy()
    temp_item.pose = normalized_pick

    return build_rotation_steps_from_relative_rotation(
        temp_item,
        rel,
        logger,
    )


# ============================
# Utility
# ============================

def summarize_rotation_steps(steps, logger=None) -> str:
    if not steps:
        summary = "no rotation required"
        _log(logger, "info", summary)
        return summary

    tokens = []
    for idx, step in enumerate(steps, start=1):
        tokens.append(f"{idx}:{step.axis}{step.angle_deg}")

    summary = " | ".join(tokens)
    _log(logger, "info", f"rotation summary: {summary}")
    return summary