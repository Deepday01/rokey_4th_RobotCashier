#!/usr/bin/env python3

from typing import Tuple

from .packing_models import Pose3D, RelativeRotation

from cashier_execute_packing.packing_models import Pose3D
from cashier_execute_packing.packing_models import Pose3D, RelativeRotation



ALLOWED_RIGHT_ANGLES = (0, 90)


# ============================
# Logger helper
# ============================

def _log(logger, level: str, msg: str):
    if logger is None:
        print(f"[POSE_NORMALIZER][{level}] {msg}")
        return

    if level == "debug":
        logger.debug(msg)
    elif level == "warn":
        logger.warn(msg)
    else:
        logger.info(msg)


# ============================
# Angle normalization
# ============================












def _normalize_single_angle(angle: float, logger=None) -> int:
    """
    각도를 0 또는 90 으로 정규화

    규칙
    abs(angle) % 180
    45 기준 반올림
    """

    remainder = abs(angle) % 180.0

    result = 90 if remainder >= 45.0 else 0

    _log(
        logger,
        "debug",
        f"angle normalize: input={angle:.3f}, remainder={remainder:.3f}, result={result}",
    )

    return result


# ============================
# Canonical rotation
# ============================

def _canonicalize_rpy(roll: int, pitch: int, yaw: int, logger=None) -> Tuple[int, int, int]:
    """
    협업 규칙 기반 RPY 치환
    """

    original = (roll, pitch, yaw)

    if (roll, pitch, yaw) == (90, 90, 90):
        new_value = (0, 90, 0)

        _log(
            logger,
            "info",
            f"canonical rotation applied: {original} -> {new_value}",
        )

        return new_value

    return original


# ============================
# RPY normalize
# ============================

def normalize_rpy(roll: float, pitch: float, yaw: float, logger=None) -> Tuple[int, int, int]:

    _log(
        logger,
        "debug",
        f"normalize_rpy input: roll={roll}, pitch={pitch}, yaw={yaw}",
    )

    nr = _normalize_single_angle(roll, logger)
    np = _normalize_single_angle(pitch, logger)
    ny = _normalize_single_angle(yaw, logger)

    nr, np, ny = _canonicalize_rpy(nr, np, ny, logger)

    _log(
        logger,
        "debug",
        f"normalize_rpy result: ({nr}, {np}, {ny})",
    )

    return nr, np, ny


# ============================
# Pose normalize
# ============================
def normalize_pose_rpy(pose: Pose3D, logger=None) -> Pose3D:
    _log(
        logger,
        "info",
        f"[POSE NORMALIZER DISABLED] use raw rpy=({pose.roll},{pose.pitch},{pose.yaw})"
    )
    return pose

# def normalize_pose_rpy(pose: Pose3D, logger=None) -> Pose3D:

#     _log(
#         logger,
#         "debug",
#         f"normalize_pose input: "
#         f"x={pose.x}, y={pose.y}, z={pose.z}, "
#         f"rpy=({pose.roll},{pose.pitch},{pose.yaw})",
#     )

#     nr, np, ny = normalize_rpy(
#         pose.roll,
#         pose.pitch,
#         pose.yaw,
#         logger,
#     )

#     normalized = Pose3D(
#         x=pose.x,
#         y=pose.y,
#         z=pose.z,
#         roll=nr,
#         pitch=np,
#         yaw=ny,
#     )

#     _log(
#         logger,
#         "info",
#         f"pose normalized: rpy ({pose.roll},{pose.pitch},{pose.yaw}) -> ({nr},{np},{ny})",
#     )

#     return normalized


# ============================
# Pick / Place normalize
# ============================


# def normalize_pick_and_place(
#     pick_pose: Pose3D,
#     place_pose: Pose3D,
#     logger=None,
# ) -> Tuple[Pose3D, Pose3D]:

#     _log(logger, "info", "normalize_pick_and_place start")

#     norm_pick = normalize_pose_rpy(pick_pose, logger)
#     norm_place = normalize_pose_rpy(place_pose, logger)

#     _log(
#         logger,
#         "info",
#         f"pick normalized rpy={norm_pick.rpy()} | place normalized rpy={norm_place.rpy()}",
#     )

#     return norm_pick, norm_place


def normalize_pick_and_place(pick_pose: Pose3D, place_pose: Pose3D, logger=None):
    _log(logger, "info", "[POSE NORMALIZER DISABLED]")
    return pick_pose, place_pose


# ============================
# Relative rotation
# ============================

# def relative_rotation_from_pick_to_place(
#     pick_pose: Pose3D,
#     place_pose: Pose3D,
#     logger=None,
# ) -> RelativeRotation:

#     _log(logger, "info", "relative_rotation calculation start")

#     norm_pick = normalize_pose_rpy(pick_pose, logger)
#     norm_place = normalize_pose_rpy(place_pose, logger)

#     _log(
#         logger,
#         "debug",
#         f"normalized pick rpy={norm_pick.rpy()}",
#     )

#     _log(
#         logger,
#         "debug",
#         f"normalized place rpy={norm_place.rpy()}",
#     )

#     rr = 0 if norm_pick.roll == norm_place.roll else 90
#     rp = 0 if norm_pick.pitch == norm_place.pitch else 90
#     ry = 0 if norm_pick.yaw == norm_place.yaw else 90

#     rr, rp, ry = _canonicalize_rpy(rr, rp, ry, logger)

#     result = RelativeRotation(
#         roll=rr,
#         pitch=rp,
#         yaw=ry,
#     )

#     _log(
#         logger,
#         "info",
#         f"relative rotation result: roll={rr}, pitch={rp}, yaw={ry}",
#     )

#     return result


def relative_rotation_from_pick_to_place(
    pick_pose: Pose3D,
    place_pose: Pose3D,
    logger=None,
) -> RelativeRotation:
    roll = place_pose.roll - pick_pose.roll
    pitch = place_pose.pitch - pick_pose.pitch
    yaw = place_pose.yaw - pick_pose.yaw

    _log(
        logger,
        "info",
        f"[RELATIVE ROTATION RAW] roll={roll}, pitch={pitch}, yaw={yaw}"
    )

    return RelativeRotation(
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )


# ============================
# Validation
# ============================

def validate_normalized_rpy(roll: int, pitch: int, yaw: int, logger=None):

    for value in (roll, pitch, yaw):

        if value not in ALLOWED_RIGHT_ANGLES:

            msg = f"Invalid normalized angle: {value}"

            _log(logger, "warn", msg)

            raise ValueError(msg)


def validate_normalized_pose(pose: Pose3D, logger=None):

    validate_normalized_rpy(
        int(pose.roll),
        int(pose.pitch),
        int(pose.yaw),
        logger,
    )