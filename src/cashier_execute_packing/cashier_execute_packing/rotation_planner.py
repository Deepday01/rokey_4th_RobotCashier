from typing import List

from .config import AlignPlan, AlignStep, ItemState, PlacementState


ANGLE_EPSILON = 1e-3



def make_align_plan(item, placement: PlacementState) -> AlignPlan:
    delta_rx = placement.pose.roll - item.pose.roll
    delta_ry = placement.pose.pitch - item.pose.pitch
    delta_rz = placement.pose.yaw - item.pose.yaw

    rx = _normalize_rotation_delta(delta_rx)
    ry = _normalize_rotation_delta(delta_ry)
    rz = _normalize_rotation_delta(delta_rz)

    steps = []

    if rx != 0:
        steps.append(AlignStep(rx_deg=rx))

    if ry != 0:
        steps.append(AlignStep(ry_deg=ry))

    if rz != 0:
        steps.append(AlignStep(rz_deg=rz))

    return AlignPlan(
        required=len(steps) > 0,
        steps=steps,
    )


def _normalize_rotation_delta(angle: float) -> int:
    normalized = angle % 360.0

    if normalized > 180.0:
        normalized -= 360.0

    candidates = [-180, -90, 0, 90, 180]
    return min(candidates, key=lambda x: abs(x - normalized))