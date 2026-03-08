from typing import List

from .packing_models import AlignPlan, AlignStep, ItemState, PlacementState


ANGLE_EPSILON = 1e-3


def build_align_plan(item: ItemState, placement: PlacementState, logger=None) -> AlignPlan:
    delta_roll = placement.pose.roll - item.pose.roll
    delta_pitch = placement.pose.pitch - item.pose.pitch
    delta_yaw = placement.pose.yaw - item.pose.yaw

    steps: List[AlignStep] = []

    if abs(delta_roll) > ANGLE_EPSILON:
        steps.append(AlignStep(rx_deg=delta_roll))

    if abs(delta_pitch) > ANGLE_EPSILON:
        steps.append(AlignStep(ry_deg=delta_pitch))

    if abs(delta_yaw) > ANGLE_EPSILON:
        steps.append(AlignStep(rz_deg=delta_yaw))

    return AlignPlan(required=bool(steps), steps=steps)
