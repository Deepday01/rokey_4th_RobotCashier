from typing import List

from .packing_models import AlignPlan, AlignStep, ItemState, PlacementState


ANGLE_EPSILON = 1e-3



def build_align_plan(item, placement: PlacementState) -> AlignPlan:
    delta_rx = placement.pose.roll - item.pose.roll
    delta_ry = placement.pose.pitch - item.pose.pitch
    delta_rz = placement.pose.yaw - item.pose.yaw

    steps = []

    if abs(delta_rx) > 1e-6:
        steps.append(AlignStep(rx_deg=delta_rx))

    if abs(delta_ry) > 1e-6:
        steps.append(AlignStep(ry_deg=delta_ry))

    if abs(delta_rz) > 1e-6:
        steps.append(AlignStep(rz_deg=delta_rz))

    return AlignPlan(
        required=len(steps) > 0,
        steps=steps,
    )
