from typing import List

from .config import AlignPlan, AlignStep, ItemState, PlacementState


ANGLE_EPSILON = 1e-3



def make_align_plan(item: ItemState, placement: PlacementState) -> AlignPlan:

    # delta_rx = placement.pose.roll - item.pose.roll
    # delta_ry = placement.pose.pitch - item.pose.pitch
    # delta_rz = placement.pose.yaw - item.pose.yaw

    # rx = _normalize_rotation_delta(delta_rx)
    # ry = _normalize_rotation_delta(delta_ry)
    # rz = _normalize_rotation_delta(delta_rz)

    rx = _normalize_rotation_delta(placement.pose.roll)
    ry = _normalize_rotation_delta(placement.pose.pitch)
    rz = _normalize_rotation_delta(placement.pose.yaw)
    
    steps = []
    # ==============z가 0일때==============
    if rx == 0 and ry == 0 and rz == 0:
        # (사전 작업) 90 0 rz > 0 0 0	
        # 회전 불필요	
        # 박스로 이동
        return AlignPlan(
            required= False,
            steps=steps,
        )

    if rx == 90 and ry == 0 and rz == 0:
        # (사전 작업) 90 0 rz > 90 0 0
        # rz 90 회전 > ry 90 회전> rz 90 회전 
        # 박스로 이동
        steps.append(AlignStep(rz_deg=90))
        steps.append(AlignStep(ry_deg=90))
        steps.append(AlignStep(rz_deg=90))
        return AlignPlan(
            required= True,
            steps=steps,
        )

    if rx == 90 and ry == 90 and rz == 0:
        # (사전 작업) 90 90 rz > 90 90 0 
        # ry 90 회전 > rz 90 회전
        # 박스로 이동
        steps.append(AlignStep(ry_deg=90))
        steps.append(AlignStep(rz_deg=90))
        return AlignPlan(
            required= True,
            steps=steps,
        )

    if rx == 0 and ry == 90 and rz == 0:
        # (사전 작업) 0 90 rz	> 0 90 0
        # ry 90 회전
        # 박스로 이동
        steps.append(AlignStep(ry_deg=90))
        return AlignPlan(
            required= True,
            steps=steps,
        )
    
    # ==============z가 90일때==============
    if rx == 0 and ry == 0 and rz == 90:
        # (사전 작업) 90 0 rz > 0 0 0	
        # 회전 불필요	
        # 박스로 이동
        return AlignPlan(
            required= False,
            steps=steps,
        )

    if rx == 90 and ry == 0 and rz == 90:
        # (사전 작업) 90 0 rz > 90 0 0
        # rz 90 회전 > ry 90 회전> rz 90 회전 
        # 박스로 이동
        steps.append(AlignStep(ry_deg=90))
        steps.append(AlignStep(rz_deg=90))
        return AlignPlan(
            required= True,
            steps=steps,
        )

    if rx == 90 and ry == 90 and rz == 90:
        # (사전 작업) 90 90 rz > 90 90 0 
        # ry 90 회전 > rz 90 회전
        # 박스로 이동
        steps.append(AlignStep(rz_deg=90))
        steps.append(AlignStep(ry_deg=90))
        steps.append(AlignStep(rz_deg=90))
        return AlignPlan(
            required= True,
            steps=steps,
        )

    if rx == 0 and ry == 90 and rz == 90:
        # (사전 작업) 0 90 rz	> 0 90 0
        # ry 90 회전
        # 박스로 이동
        steps.append(AlignStep(rz_deg=90))
        steps.append(AlignStep(ry_deg=90))
        return AlignPlan(
            required= True,
            steps=steps,
        )

    raise ValueError(f"AlignPlan 못 만듬: rx: {rx}, ry: {ry}, rz: {rz}")

    


def _normalize_rotation_delta(angle: float) -> int:
    # 해당값은 0 or 90 으로만 이루어짐
    normalized = angle % 180.0 

    # 절대값 처리
    if normalized < 0:
        normalized = -normalized

    # 90 아니면 0 이기에 이렇게 처리
    if 80 < normalized and normalized < 100:
        return 90
    elif 0 <= normalized and normalized < 20:
        return 0
    else:
        raise ValueError(f"invalid rotation delta {normalized}")