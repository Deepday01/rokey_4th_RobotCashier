from typing import List, Tuple

from .config import *

from .rotation_planner import make_align_plan

def get_approach_pose(target_pose: Pose3D, offset_z: float = APPROACH_OFFSET_Z) -> Pose3D:
    return Pose3D(
        x=target_pose.x,
        y=target_pose.y,
        z=BASE_Z + offset_z, 
        roll=target_pose.roll,
        pitch=target_pose.pitch,
        yaw=target_pose.yaw,
    )

def make_init_object_pick_plan(item: ItemState) -> InitObjectPickPlan:
    pick_pose = item.pose

    # 큐알코드가 항상 위를 보고 있음으로 기본 좌표 틀은 (x,y,base_z + 물체의 h/2, 0, 0, yaw) 가됨
    pick_approach_pose = get_approach_pose(pick_pose)
    pick_retreat_pose = get_approach_pose(pick_pose)

    return InitObjectPickPlan(
        pick_approach_pose=pick_approach_pose,
        pick_pose=pick_pose,
        pick_retreat_pose=pick_retreat_pose,
    )

def build_box_plan(
    placement: PlacementState,
    station_pick_pose: Pose3D,
) -> BoxPlan:
    box_place_pose = placement.pose
    box_approach_pose = get_approach_pose(box_place_pose)
    box_retreat_pose = get_approach_pose(box_place_pose)

    station_pick_approach_pose = get_approach_pose(station_pick_pose)
    station_pick_retreat_pose = get_approach_pose(station_pick_pose)

    return BoxPlan(
        station_pick_approach_pose=station_pick_approach_pose,
        station_pick_pose=station_pick_pose,
        station_pick_retreat_pose=station_pick_retreat_pose,
        box_approach_pose=box_approach_pose,
        box_place_pose=box_place_pose,
        box_retreat_pose=box_retreat_pose,
    )

# 모델 검증
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
    items = [item_from_ros_msg(item_msg) for item_msg in pick_items] # item_msg는 클라이언트의 item1 = Item() 다
    placements = [placement_from_ros_msg(placement_msg) for placement_msg in place_items]
    return items, placements

# 플랜 생성(하나 생성)
def makePlan(task_index: int, item, placement) -> PackingPlan:
    # 초기 물체 픽 플랜 
    init_object_pick_plan = make_init_object_pick_plan(item)
    # 정렬 플랜
    align_plan = make_align_plan(item, placement)

    box_plan = build_box_plan(
        placement=placement,
        # TODO: station_pick_pose 계산해서 만들기
        station_pick_pose=init_object_pick_plan.pick_pose
    )

    return PackingPlan(
        task_index=task_index,
        item=item,
        placement=placement,
        init_object_pick_plan = init_object_pick_plan,
        align_plan=align_plan,
        box_plan=box_plan,
    )

# 플랜 만들기
def make_packing_plan_list(pick_items: List, place_items: List, logger=None) -> PackingPlanList:
    # 아이템 검증 
    validate_request_items_and_places(pick_items=pick_items, place_items=place_items)
    # 클라이언트 데이터 핸들링 가능한 아이템 플레이스먼트 변환
    itemList, placementList = convert_request_to_internal_models(pick_items=pick_items, place_items=place_items)
    # 플랜 리스트 만들기
    planList: List[PackingPlan] = []
    for index, placement in enumerate(placementList, start=1):
        item = itemList[placement.object_index]
        planList.append(makePlan(task_index=index, item=item, placement=placement))
    return PackingPlanList(planList=planList)


def execute_plan_with_callbacks(
    planList: PackingPlanList,
    excute_init_object_pick,
    align_object,
    execute_pick_and_place_to_box,
    logger=None,
) -> None:
    for plan in planList.planList:
        # excute_init_object_pick(plan)
        align_object(plan, plan.align_plan)
        # execute_pick_and_place_to_box(plan, plan.box_plan)


def build_station_place_pose_from_item_z(item) -> Pose3D:
    return Pose3D(
        x=ROTATION_STATION_PLACE_BASE_POSE.x,
        y=ROTATION_STATION_PLACE_BASE_POSE.y,
        z=item.pose.z,
        roll=ROTATION_STATION_PLACE_BASE_POSE.roll,
        pitch=ROTATION_STATION_PLACE_BASE_POSE.pitch,
        yaw=ROTATION_STATION_PLACE_BASE_POSE.yaw,
    )