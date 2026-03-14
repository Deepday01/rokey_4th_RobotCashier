import rclpy
from rclpy.node import Node

from cashier_interfaces.srv import ComputePackingPlan
from cashier_interfaces.msg import Placement as PlacementMsg

from .inference import (
    CHECKPOINT_PATH,
    MAX_OBJECTS,
    N_TRIALS,
    GREEDY,
    SUPPORT_THRESHOLD,
    device,
    load_models,
    run_multi_trial,
    run
)

BASKET_WORLD_ORIGIN = (220.0, 20.0, 5.0)

TRAINED_ITEM_SPECS = {
    "cream": {
        "train_name": "애크논크림",
        "size": (125, 35, 20),
        "durability": 3,
    },
    "caramel1": {
        "train_name": "카라멜1",
        "size": (70, 45, 25),
        "durability": 1,
    },
    "caramel2": {
        "train_name": "카라멜2",
        "size": (70, 45, 25),
        "durability": 1,
    },
    "insect": {
        "train_name": "나비 블럭",
        "size": (80, 50, 40),
        "durability": 4,
    },
    "candy": {
        "train_name": "아이셔",
        "size": (110, 75, 40),
        "durability": 2,
    },
    "halls": {
        "train_name": "이클립스",
        "size": (80, 50, 15),
        "durability": 5,
    },
    "eclipse_red": {
        "train_name": "이클립스빨강",
        "size": (80, 45, 25),
        "durability": 5,
    },
    "eclipse_gre": {
        "train_name": "이클립스노랑",
        "size": (80, 45, 25),
        "durability": 5,
    },
}


def normalize_item_name(name: str) -> str:
    if not name:
        return ""

    normalized = name.strip().lower()

    # 혹시 caramel 로만 들어오는 경우도 대비
    if normalized == "caramel":
        return "caramel1"

    return normalized


def objects_from_request_items(items):
    if len(items) > MAX_OBJECTS:
        raise ValueError(f"received {len(items)} items, but MAX_OBJECTS={MAX_OBJECTS}")

    objects = []

    for episode_index, item in enumerate(items):
        raw_name = item.name if item.name else f"item_{episode_index}"
        mapped_name = normalize_item_name(raw_name)

        if mapped_name not in TRAINED_ITEM_SPECS:
            raise ValueError(
                f"unknown item name at index {episode_index}: '{raw_name}'. "
                f"supported names: {list(TRAINED_ITEM_SPECS.keys())}"
            )

        spec = TRAINED_ITEM_SPECS[mapped_name]
        sx, sy, sz = spec["size"]
        durability = spec["durability"]

        objects.append({
            # RL 모델에는 학습 당시 이름/크기/내구도 사용
            "name": spec["train_name"],
            "item_id": item.item_id,
            "size": (sx, sy, sz),
            "durability": durability,

            # 응답 순서 복원을 위해 원래 인덱스는 유지
            "base_index": episode_index,
            "episode_index": episode_index,

            # 필요하면 디버깅용으로 원본 이름도 남길 수 있음
            "request_name": raw_name,
        })

    objects.sort(key=lambda obj: obj["name"])
    return objects
def placements_to_response_msgs(placements):
    response_msgs = []

    bx, by, bz = BASKET_WORLD_ORIGIN

    for p in placements:
        msg = PlacementMsg()
        msg.object_index = int(p.base_index)

        sx, sy, sz = p.size

        local_cx = float(p.position[0] + sx / 2.0)
        local_cy = float(p.position[1] + sy / 2.0)
        local_cz = float(p.position[2] + sz / 2.0)

        msg.x = bx + local_cx
        msg.y = by - local_cy
        msg.z = bz + local_cz

        msg.roll = float(p.rotation_rpy[0])
        msg.pitch = float(p.rotation_rpy[1])
        msg.yaw = float(p.rotation_rpy[2])

        response_msgs.append(msg)

    return response_msgs
class PlanPackingNode(Node):
    def __init__(self):
        super().__init__("plan_packing_node")

        self.get_logger().info(f"device: {device}")
        self.get_logger().info(f"loading checkpoint: {CHECKPOINT_PATH}")
        self.get_logger().info(
            f"multi-trial inference | n_trials={N_TRIALS} | greedy={GREEDY}"
        )

        self.order_model, self.placement_model = load_models(CHECKPOINT_PATH)

        self.srv = self.create_service(
            ComputePackingPlan,
            "/plan_packing/compute_plan",
            self.handle_compute_plan,
        )

        self.get_logger().info("service ready: /plan_packing/compute_plan")

    def handle_compute_plan(self, request, response):
        try:
            if len(request.items) == 0:
                response.success = False
                response.placements = []
                self.get_logger().warn("empty request received")
                return response

            objects = objects_from_request_items(request.items)
            self.get_logger().info(f"request : {request.items}")
            self.get_logger().info(f"objects : {objects}")
            self.get_logger().info(f"received {len(objects)} items")

            best_placements, best_reward, best_plan, trial_logs, best_post_logs = run(objects)

            for trial, reward, placed_count in trial_logs:
                self.get_logger().info(
                    f"trial {trial:3d} | reward {reward:7.3f} | placed {placed_count}/{len(objects)}"
                )

            if best_post_logs is not None:
                for line in best_post_logs:
                    self.get_logger().info(str(line))

            response.success = len(best_placements) > 0
            response.placements = placements_to_response_msgs(best_placements)

            self.get_logger().info(
                f"plan done | success={response.success} | "
                f"placed={len(best_placements)}/{len(objects)} | reward={best_reward:.3f}"
            )

            self.get_logger().info(f"best plan: {best_plan}")

            return response

        except Exception as e:
            response.success = False
            response.placements = []
            self.get_logger().error(f"compute_plan failed: {e}")
            return response


def main(args=None):
    rclpy.init(args=args)

    node = PlanPackingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("shutting down node")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
