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

BASKET_WORLD_ORIGIN = (200,20,5)
def objects_from_request_items(items):
    if len(items) > MAX_OBJECTS:
        raise ValueError(f"received {len(items)} items, but MAX_OBJECTS={MAX_OBJECTS}")

    objects = []

    for episode_index, item in enumerate(items):
        sx = int(item.width)
        sy = int(item.depth)
        sz = int(item.height)
        durability = int(item.durability)

        if sx <= 0 or sy <= 0 or sz <= 0:
            raise ValueError(
                f"invalid size at index {episode_index}: ({sx}, {sy}, {sz})"
            )

        objects.append({
            "name": item.name if item.name else f"item_{episode_index}",
            "item_id": item.item_id,
            "size": (sx, sy, sz),
            "durability": durability,
            "base_index": episode_index,
            "episode_index": episode_index,
        })

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
