import rclpy
from rclpy.node import Node

from cashier_interfaces.srv import ComputePackingPlan

from .packing_service_core import (
    CHECKPOINT_PATH,
    GREEDY,
    SUPPORT_THRESHOLD,
    device,
    load_models,
    objects_from_request_items,
    placements_to_response_msgs,
    run_inference_once,
)


class PlanPackingNode(Node):
    def __init__(self):
        super().__init__("plan_packing_node")
        self.order_model, self.placement_model = load_models(CHECKPOINT_PATH)

        self.srv = self.create_service(
            ComputePackingPlan,
            "/plan_packing/compute_plan",
            self.handle_compute_plan,
        )

        self.get_logger().info(f"device: {device}")
        self.get_logger().info(f"checkpoint loaded: {CHECKPOINT_PATH}")
        self.get_logger().info("beam search removed: using single-pass inference only")
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

            placements, total_reward = run_inference_once(
                self.order_model,
                self.placement_model,
                objects,
                support_threshold=SUPPORT_THRESHOLD,
                greedy=GREEDY,
            )

            response.success = len(placements) > 0
            response.placements = placements_to_response_msgs(placements)

            self.get_logger().info(
                f"plan done | success={response.success} | "
                f"placed={len(placements)}/{len(objects)} | reward={total_reward:.3f}"
            )
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
