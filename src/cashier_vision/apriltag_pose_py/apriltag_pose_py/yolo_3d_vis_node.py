#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
YOLO 2D/3D Visualizer (ROS2) - upgraded version

기능
- RGB, CameraInfo, Depth 구독
- depth 없으면 YOLO 2D bbox만 표시
- depth 있으면 preview용 3D bounding box 표시
- /yolo_3d/debug_image publish
- /yolo_3d/markers publish

특징
- 중앙 ROI depth 사용 -> 배경 영향 감소
- depth 최신 여부 체크
- 화면에 현재 모드 표시 (2D only / 3D preview)
- model_path만 바꾸면 yolov8n.pt -> best.pt 교체 가능
"""

import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge

from ultralytics import YOLO


class Yolo3DVisNode(Node):
    def __init__(self):
        super().__init__("yolo_3d_vis_node")

        # -------------------------
        # Parameters
        # -------------------------
        self.declare_parameter("rgb_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")

        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("conf", 0.35)
        self.declare_parameter("imgsz", 640)

        # depth params
        self.declare_parameter("depth_scale", 0.001)   # 16UC1(mm) -> meters
        self.declare_parameter("min_depth_m", 0.15)
        self.declare_parameter("max_depth_m", 3.0)
        self.declare_parameter("default_box_depth_m", 0.08)
        self.declare_parameter("min_valid_depth_pixels", 50)
        self.declare_parameter("depth_timeout_sec", 0.5)

        # ROI depth 안정화
        self.declare_parameter("center_roi_scale", 0.5)   # bbox 중앙 50% 영역만 depth 계산
        self.declare_parameter("min_box_pixels", 12)

        self.rgb_topic = self.get_parameter("rgb_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.model_path = self.get_parameter("model_path").value
        self.device = self.get_parameter("device").value
        self.conf = float(self.get_parameter("conf").value)
        self.imgsz = int(self.get_parameter("imgsz").value)

        self.depth_scale = float(self.get_parameter("depth_scale").value)
        self.min_depth_m = float(self.get_parameter("min_depth_m").value)
        self.max_depth_m = float(self.get_parameter("max_depth_m").value)
        self.default_box_depth_m = float(self.get_parameter("default_box_depth_m").value)
        self.min_valid_depth_pixels = int(self.get_parameter("min_valid_depth_pixels").value)
        self.depth_timeout_sec = float(self.get_parameter("depth_timeout_sec").value)

        self.center_roi_scale = float(self.get_parameter("center_roi_scale").value)
        self.min_box_pixels = int(self.get_parameter("min_box_pixels").value)

        self.bridge = CvBridge()

        # camera intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.camera_frame = "camera_color_optical_frame"
        self.have_cam_info = False

        # latest depth
        self.latest_depth = None
        self.latest_depth_time = None
        self.have_depth = False

        self.get_logger().info(f"Loading YOLO model: {self.model_path}")
        self.model = YOLO(self.model_path)

        self.debug_pub = self.create_publisher(Image, "/yolo_3d/debug_image", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/yolo_3d/markers", 10)

        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
        self.create_subscription(Image, self.rgb_topic, self.on_rgb, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth, 10)

        self.get_logger().info(f"RGB topic        : {self.rgb_topic}")
        self.get_logger().info(f"Depth topic      : {self.depth_topic}")
        self.get_logger().info(f"CameraInfo topic : {self.camera_info_topic}")
        self.get_logger().info("Ready.")

    # -------------------------
    # Callbacks
    # -------------------------
    def on_camera_info(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])
        self.have_cam_info = True
        if msg.header.frame_id:
            self.camera_frame = msg.header.frame_id

    def on_depth(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_depth = depth.copy()
            self.latest_depth_time = self.get_clock().now()
            self.have_depth = True
        except Exception as e:
            self.get_logger().warn(f"Depth conversion failed: {e}")
            self.have_depth = False

    def on_rgb(self, msg: Image):
        if not self.have_cam_info:
            self.get_logger().warn("Waiting for CameraInfo...")
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
            return

        depth_valid = self.is_depth_recent()

        if depth_valid and self.latest_depth is not None:
            debug_img, marker_array = self.process_frame_3d(frame, self.latest_depth)
        else:
            debug_img, marker_array = self.process_frame_2d(frame)

        debug_msg = self.bridge.cv2_to_imgmsg(debug_img, encoding="bgr8")
        debug_msg.header = msg.header
        self.debug_pub.publish(debug_msg)
        self.marker_pub.publish(marker_array)

    # -------------------------
    # Helpers
    # -------------------------
    def is_depth_recent(self):
        if not self.have_depth or self.latest_depth_time is None:
            return False
        now = self.get_clock().now()
        dt = (now - self.latest_depth_time).nanoseconds / 1e9
        return dt <= self.depth_timeout_sec

    def run_yolo(self, frame_bgr):
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False
        )
        if len(results) == 0:
            return None
        return results[0]

    def get_center_roi(self, x1, y1, x2, y2):
        """bbox 중앙부만 depth 계산용으로 잘라냄"""
        bw = x2 - x1
        bh = y2 - y1

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        rw = int(bw * self.center_roi_scale)
        rh = int(bh * self.center_roi_scale)

        rx1 = cx - rw // 2
        ry1 = cy - rh // 2
        rx2 = cx + rw // 2
        ry2 = cy + rh // 2

        return rx1, ry1, rx2, ry2

    def process_frame_2d(self, frame_bgr):
        debug_img = frame_bgr.copy()
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        result = self.run_yolo(frame_bgr)
        if result is None or result.boxes is None or len(result.boxes) == 0:
            cv2.putText(debug_img, "No detections", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(debug_img, "YOLO 2D only (no depth)", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            return debug_img, marker_array

        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(debug_img, "YOLO 2D only (no depth)", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return debug_img, marker_array

    def process_frame_3d(self, frame_bgr, depth_raw):
        debug_img = frame_bgr.copy()
        marker_array = MarkerArray()

        clear_marker = Marker()
        clear_marker.action = Marker.DELETEALL
        marker_array.markers.append(clear_marker)

        result = self.run_yolo(frame_bgr)
        if result is None or result.boxes is None or len(result.boxes) == 0:
            cv2.putText(debug_img, "No detections", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(debug_img, "3D preview mode", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            return debug_img, marker_array

        names = result.names
        h, w = frame_bgr.shape[:2]
        marker_id = 0

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            label = names.get(cls_id, str(cls_id))

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            if (x2 - x1) < self.min_box_pixels or (y2 - y1) < self.min_box_pixels:
                continue

            # 2D bbox
            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(debug_img, f"{label} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 중앙 ROI
            rx1, ry1, rx2, ry2 = self.get_center_roi(x1, y1, x2, y2)
            rx1 = max(0, min(w - 1, rx1))
            ry1 = max(0, min(h - 1, ry1))
            rx2 = max(0, min(w - 1, rx2))
            ry2 = max(0, min(h - 1, ry2))

            cv2.rectangle(debug_img, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

            depth_roi = depth_raw[ry1:ry2, rx1:rx2]
            if depth_roi.size == 0:
                continue

            # depth image -> meters
            depth_m = depth_roi.astype(np.float32)

            # 16UC1(mm)일 가능성 처리
            if depth_m.dtype != np.float32:
                depth_m = depth_m.astype(np.float32)

            depth_m = depth_m * self.depth_scale

            valid = np.isfinite(depth_m)
            valid &= (depth_m > self.min_depth_m)
            valid &= (depth_m < self.max_depth_m)

            valid_depths = depth_m[valid]
            if valid_depths.size < self.min_valid_depth_pixels:
                cv2.putText(debug_img, "depth invalid", (x1, min(h - 10, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                continue

            # robust depth
            z_center = float(np.median(valid_depths))
            z_front = float(np.percentile(valid_depths, 10))
            z_back = float(np.percentile(valid_depths, 90))

            if (z_back - z_front) < 0.02:
                z_front = max(self.min_depth_m, z_center - self.default_box_depth_m / 2.0)
                z_back = min(self.max_depth_m, z_center + self.default_box_depth_m / 2.0)

            # bbox의 좌우/상하 범위는 z_center 기준으로 계산
            x_min = (x1 - self.cx) * z_center / self.fx
            x_max = (x2 - self.cx) * z_center / self.fx
            y_min = (y1 - self.cy) * z_center / self.fy
            y_max = (y2 - self.cy) * z_center / self.fy

            corners_3d = np.array([
                [x_min, y_min, z_front],
                [x_max, y_min, z_front],
                [x_max, y_max, z_front],
                [x_min, y_max, z_front],
                [x_min, y_min, z_back],
                [x_max, y_min, z_back],
                [x_max, y_max, z_back],
                [x_min, y_max, z_back],
            ], dtype=np.float32)

            corners_2d = []
            ok_project = True
            for X, Y, Z in corners_3d:
                uv = self.project_point(X, Y, Z)
                if uv is None:
                    ok_project = False
                    break
                corners_2d.append(uv)

            if ok_project:
                self.draw_3d_box(debug_img, corners_2d)

            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0
            center_z = z_center

            cv2.putText(
                debug_img,
                f"XYZ=({center_x:.2f},{center_y:.2f},{center_z:.2f})m",
                (x1, min(h - 10, y2 + 22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
            )

            cv2.putText(
                debug_img,
                f"Zf={z_front:.2f} Zb={z_back:.2f}",
                (x1, min(h - 30, y2 + 42)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2
            )

            marker = self.make_box_marker(marker_id, corners_3d)
            marker_array.markers.append(marker)
            marker_id += 1

        cv2.putText(debug_img, "YOLO 3D preview mode", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)

        return debug_img, marker_array

    def project_point(self, X, Y, Z):
        if Z <= 1e-6:
            return None
        u = int((X * self.fx / Z) + self.cx)
        v = int((Y * self.fy / Z) + self.cy)
        return (u, v)

    def draw_3d_box(self, img, pts):
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        for a, b in edges:
            cv2.line(img, pts[a], pts[b], (255, 0, 255), 2)

    def make_box_marker(self, marker_id, corners_3d):
        marker = Marker()
        marker.header.frame_id = self.camera_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "yolo_3d_boxes"
        marker.id = marker_id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.scale.x = 0.005

        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0

        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for a, b in edges:
            pa = Point()
            pb = Point()
            pa.x, pa.y, pa.z = map(float, corners_3d[a])
            pb.x, pb.y, pb.z = map(float, corners_3d[b])
            marker.points.append(pa)
            marker.points.append(pb)

        return marker


def main():
    rclpy.init()
    node = Yolo3DVisNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()