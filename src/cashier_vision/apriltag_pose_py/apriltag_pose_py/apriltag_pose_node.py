#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import math
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from pupil_apriltags import Detector
from pyzbar.pyzbar import decode as qr_decode

# 패키지명은 네 환경에 맞게 수정
from apriltag_pose_py import ScanItems
from apriltag_pose_py import Item


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class VisionScanItemsActionServer(Node):
    def __init__(self):
        super().__init__("vision_scan_items_action_server")

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------
        self.declare_parameter("color_topic", "/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/color/camera_info")
        self.declare_parameter("use_compressed", False)

        self.declare_parameter("debug_image_topic", "/vision/debug_image")

        self.declare_parameter("model_path", "yolov8n.pt")
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("max_det", 30)
        self.declare_parameter("imgsz", 960)

        self.declare_parameter("scan_duration_sec", 2.0)
        self.declare_parameter("feedback_hz", 5.0)

        self.declare_parameter("tag_family", "tag36h11")
        self.declare_parameter("tag_size_m", 0.05)
        self.declare_parameter("camera_frame_override", "")
        self.declare_parameter("decimate", 1.0)
        self.declare_parameter("sigma", 0.0)
        self.declare_parameter("nthreads", 2)
        self.declare_parameter("refine_edges", True)

        # 16UC1 depth(mm)면 0.001, 32FC1(m)이면 1.0
        self.declare_parameter("depth_value_scale_to_m", 0.001)

        # bbox center 주변 depth median kernel half size
        self.declare_parameter("depth_kernel", 2)

        # 작업영역 태그 월드좌표(mm)
        # 예:
        # {"0":[0,0], "1":[150,0], "2":[300,0], "3":[300,200], "4":[150,200], "5":[0,200]}
        self.declare_parameter(
            "tag_world_points_json",
            '{"0":[0,0], "1":[150,0], "2":[300,0], "3":[300,200], "4":[150,200], "5":[0,200]}'
        )

        # YOLO 클래스명 강제 지정하고 싶으면 JSON 사용
        # 예: {"0":"snack_a","1":"snack_b"}
        self.declare_parameter("class_name_map_json", "{}")

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.use_compressed = bool(self.get_parameter("use_compressed").value)
        self.debug_image_topic = self.get_parameter("debug_image_topic").value

        self.model_path = self.get_parameter("model_path").value
        self.conf_thres = float(self.get_parameter("conf_thres").value)
        self.iou_thres = float(self.get_parameter("iou_thres").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.imgsz = int(self.get_parameter("imgsz").value)

        self.scan_duration_sec = float(self.get_parameter("scan_duration_sec").value)
        self.feedback_hz = float(self.get_parameter("feedback_hz").value)

        self.tag_family = self.get_parameter("tag_family").value
        self.tag_size_m = float(self.get_parameter("tag_size_m").value)
        self.camera_frame_override = self.get_parameter("camera_frame_override").value
        self.depth_value_scale_to_m = float(self.get_parameter("depth_value_scale_to_m").value)
        self.depth_kernel = int(self.get_parameter("depth_kernel").value)

        self.tag_world_points: Dict[int, Tuple[float, float]] = {
            int(k): (float(v[0]), float(v[1]))
            for k, v in json.loads(self.get_parameter("tag_world_points_json").value).items()
        }

        self.class_name_map = {
            int(k): str(v)
            for k, v in json.loads(self.get_parameter("class_name_map_json").value).items()
        }

        # -------------------------------------------------
        # Core objects
        # -------------------------------------------------
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.detector = Detector(
            families=self.tag_family,
            nthreads=int(self.get_parameter("nthreads").value),
            quad_decimate=float(self.get_parameter("decimate").value),
            quad_sigma=float(self.get_parameter("sigma").value),
            refine_edges=bool(self.get_parameter("refine_edges").value),
        )

        # -------------------------------------------------
        # Camera info
        # -------------------------------------------------
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.have_cam_info = False
        self.camera_frame = "camera_frame"

        # -------------------------------------------------
        # Latest buffers
        # -------------------------------------------------
        self.latest_color = None
        self.latest_color_stamp = None
        self.latest_depth = None
        self.latest_depth_stamp = None

        self.latest_items: List[Item] = []
        self.latest_debug = None

        # -------------------------------------------------
        # Publishers
        # -------------------------------------------------
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)

        # -------------------------------------------------
        # Subscribers
        # -------------------------------------------------
        self.create_subscription(CameraInfo, self.camera_info_topic, self.on_camera_info, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth, 10)

        if self.use_compressed:
            self.create_subscription(CompressedImage, self.color_topic, self.on_color_compressed, 10)
            self.get_logger().info(f"Subscribing CompressedImage: {self.color_topic}")
        else:
            self.create_subscription(Image, self.color_topic, self.on_color, 10)
            self.get_logger().info(f"Subscribing Image: {self.color_topic}")

        # -------------------------------------------------
        # Action server
        # -------------------------------------------------
        self._action_server = ActionServer(
            self,
            ScanItems,
            "/vision/scan_items",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"CameraInfo topic: {self.camera_info_topic}")
        self.get_logger().info(f"YOLO model: {self.model_path}")
        self.get_logger().info(f"AprilTag world map: {self.tag_world_points}")
        self.get_logger().info("VisionScanItemsActionServer ready.")

    # =====================================================
    # Action
    # =====================================================
    def goal_callback(self, goal_request):
        if not goal_request.start_vision:
            self.get_logger().warn("Rejected goal: start_vision == False")
            return GoalResponse.REJECT
        self.get_logger().info("Accepted goal: start_vision == True")
        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        self.get_logger().info("Received cancel request.")
        return CancelResponse.ACCEPT

    async def execute_callback(self, goal_handle):
        self.get_logger().info("ScanItems action started.")

        feedback_msg = ScanItems.Feedback()
        result = ScanItems.Result()
        result.success = False
        result.items_scan = []

        start_time = time.time()
        end_time = start_time + self.scan_duration_sec

        # item_id 기준으로 최고 confidence 항목만 유지
        best_by_id: Dict[str, Tuple[float, Item]] = {}

        interval = 1.0 / max(self.feedback_hz, 1.0)

        while time.time() < end_time:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().warn("ScanItems canceled.")
                return result

            runtime = int(time.time() - start_time)
            feedback_msg.runtime = runtime
            goal_handle.publish_feedback(feedback_msg)

            # 최신 프레임 한번 처리
            self.process_latest_frame()

            for item in self.latest_items:
                item_key = item.item_id if item.item_id else item.name
                conf = getattr(item, "_score", 0.0) if hasattr(item, "_score") else 0.0

                if item_key not in best_by_id:
                    best_by_id[item_key] = (conf, item)
                else:
                    if conf > best_by_id[item_key][0]:
                        best_by_id[item_key] = (conf, item)

            await self._sleep_non_blocking(interval)

        final_items = [v[1] for v in best_by_id.values()]

        # x, y 기준 정렬
        final_items.sort(key=lambda m: (m.y, m.x))

        result.success = len(final_items) > 0
        result.items_scan = final_items

        goal_handle.succeed()
        self.get_logger().info(f"ScanItems done. count={len(final_items)}")
        return result

    async def _sleep_non_blocking(self, sec: float):
        # rclpy async callback 안에서 가볍게 대기
        end_t = time.time() + sec
        while time.time() < end_t:
            await rclpy.task.Future()

    # =====================================================
    # Subscriptions
    # =====================================================
    def on_camera_info(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.have_cam_info = True

        if self.camera_frame_override:
            self.camera_frame = self.camera_frame_override
        elif msg.header.frame_id:
            self.camera_frame = msg.header.frame_id

    def on_depth(self, msg: Image):
        try:
            if msg.encoding in ("16UC1", "mono16"):
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            else:
                depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_depth = depth
            self.latest_depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def on_color(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_color = frame
            self.latest_color_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Color conversion failed: {e}")

    def on_color_compressed(self, msg: CompressedImage):
        try:
            np_arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            self.latest_color = frame
            self.latest_color_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Compressed color conversion failed: {e}")

    # =====================================================
    # Main process
    # =====================================================
    def process_latest_frame(self):
        if self.latest_color is None or self.latest_depth is None:
            return
        if not self.have_cam_info:
            return

        color = self.latest_color.copy()
        depth = self.latest_depth.copy()
        stamp = self.latest_color_stamp

        debug_img = color.copy()

        # 1) AprilTag -> workspace homography
        H_img_to_ws, tag_debug_info = self.compute_workspace_homography(color, debug_img)

        if H_img_to_ws is None:
            cv2.putText(
                debug_img,
                "Workspace homography not ready",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            self.latest_items = []
            self.publish_debug(debug_img, stamp)
            return

        # 2) YOLO detection
        results = self.model.predict(
            source=color,
            conf=self.conf_thres,
            iou=self.iou_thres,
            max_det=self.max_det,
            imgsz=self.imgsz,
            verbose=False,
        )

        items: List[Item] = []

        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            names = result.names if hasattr(result, "names") else self.model.names

            if boxes is not None:
                for box in boxes:
                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    x1 = clamp(x1, 0, color.shape[1] - 1)
                    y1 = clamp(y1, 0, color.shape[0] - 1)
                    x2 = clamp(x2, 0, color.shape[1] - 1)
                    y2 = clamp(y2, 0, color.shape[0] - 1)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    cx = int((x1 + x2) * 0.5)
                    cy = int((y1 + y2) * 0.5)

                    # 3) depth -> z(mm)
                    z_mm = self.get_depth_median_mm(depth, cx, cy, self.depth_kernel)
                    if z_mm is None:
                        continue

                    # 4) pixel -> workspace(mm)
                    ws_center = self.transform_point(H_img_to_ws, (cx, cy))
                    if ws_center is None:
                        continue
                    x_mm, y_mm = ws_center

                    # 5) bbox width/height approximate in workspace(mm)
                    w_mm, h_mm = self.estimate_bbox_size_mm(H_img_to_ws, x1, y1, x2, y2)

                    # 6) yaw estimate in workspace frame
                    roi = color[y1:y2, x1:x2]
                    yaw_deg = self.estimate_yaw_deg_from_roi(roi, (x1, y1), H_img_to_ws)

                    if yaw_deg is None:
                        yaw_deg = 0.0

                    # 7) QR decode in ROI
                    qr_text = self.decode_qr_from_roi(roi)

                    # 8) class name
                    cls_name = self.class_name_map.get(cls_id, str(names.get(cls_id, f"class_{cls_id}")))

                    # 9) item_id
                    item_id = qr_text if qr_text else f"{cls_name}_{cx}_{cy}"

                    item = Item()
                    item.item_id = item_id
                    item.name = cls_name
                    item.width = float(w_mm if w_mm is not None else 0.0)
                    item.height = float(h_mm if h_mm is not None else 0.0)
                    item.depth = 0.0
                    item.quantity = 1
                    item.x = float(x_mm)
                    item.y = float(y_mm)
                    item.z = float(z_mm)
                    item.pitch = 0.0
                    item.yaw = float(yaw_deg)

                    # 내부 점수 저장용
                    setattr(item, "_score", conf)

                    items.append(item)

                    # debug draw
                    cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 200, 0), 2)
                    cv2.circle(debug_img, (cx, cy), 4, (0, 0, 255), -1)

                    label = f"{cls_name} {conf:.2f}"
                    cv2.putText(
                        debug_img,
                        label,
                        (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )

                    info2 = f"x={x_mm:.1f} y={y_mm:.1f} z={z_mm:.1f} yaw={yaw_deg:.1f}"
                    cv2.putText(
                        debug_img,
                        info2,
                        (x1, min(debug_img.shape[0] - 10, y2 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                    if qr_text:
                        cv2.putText(
                            debug_img,
                            f"QR:{qr_text}",
                            (x1, min(debug_img.shape[0] - 30, y2 + 40)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 255),
                            2,
                        )

        self.latest_items = items
        self.publish_debug(debug_img, stamp)

    # =====================================================
    # AprilTag -> workspace homography
    # =====================================================
    def compute_workspace_homography(self, color_bgr, debug_img):
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)

        tags = self.detector.detect(
            gray,
            estimate_tag_pose=False,
        )

        img_pts = []
        ws_pts = []

        for tag in tags:
            tag_id = int(tag.tag_id)
            if tag_id not in self.tag_world_points:
                continue

            c = tag.center.astype(np.float32)
            img_pts.append([float(c[0]), float(c[1])])

            wx, wy = self.tag_world_points[tag_id]
            ws_pts.append([wx, wy])

            corners = tag.corners.astype(int)
            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(debug_img, pt1, pt2, (0, 255, 0), 2)

            center = tuple(tag.center.astype(int))
            cv2.circle(debug_img, center, 5, (0, 0, 255), -1)
            cv2.putText(
                debug_img,
                f"ID:{tag_id}",
                (center[0] + 8, center[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 0, 0),
                2,
            )

        cv2.putText(
            debug_img,
            f"workspace tags used: {len(img_pts)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if len(img_pts) >= 4 else (0, 0, 255),
            2,
        )

        if len(img_pts) < 4:
            return None, None

        img_pts = np.array(img_pts, dtype=np.float32)
        ws_pts = np.array(ws_pts, dtype=np.float32)

        H, mask = cv2.findHomography(img_pts, ws_pts, method=0)
        if H is None:
            return None, None

        return H, (img_pts, ws_pts)

    # =====================================================
    # Geometry helpers
    # =====================================================
    def transform_point(self, H, pt_xy):
        try:
            src = np.array([[[float(pt_xy[0]), float(pt_xy[1])]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, H)
            return float(dst[0, 0, 0]), float(dst[0, 0, 1])
        except Exception:
            return None

    def estimate_bbox_size_mm(self, H, x1, y1, x2, y2):
        p1 = self.transform_point(H, (x1, y1))
        p2 = self.transform_point(H, (x2, y1))
        p3 = self.transform_point(H, (x2, y2))
        p4 = self.transform_point(H, (x1, y2))

        if None in (p1, p2, p3, p4):
            return None, None

        def dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        top = dist(p1, p2)
        right = dist(p2, p3)
        bottom = dist(p3, p4)
        left = dist(p4, p1)

        w = 0.5 * (top + bottom)
        h = 0.5 * (left + right)
        return w, h

    # =====================================================
    # Depth
    # =====================================================
    def get_depth_median_mm(self, depth_img, cx, cy, k=2) -> Optional[float]:
        h, w = depth_img.shape[:2]
        x1 = clamp(cx - k, 0, w - 1)
        x2 = clamp(cx + k + 1, 0, w)
        y1 = clamp(cy - k, 0, h - 1)
        y2 = clamp(cy + k + 1, 0, h)

        patch = depth_img[y1:y2, x1:x2].astype(np.float32)
        vals = patch[np.isfinite(patch)]
        vals = vals[vals > 0]

        if len(vals) == 0:
            return None

        raw = float(np.median(vals))
        z_m = raw * self.depth_value_scale_to_m
        z_mm = z_m * 1000.0
        return z_mm

    # =====================================================
    # Yaw estimation
    # =====================================================
    def estimate_yaw_deg_from_roi(self, roi_bgr, roi_origin_xy, H_img_to_ws) -> Optional[float]:
        if roi_bgr is None or roi_bgr.size == 0:
            return None

        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 배경에 따라 필요시 바꾸면 됨
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # foreground/background 뒤집힘 보정
        white_ratio = float(np.count_nonzero(th)) / float(th.size)
        if white_ratio > 0.7:
            th = 255 - th

        kernel = np.ones((3, 3), np.uint8)
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 50:
            return None

        pts = cnt.reshape(-1, 2).astype(np.float32)

        # PCA
        mean = np.mean(pts, axis=0)
        pts_centered = pts - mean
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eig(cov)

        major_idx = int(np.argmax(eigvals))
        v = eigvecs[:, major_idx].astype(np.float32)
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            return None
        v = v / norm

        # ROI 좌표 -> 이미지 전체 좌표
        ox, oy = roi_origin_xy
        p0_img = (float(mean[0] + ox), float(mean[1] + oy))
        p1_img = (float(mean[0] + ox + 40.0 * v[0]), float(mean[1] + oy + 40.0 * v[1]))

        p0_ws = self.transform_point(H_img_to_ws, p0_img)
        p1_ws = self.transform_point(H_img_to_ws, p1_img)
        if p0_ws is None or p1_ws is None:
            return None

        dx = p1_ws[0] - p0_ws[0]
        dy = p1_ws[1] - p0_ws[1]
        yaw_deg = math.degrees(math.atan2(dy, dx))

        # 180도 중복축 정리
        while yaw_deg > 90.0:
            yaw_deg -= 180.0
        while yaw_deg <= -90.0:
            yaw_deg += 180.0

        return yaw_deg

    # =====================================================
    # QR
    # =====================================================
    def decode_qr_from_roi(self, roi_bgr) -> str:
        if roi_bgr is None or roi_bgr.size == 0:
            return ""

        try:
            qrs = qr_decode(roi_bgr)
            if len(qrs) == 0:
                return ""

            # 첫 번째 값 사용
            text = qrs[0].data.decode("utf-8", errors="ignore").strip()
            return text
        except Exception:
            return ""

    # =====================================================
    # Debug
    # =====================================================
    def publish_debug(self, img_bgr, stamp):
        try:
            msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            msg.header.stamp = stamp
            msg.header.frame_id = self.camera_frame
            self.debug_pub.publish(msg)
        except Exception as e:
            self.get_logger().error(f"Debug publish failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = VisionScanItemsActionServer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()