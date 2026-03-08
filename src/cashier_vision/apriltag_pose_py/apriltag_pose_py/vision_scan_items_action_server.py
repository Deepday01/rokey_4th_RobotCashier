#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import traceback
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

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

from apriltag_pose_interfaces.action import ScanItems
from apriltag_pose_interfaces.msg import Item


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class VisionScanItemsActionServer(Node):
    def __init__(self):
        super().__init__("vision_scan_items_action_server")

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------
        self.declare_parameter("color_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/camera/color/camera_info")
        self.declare_parameter("use_compressed", False)

        self.declare_parameter("debug_image_topic", "/vision/debug_image")
        self.declare_parameter("debug_timer_hz", 5.0)
        self.declare_parameter("show_cv", True)

        default_model_path = os.path.expanduser("~/Downloads/best.pt")
        self.declare_parameter("model_path", default_model_path)

        self.declare_parameter("conf_thres", 0.45)
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

        self.declare_parameter("depth_value_scale_to_m", 0.001)
        self.declare_parameter("depth_kernel", 2)

        # yaw stabilization
        self.declare_parameter("yaw_history_size", 10)
        self.declare_parameter("yaw_bin_deg", 5.0)
        self.declare_parameter("yaw_min_aspect_ratio", 1.15)
        self.declare_parameter("yaw_hold_frames", 5)

        # ROI / OBB
        self.declare_parameter("roi_pad_px", 6)

        # yellow board hsv range
        self.declare_parameter("yellow_h_min", 18)
        self.declare_parameter("yellow_s_min", 40)
        self.declare_parameter("yellow_v_min", 40)
        self.declare_parameter("yellow_h_max", 42)
        self.declare_parameter("yellow_s_max", 255)
        self.declare_parameter("yellow_v_max", 255)

        # ID -> workspace(mm)
        self.declare_parameter(
            "tag_world_points_json",
            '{"0":[0,0], "2":[360,0], "6":[0,370], "8":[360,370]}'
        )

        self.declare_parameter("class_name_map_json", "{}")

        self.color_topic = self.get_parameter("color_topic").value
        self.depth_topic = self.get_parameter("depth_topic").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value
        self.use_compressed = bool(self.get_parameter("use_compressed").value)

        self.debug_image_topic = self.get_parameter("debug_image_topic").value
        self.debug_timer_hz = float(self.get_parameter("debug_timer_hz").value)
        self.show_cv = bool(self.get_parameter("show_cv").value)

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

        self.yaw_history_size = int(self.get_parameter("yaw_history_size").value)
        self.yaw_bin_deg = float(self.get_parameter("yaw_bin_deg").value)
        self.yaw_min_aspect_ratio = float(self.get_parameter("yaw_min_aspect_ratio").value)
        self.yaw_hold_frames = int(self.get_parameter("yaw_hold_frames").value)

        self.roi_pad_px = int(self.get_parameter("roi_pad_px").value)

        self.yellow_h_min = int(self.get_parameter("yellow_h_min").value)
        self.yellow_s_min = int(self.get_parameter("yellow_s_min").value)
        self.yellow_v_min = int(self.get_parameter("yellow_v_min").value)
        self.yellow_h_max = int(self.get_parameter("yellow_h_max").value)
        self.yellow_s_max = int(self.get_parameter("yellow_s_max").value)
        self.yellow_v_max = int(self.get_parameter("yellow_v_max").value)

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

        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise

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
        self.latest_items: List[Tuple[float, Item]] = []

        self._last_missing_log_t = 0.0
        self._last_process_error_t = 0.0
        self.scan_in_progress = False

        self.last_good_yaw_by_track: Dict[str, float] = {}
        self.last_good_yaw_miss_count: Dict[str, int] = defaultdict(int)

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
        # Timer
        # -------------------------------------------------
        timer_period = 1.0 / max(self.debug_timer_hz, 1.0)
        self.create_timer(timer_period, self.timer_debug_callback)

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

        self.get_logger().info(f"Color topic: {self.color_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"CameraInfo topic: {self.camera_info_topic}")
        self.get_logger().info(f"Debug topic: {self.debug_image_topic}")
        self.get_logger().info(f"Show CV: {self.show_cv}")
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

    def execute_callback(self, goal_handle):
        result = ScanItems.Result()
        result.success = False
        result.items_scan = []

        self.scan_in_progress = True

        try:
            self.get_logger().info("ScanItems action started.")

            feedback_msg = ScanItems.Feedback()
            start_time = time.time()
            end_time = start_time + self.scan_duration_sec
            interval = 1.0 / max(self.feedback_hz, 1.0)

            best_by_track: Dict[str, Tuple[float, Item]] = {}
            yaw_history: Dict[str, List[float]] = defaultdict(list)

            while time.time() < end_time:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    self.get_logger().warn("ScanItems canceled.")
                    self.scan_in_progress = False
                    return result

                runtime = int(time.time() - start_time)
                feedback_msg.runtime = runtime
                goal_handle.publish_feedback(feedback_msg)

                self.process_latest_frame()

                for conf, item in self.latest_items:
                    track_key = self.make_track_key(item)

                    if track_key not in best_by_track or conf > best_by_track[track_key][0]:
                        best_by_track[track_key] = (conf, item)

                    if -90.0 <= float(item.yaw) <= 90.0:
                        yaw_history[track_key].append(float(item.yaw))
                        if len(yaw_history[track_key]) > self.yaw_history_size:
                            yaw_history[track_key].pop(0)

                time.sleep(interval)

            final_items = []

            for track_key, (best_conf, item) in best_by_track.items():
                hist = yaw_history.get(track_key, [])
                stable_yaw = self.robust_mode_yaw(hist, self.yaw_bin_deg)

                if stable_yaw is not None:
                    item.yaw = float(stable_yaw)

                final_items.append(item)

            final_items.sort(key=lambda m: (m.y, m.x))

            result.success = len(final_items) > 0
            result.items_scan = final_items

            goal_handle.succeed()
            self.get_logger().info(f"ScanItems done. count={len(final_items)}")
            self.scan_in_progress = False
            return result

        except Exception as e:
            self.get_logger().error(f"execute_callback failed: {repr(e)}")
            self.get_logger().error(traceback.format_exc())
            try:
                goal_handle.abort()
            except Exception:
                pass
            self.scan_in_progress = False
            return result

    # =====================================================
    # Timer
    # =====================================================
    def timer_debug_callback(self):
        if self.scan_in_progress:
            return

        try:
            self.process_latest_frame()
        except Exception as e:
            now = time.time()
            if now - self._last_process_error_t > 1.0:
                self.get_logger().error(f"timer_debug_callback failed: {repr(e)}")
                self._last_process_error_t = now

    # =====================================================
    # Subscriptions
    # =====================================================
    def on_camera_info(self, msg: CameraInfo):
        try:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.have_cam_info = True

            if self.camera_frame_override:
                self.camera_frame = self.camera_frame_override
            elif msg.header.frame_id:
                self.camera_frame = msg.header.frame_id
        except Exception as e:
            self.get_logger().error(f"CameraInfo callback failed: {e}")

    def on_depth(self, msg: Image):
        try:
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
        try:
            if self.latest_color is None:
                self._log_missing_once("latest_color is None")
                return

            color = self.latest_color.copy()
            stamp = self.latest_color_stamp
            debug_img = color.copy()

            missing_msgs = []
            if self.latest_depth is None:
                missing_msgs.append("depth not ready")
            if not self.have_cam_info:
                missing_msgs.append("camera_info not ready")

            if missing_msgs:
                cv2.putText(
                    debug_img,
                    " / ".join(missing_msgs),
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2,
                )
                self.latest_items = []
                self.publish_debug(debug_img, stamp)
                return

            depth = self.latest_depth.copy()

            H_img_to_ws, tag_boxes = self.compute_workspace_homography(color, debug_img)
            workspace_ready = H_img_to_ws is not None

            if not workspace_ready:
                cv2.putText(
                    debug_img,
                    "Workspace homography not ready (need >=4 mapped tags)",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            results = self.model.predict(
                source=color,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                imgsz=self.imgsz,
                verbose=False,
            )

            items_with_score: List[Tuple[float, Item]] = []

            if len(results) > 0:
                result = results[0]
                boxes = result.boxes
                names = result.names if hasattr(result, "names") else self.model.names

                if boxes is not None:
                    for box in boxes:
                        try:
                            cls_id = int(box.cls.item())
                            conf = float(box.conf.item())
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                            x1 = clamp(x1, 0, color.shape[1] - 1)
                            y1 = clamp(y1, 0, color.shape[0] - 1)
                            x2 = clamp(x2, 0, color.shape[1] - 1)
                            y2 = clamp(y2, 0, color.shape[0] - 1)
                            if x2 <= x1 or y2 <= y1:
                                continue

                            if self.is_overlapping_tag((x1, y1, x2, y2), tag_boxes, thr=0.15):
                                continue

                            cx = int((x1 + x2) * 0.5)
                            cy = int((y1 + y2) * 0.5)

                            z_mm = self.get_depth_median_mm(depth, cx, cy, self.depth_kernel)
                            if z_mm is None:
                                continue

                            box_w = x2 - x1
                            box_h = y2 - y1
                            pad = int(max(3, min(self.roi_pad_px, 0.08 * max(box_w, box_h))))

                            rx1 = clamp(x1 - pad, 0, color.shape[1] - 1)
                            ry1 = clamp(y1 - pad, 0, color.shape[0] - 1)
                            rx2 = clamp(x2 + pad, 0, color.shape[1] - 1)
                            ry2 = clamp(y2 + pad, 0, color.shape[0] - 1)

                            if rx2 <= rx1 or ry2 <= ry1:
                                continue

                            roi = color[ry1:ry2, rx1:rx2]
                            qr_text = self.decode_qr_from_roi(roi)

                            if isinstance(names, dict):
                                default_name = names.get(cls_id, f"class_{cls_id}")
                            else:
                                default_name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"

                            cls_name = self.class_name_map.get(cls_id, str(default_name))

                            obb_ws = None

                            if workspace_ready:
                                ws_center = self.transform_point(H_img_to_ws, (cx, cy))
                                if ws_center is None:
                                    continue
                                x_mm, y_mm = ws_center
                                w_mm, h_mm = self.estimate_bbox_size_mm(H_img_to_ws, x1, y1, x2, y2)
                            else:
                                x_mm, y_mm = 0.0, 0.0
                                w_mm, h_mm = 0.0, 0.0

                            item_id = qr_text if qr_text else f"{cls_name}_{cx}_{cy}"

                            temp_item = Item()
                            temp_item.item_id = item_id
                            temp_item.name = cls_name
                            temp_item.x = float(x_mm)
                            temp_item.y = float(y_mm)
                            track_key = self.make_track_key(temp_item)

                            yaw_deg = None
                            if workspace_ready:
                                other_boxes = []
                                for other_box in boxes:
                                    ox1, oy1, ox2, oy2 = map(int, other_box.xyxy[0].tolist())
                                    ox1 = clamp(ox1, 0, color.shape[1] - 1)
                                    oy1 = clamp(oy1, 0, color.shape[0] - 1)
                                    ox2 = clamp(ox2, 0, color.shape[1] - 1)
                                    oy2 = clamp(oy2, 0, color.shape[0] - 1)

                                    if ox1 == x1 and oy1 == y1 and ox2 == x2 and oy2 == y2:
                                        continue

                                    other_boxes.append((ox1, oy1, ox2, oy2))

                                yaw_deg, obb_ws = self.estimate_yaw_deg_from_roi(
                                    roi_bgr=roi,
                                    roi_origin_xy=(rx1, ry1),
                                    H_img_to_ws=H_img_to_ws,
                                    current_box=(x1, y1, x2, y2),
                                    other_boxes=other_boxes,
                                )

                            if yaw_deg is None:
                                miss_cnt = self.last_good_yaw_miss_count[track_key]
                                if miss_cnt < self.yaw_hold_frames:
                                    yaw_deg = self.get_fallback_yaw(track_key)
                                else:
                                    yaw_deg = 0.0
                                self.last_good_yaw_miss_count[track_key] += 1
                            else:
                                self.last_good_yaw_by_track[track_key] = float(yaw_deg)
                                self.last_good_yaw_miss_count[track_key] = 0

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

                            items_with_score.append((conf, item))

                            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 200, 0), 2)
                            cv2.circle(debug_img, (cx, cy), 4, (0, 0, 255), -1)
                            cv2.rectangle(debug_img, (rx1, ry1), (rx2, ry2), (180, 180, 0), 1)

                            if workspace_ready and obb_ws is not None:
                                obb_img = []
                                for p in obb_ws:
                                    pi = self.transform_point_inv(H_img_to_ws, (float(p[0]), float(p[1])))
                                    if pi is not None:
                                        obb_img.append((int(pi[0]), int(pi[1])))

                                if len(obb_img) == 4:
                                    for i in range(4):
                                        cv2.line(debug_img, obb_img[i], obb_img[(i + 1) % 4], (0, 0, 255), 2)

                            cv2.putText(
                                debug_img,
                                f"{cls_name} {conf:.2f}",
                                (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 255),
                                2,
                            )

                            if workspace_ready:
                                cv2.putText(
                                    debug_img,
                                    f"x={x_mm:.1f} y={y_mm:.1f} z={z_mm:.1f} yaw={yaw_deg:.1f}",
                                    (x1, min(debug_img.shape[0] - 10, y2 + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.55,
                                    (0, 255, 0),
                                    2,
                                )
                            else:
                                cv2.putText(
                                    debug_img,
                                    f"z={z_mm:.1f} (workspace not ready)",
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

                        except Exception as box_e:
                            self.get_logger().error(f"box processing failed: {repr(box_e)}")
                            continue

            self.latest_items = items_with_score
            self.publish_debug(debug_img, stamp)

        except Exception as e:
            now = time.time()
            if self.latest_color is not None:
                debug_img = self.latest_color.copy()
                cv2.putText(
                    debug_img,
                    f"process error: {type(e).__name__}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                self.publish_debug(debug_img, self.latest_color_stamp)

            if now - self._last_process_error_t > 1.0:
                self.get_logger().error(f"process_latest_frame failed: {repr(e)}")
                self.get_logger().error(traceback.format_exc())
                self._last_process_error_t = now

    def _log_missing_once(self, text: str):
        now = time.time()
        if now - self._last_missing_log_t > 1.0:
            self.get_logger().warn(text)
            self._last_missing_log_t = now

    # =====================================================
    # AprilTag -> workspace homography
    # =====================================================
    def compute_workspace_homography(self, color_bgr, debug_img):
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, estimate_tag_pose=False)

        img_pts = []
        ws_pts = []
        tag_boxes = []

        for tag in tags:
            tag_id = int(tag.tag_id)

            corners = tag.corners.astype(int)
            min_x = int(np.min(corners[:, 0]))
            min_y = int(np.min(corners[:, 1]))
            max_x = int(np.max(corners[:, 0]))
            max_y = int(np.max(corners[:, 1]))
            tag_boxes.append((min_x, min_y, max_x, max_y))

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

            if tag_id not in self.tag_world_points:
                continue

            c = tag.center.astype(np.float32)
            img_pts.append([float(c[0]), float(c[1])])

            wx, wy = self.tag_world_points[tag_id]
            ws_pts.append([wx, wy])

        cv2.putText(
            debug_img,
            f"workspace tags used: {len(img_pts)}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0) if len(img_pts) >= 4 else (0, 0, 255),
            2,
        )

        if len(img_pts) < 4:
            return None, tag_boxes

        img_pts = np.array(img_pts, dtype=np.float32)
        ws_pts = np.array(ws_pts, dtype=np.float32)

        H, _ = cv2.findHomography(img_pts, ws_pts, method=cv2.RANSAC)
        if H is None:
            return None, tag_boxes

        return H, tag_boxes

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

    def transform_point_inv(self, H, pt_xy):
        try:
            H_inv = np.linalg.inv(H)
            src = np.array([[[float(pt_xy[0]), float(pt_xy[1])]]], dtype=np.float32)
            dst = cv2.perspectiveTransform(src, H_inv)
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

    def bbox_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        if union <= 0:
            return 0.0
        return inter / union

    def is_overlapping_tag(self, box, tag_boxes, thr=0.15):
        for tb in tag_boxes:
            if self.bbox_iou(box, tb) > thr:
                return True
        return False

    def normalize_yaw_deg_180(self, yaw_deg: float) -> float:
        while yaw_deg > 90.0:
            yaw_deg -= 180.0
        while yaw_deg <= -90.0:
            yaw_deg += 180.0
        return yaw_deg

    def yaw_to_axis_0_180(self, yaw_deg: float) -> float:
        y = self.normalize_yaw_deg_180(yaw_deg)
        if y < 0.0:
            y += 180.0
        return y

    def robust_mode_yaw(self, yaw_list: List[float], bin_deg: float = 5.0) -> Optional[float]:
        if not yaw_list:
            return None

        bins = defaultdict(list)
        for yaw in yaw_list:
            axis_yaw = self.yaw_to_axis_0_180(float(yaw))
            bin_idx = int(axis_yaw // bin_deg)
            bins[bin_idx].append(axis_yaw)

        if not bins:
            return None

        _, dominant_vals = max(
            bins.items(),
            key=lambda kv: (len(kv[1]), -abs(np.median(kv[1]) - 90.0))
        )

        rep = float(np.median(dominant_vals))
        if rep >= 90.0:
            rep -= 180.0

        return self.normalize_yaw_deg_180(rep)

    def make_track_key(self, item: Item) -> str:
        if item.item_id and not item.item_id.startswith(f"{item.name}_"):
            return f"qr:{item.item_id}"

        gx = int(round(item.x / 30.0))
        gy = int(round(item.y / 30.0))
        return f"{item.name}_{gx}_{gy}"

    def contour_center(self, cnt):
        m = cv2.moments(cnt)
        if abs(m["m00"]) < 1e-6:
            pts = cnt.reshape(-1, 2).astype(np.float32)
            return float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))
        return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])

    def get_fallback_yaw(self, track_key: str, current_hist: Optional[List[float]] = None) -> float:
        if current_hist:
            stable = self.robust_mode_yaw(current_hist, self.yaw_bin_deg)
            if stable is not None:
                return float(stable)

        if track_key in self.last_good_yaw_by_track:
            return float(self.last_good_yaw_by_track[track_key])

        return 0.0

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
    def estimate_yaw_deg_from_roi(
        self,
        roi_bgr,
        roi_origin_xy,
        H_img_to_ws,
        current_box=None,
        other_boxes=None,
    ):
        if roi_bgr is None or roi_bgr.size == 0 or H_img_to_ws is None:
            return None, None

        if other_boxes is None:
            other_boxes = []

        roi_h, roi_w = roi_bgr.shape[:2]
        ox, oy = roi_origin_xy

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min], dtype=np.uint8)
        upper_yellow = np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        fg_mask = 255 - yellow_mask

        # 다른 bbox 제거(ROI 중심 보호 + shrink)
        for bx1, by1, bx2, by2 in other_boxes:
            rx1 = clamp(bx1 - ox, 0, roi_w - 1)
            ry1 = clamp(by1 - oy, 0, roi_h - 1)
            rx2 = clamp(bx2 - ox, 0, roi_w - 1)
            ry2 = clamp(by2 - oy, 0, roi_h - 1)

            if rx2 <= rx1 or ry2 <= ry1:
                continue

            protect_cx = roi_w * 0.5
            protect_cy = roi_h * 0.5
            protect_r = min(roi_w, roi_h) * 0.22

            ob_cx = 0.5 * (rx1 + rx2)
            ob_cy = 0.5 * (ry1 + ry2)

            dist_to_center = math.hypot(ob_cx - protect_cx, ob_cy - protect_cy)
            if dist_to_center < protect_r:
                continue

            shrink = 4
            sx1 = clamp(rx1 + shrink, 0, roi_w - 1)
            sy1 = clamp(ry1 + shrink, 0, roi_h - 1)
            sx2 = clamp(rx2 - shrink, 0, roi_w - 1)
            sy2 = clamp(ry2 - shrink, 0, roi_h - 1)

            if sx2 > sx1 and sy2 > sy1:
                cv2.rectangle(fg_mask, (sx1, sy1), (sx2, sy2), 0, -1)

        kernel = np.ones((3, 3), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        cx_roi = roi_w * 0.5
        cy_roi = roi_h * 0.5

        valid_contours = []
        center_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 120:
                continue

            valid_contours.append(cnt)

            inside = cv2.pointPolygonTest(cnt, (cx_roi, cy_roi), False)
            if inside >= 0:
                center_contours.append(cnt)

        if not valid_contours:
            return None, None

        def contour_score(cnt):
            area = cv2.contourArea(cnt)
            ccx, ccy = self.contour_center(cnt)
            dist = math.hypot(ccx - cx_roi, ccy - cy_roi)
            return area - 4.0 * dist

        if center_contours:
            cnt = max(center_contours, key=contour_score)
        else:
            cnt = max(valid_contours, key=contour_score)

        pts_img = cnt.reshape(-1, 2).astype(np.float32)
        pts_img[:, 0] += float(ox)
        pts_img[:, 1] += float(oy)

        pts_ws = []
        for p in pts_img:
            pw = self.transform_point(H_img_to_ws, (float(p[0]), float(p[1])))
            if pw is not None:
                pts_ws.append([pw[0], pw[1]])

        if len(pts_ws) < 5:
            return None, None

        pts_ws = np.array(pts_ws, dtype=np.float32)

        rect = cv2.minAreaRect(pts_ws)
        (_, _), (rw, rh), _ = rect

        if rw < 1e-6 or rh < 1e-6:
            return None, None

        long_side = max(rw, rh)
        short_side = min(rw, rh)
        aspect_ratio = long_side / max(short_side, 1e-6)

        if aspect_ratio < self.yaw_min_aspect_ratio:
            return None, None

        box_ws = cv2.boxPoints(rect).astype(np.float32)

        best_len = -1.0
        best_vec = None
        for i in range(4):
            p0 = box_ws[i]
            p1 = box_ws[(i + 1) % 4]
            vec = p1 - p0
            length = float(np.linalg.norm(vec))
            if length > best_len:
                best_len = length
                best_vec = vec

        if best_vec is None or best_len < 1e-6:
            return None, None

        dx = float(best_vec[0])
        dy = float(best_vec[1])

        yaw_deg = math.degrees(math.atan2(dy, dx))
        yaw_deg = self.normalize_yaw_deg_180(yaw_deg)

        return yaw_deg, box_ws

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
            return qrs[0].data.decode("utf-8", errors="ignore").strip()
        except Exception:
            return ""

    # =====================================================
    # Debug
    # =====================================================
    def publish_debug(self, img_bgr, stamp):
        try:
            if img_bgr is None:
                return

            if self.show_cv:
                try:
                    cv2.imshow("vision_debug", img_bgr)
                    cv2.waitKey(1)
                except Exception as e:
                    self.get_logger().warn(f"cv show failed: {repr(e)}")

            msg = self.bridge.cv2_to_imgmsg(img_bgr, encoding="bgr8")
            if stamp is not None:
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
    finally:
        if node.show_cv:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()