#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import traceback
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ultralytics import YOLO
from pupil_apriltags import Detector
from pyzbar.pyzbar import decode as qr_decode

from cashier_interfaces.action import ScanItems
from cashier_interfaces.msg import Item


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


class VisionScanItemsActionServer(Node):
    def __init__(self):
        super().__init__("vision_scan_items_action_server")

        default_model_path = os.path.expanduser("~/rokey_4th_RobotCashier/src/cashier_vision/best_fin.pt")
        default_specs_path = os.path.expanduser(
            "~/rokey_4th_RobotCashier/src/cashier_vision/apriltag_pose_py/config/item_specs.json"
        )

        params = {
            # camera
            "color_topic": "/camera/camera/color/image_raw",
            "depth_topic": "/camera/camera/aligned_depth_to_color/image_raw",
            "use_compressed": False,
            "debug_image_topic": "/vision/debug_image",
            "debug_timer_hz": 5.0,
            "show_cv": True,
            "cv_window_width": 1280,
            "cv_window_height": 720,

            # yolo
            "model_path": default_model_path,
            "conf_thres": 0.15,
            "iou_thres": 0.45,
            "max_det": 20,
            "imgsz": 960,

            # action
            "scan_duration_sec": 4.0,
            "feedback_hz": 5.0,

            # apriltag / homography
            "tag_family": "tag36h11",
            "tag_size_m": 0.05,
            "camera_frame_override": "",
            "decimate": 1.0,
            "sigma": 0.0,
            "nthreads": 2,
            "refine_edges": True,

            # depth
            "depth_value_scale_to_m": 0.001,
            "depth_kernel": 2,

            # yaw
            "yaw_history_size": 10,
            "yaw_bin_deg": 5.0,
            "yaw_min_aspect_ratio": 1.15,
            "yaw_hold_frames": 8,

            # roi / track
            "roi_pad_px": 6,
            "track_grid_mm": 10.0,

            # object mask
            "yellow_h_min": 18,
            "yellow_s_min": 40,
            "yellow_v_min": 40,
            "yellow_h_max": 42,
            "yellow_s_max": 255,
            "yellow_v_max": 255,

            # display: raw 느낌 유지
            "display_enhance": False,
            "display_contrast_alpha": 1.0,
            "display_brightness_beta": 0,
            "display_sharpen": False,

            # draw
            "label_font_scale": 0.33,
            "label_thickness": 1,
            "box_thickness": 2,
            "obb_thickness": 2,
            "text_bg": True,

            # box filter
            "dup_iou_thres": 0.30,
            "small_box_area_ratio": 0.65,
            "min_box_area_px": 2500,

            # workspace
            "workspace_size_mm": 410.0,

            # tag layout
            "tag_world_points_json": '{"1":[0,0], "6":[410,0], "2":[0,410], "8":[410,410]}',

            # workspace(mm) -> robot(mm)
            "robot_corner_points_json": '''
            {
                "lt": [493.0, 325.0],
                "rt": [493.0, -75.0],
                "lb": [93.0, 325.0],
                "rb": [93.0, -75.0]
            }
            ''',

            "item_specs_json_path": default_specs_path,
            "class_name_map_json": "{}",

            # qr: 화면 표시만
            "enable_fullframe_qr": True,

            # debug draw flags
            "draw_tag_boxes": False,
            "draw_workspace_text": False,
            "draw_center": False,
            "draw_label": True,
            "draw_xy": True,
            "draw_z": True,
            "draw_yaw": True,
            "draw_spec": False,
            "draw_roi": False,
            "draw_obb": True,
        }

        for k, v in params.items():
            self.declare_parameter(k, v)

        for k in params.keys():
            setattr(self, k, self.get_parameter(k).value)

        self.use_compressed = bool(self.use_compressed)
        self.debug_timer_hz = float(self.debug_timer_hz)
        self.show_cv = bool(self.show_cv)
        self.cv_window_width = int(self.cv_window_width)
        self.cv_window_height = int(self.cv_window_height)

        self.conf_thres = float(self.conf_thres)
        self.iou_thres = float(self.iou_thres)
        self.max_det = int(self.max_det)
        self.imgsz = int(self.imgsz)

        self.scan_duration_sec = float(self.scan_duration_sec)
        self.feedback_hz = float(self.feedback_hz)

        self.tag_size_m = float(self.tag_size_m)
        self.depth_value_scale_to_m = float(self.depth_value_scale_to_m)
        self.depth_kernel = int(self.depth_kernel)

        self.yaw_history_size = int(self.yaw_history_size)
        self.yaw_bin_deg = float(self.yaw_bin_deg)
        self.yaw_min_aspect_ratio = float(self.yaw_min_aspect_ratio)
        self.yaw_hold_frames = int(self.yaw_hold_frames)

        self.roi_pad_px = int(self.roi_pad_px)
        self.track_grid_mm = float(self.track_grid_mm)

        self.yellow_h_min = int(self.yellow_h_min)
        self.yellow_s_min = int(self.yellow_s_min)
        self.yellow_v_min = int(self.yellow_v_min)
        self.yellow_h_max = int(self.yellow_h_max)
        self.yellow_s_max = int(self.yellow_s_max)
        self.yellow_v_max = int(self.yellow_v_max)

        self.display_enhance = bool(self.display_enhance)
        self.display_contrast_alpha = float(self.display_contrast_alpha)
        self.display_brightness_beta = int(self.display_brightness_beta)
        self.display_sharpen = bool(self.display_sharpen)

        self.label_font_scale = float(self.label_font_scale)
        self.label_thickness = int(self.label_thickness)
        self.box_thickness = int(self.box_thickness)
        self.obb_thickness = int(self.obb_thickness)
        self.text_bg = bool(self.text_bg)

        self.dup_iou_thres = float(self.dup_iou_thres)
        self.small_box_area_ratio = float(self.small_box_area_ratio)
        self.min_box_area_px = int(self.min_box_area_px)

        self.workspace_size_mm = float(self.workspace_size_mm)
        self.enable_fullframe_qr = bool(self.enable_fullframe_qr)

        self.draw_tag_boxes = bool(self.draw_tag_boxes)
        self.draw_workspace_text = bool(self.draw_workspace_text)
        self.draw_center = bool(self.draw_center)
        self.draw_label = bool(self.draw_label)
        self.draw_xy = bool(self.draw_xy)
        self.draw_z = bool(self.draw_z)
        self.draw_yaw = bool(self.draw_yaw)
        self.draw_spec = bool(self.draw_spec)
        self.draw_roi = bool(self.draw_roi)
        self.draw_obb = bool(self.draw_obb)

        self.camera_frame = self.camera_frame_override if self.camera_frame_override else "camera_frame"

        self.tag_world_points: Dict[int, Tuple[float, float]] = {
            int(k): (float(v[0]), float(v[1]))
            for k, v in json.loads(self.tag_world_points_json).items()
        }

        self.robot_corner_points = {
            str(k): (float(v[0]), float(v[1]))
            for k, v in json.loads(self.robot_corner_points_json).items()
        }

        self.class_name_map = {
            int(k): str(v)
            for k, v in json.loads(self.class_name_map_json).items()
        }

        self.item_specs = self.load_item_specs(self.item_specs_json_path)

        self.bridge = CvBridge()

        try:
            self.model = YOLO(self.model_path)
            self.get_logger().info(f"YOLO model loaded: {self.model_path}")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLO model: {e}")
            raise

        self.item_order = [
            "halls",
            "insect",
            "caramel",
            "candy",
            "cream",
            "eclipse_red",
            "eclipse_gre",
        ]

        self.qr_map = {
            "https://m.site.naver.com/20lZW": "halls",
            "https://m.site.naver.com/20m1A": "insect",
            "https://m.site.naver.com/20suF": "caramel",
            "https://m.site.naver.com/20svK": "candy",
            "https://m.site.naver.com/20swh": "caramel",
            "https://m.site.naver.com/20swH": "cream",
            "https://m.site.naver.com/20syN": "eclipse_red",
            "https://m.site.naver.com/20y2j": "eclipse_gre",
        }

        self.latest_fullframe_qr_raws: List[str] = []
        self.latest_fullframe_qr_names: Set[str] = set()

        self.detector = Detector(
            families=self.tag_family,
            nthreads=int(self.nthreads),
            quad_decimate=float(self.decimate),
            quad_sigma=float(self.sigma),
            refine_edges=bool(self.refine_edges),
        )

        self.H_ws_to_robot = self.compute_workspace_to_robot_homography()

        self.latest_color = None
        self.latest_color_stamp = None
        self.latest_depth = None
        self.latest_depth_stamp = None
        self.latest_items: List[Tuple[float, Item, str]] = []

        self._last_missing_log_t = 0.0
        self._last_process_error_t = 0.0
        self.scan_in_progress = False

        self.last_good_yaw_by_track: Dict[str, float] = {}
        self.last_good_yaw_miss_count: Dict[str, int] = defaultdict(int)

        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 10)
        self.create_subscription(Image, self.depth_topic, self.on_depth, 10)

        if self.use_compressed:
            self.create_subscription(CompressedImage, self.color_topic, self.on_color_compressed, 10)
            self.get_logger().info(f"Subscribing CompressedImage: {self.color_topic}")
        else:
            self.create_subscription(Image, self.color_topic, self.on_color, 10)
            self.get_logger().info(f"Subscribing Image: {self.color_topic}")

        self.create_timer(1.0 / max(self.debug_timer_hz, 1.0), self.timer_debug_callback)

        self._action_server = ActionServer(
            self,
            ScanItems,
            "/vision/scan_items",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
        )

        if self.show_cv:
            try:
                cv2.namedWindow("vision_debug", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("vision_debug", self.cv_window_width, self.cv_window_height)
            except Exception as e:
                self.get_logger().warn(f"cv window init failed: {repr(e)}")

        self.get_logger().info(f"Color topic: {self.color_topic}")
        self.get_logger().info(f"Depth topic: {self.depth_topic}")
        self.get_logger().info(f"Debug topic: {self.debug_image_topic}")
        self.get_logger().info(f"Show CV: {self.show_cv}")
        self.get_logger().info(f"conf_thres: {self.conf_thres}")
        self.get_logger().info(f"AprilTag world map: {self.tag_world_points}")
        self.get_logger().info(f"Robot corner points: {self.robot_corner_points}")
        self.get_logger().info(f"Workspace->Robot H:\n{self.H_ws_to_robot}")
        self.get_logger().info(f"Item specs path: {self.item_specs_json_path}")
        self.get_logger().info(f"QR full-frame enabled: {self.enable_fullframe_qr}")
        self.get_logger().info(f"QR map count: {len(self.qr_map)}")

        test_center = self.workspace_to_robot(
            self.workspace_size_mm / 2.0,
            self.workspace_size_mm / 2.0
        )
        if test_center is not None:
            self.get_logger().info(f"Predicted robot center from corners: {test_center}")

        self.get_logger().info("VisionScanItemsActionServer ready.")

    def to_int_field(self, value, default=0):
        try:
            return int(round(float(value)))
        except Exception:
            return int(default)

    def enhance_for_display(self, img):
        if img is None:
            return img

        out = img.copy()

        if self.display_enhance:
            out = cv2.convertScaleAbs(
                out,
                alpha=self.display_contrast_alpha,
                beta=self.display_brightness_beta
            )

        if self.display_sharpen:
            blur = cv2.GaussianBlur(out, (0, 0), 1.0)
            out = cv2.addWeighted(out, 1.22, blur, -0.22, 0)

        return out

    def draw_text_with_bg(self, img, text, org, color=(0, 255, 0)):
        if not text:
            return

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self.label_font_scale
        thickness = self.label_thickness

        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        x, y = org

        x = int(clamp(x, 0, max(0, img.shape[1] - tw - 4)))
        y = int(clamp(y, th + 4, img.shape[0] - 4))

        if self.text_bg:
            cv2.rectangle(
                img,
                (x - 2, y - th - 2),
                (x + tw + 2, y + baseline + 2),
                (0, 0, 0),
                -1
            )

        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def _log_missing_once(self, text: str):
        now = time.time()
        if now - self._last_missing_log_t > 1.0:
            self.get_logger().warn(text)
            self._last_missing_log_t = now

    def load_item_specs(self, json_path: str) -> Dict[str, dict]:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, dict):
                self.get_logger().warn("item_specs.json root is not dict. Using empty specs.")
                return {}

            out = {}
            for k, v in data.items():
                if isinstance(k, str) and isinstance(v, dict):
                    out[k.strip()] = {
                        "width": int(round(float(v.get("width", 0)))),
                        "depth": int(round(float(v.get("depth", 0)))),
                        "height": int(round(float(v.get("height", 0)))),
                        "durability": int(v.get("durability", 0)),
                    }

            self.get_logger().info(f"Loaded item specs: {len(out)} entries from {json_path}")
            return out

        except FileNotFoundError:
            self.get_logger().warn(f"item_specs.json not found: {json_path}")
            return {}
        except Exception as e:
            self.get_logger().error(f"Failed to load item_specs.json: {e}")
            return {}

    def get_item_spec(self, item_id: str, name: str) -> dict:
        for key in [item_id, name]:
            if isinstance(key, str):
                key = key.strip()
                if key in self.item_specs:
                    return self.item_specs[key]

        if isinstance(name, str):
            key = name.strip()
            if "_" in key:
                base = key.split("_")[0]
                if base in self.item_specs:
                    return self.item_specs[base]

            cands = sorted(k for k in self.item_specs if k == key or k.startswith(f"{key}_"))
            if cands:
                return self.item_specs[cands[0]]

        return {}

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
                    return result

                feedback_msg.runtime = int(time.time() - start_time)
                goal_handle.publish_feedback(feedback_msg)

                self.process_latest_frame()

                for conf, item, track_key in self.latest_items:
                    if track_key not in best_by_track or conf > best_by_track[track_key][0]:
                        best_by_track[track_key] = (conf, item)

                    if -90.0 <= float(item.yaw) <= 90.0:
                        yaw_history[track_key].append(float(item.yaw))
                        if len(yaw_history[track_key]) > self.yaw_history_size:
                            yaw_history[track_key].pop(0)

                time.sleep(interval)

            final_items = []
            for track_key, (_, item) in best_by_track.items():
                stable_yaw = self.robust_mode_yaw(yaw_history.get(track_key, []), self.yaw_bin_deg)
                if stable_yaw is not None:
                    item.yaw = float(stable_yaw)
                final_items.append(item)

            final_items.sort(key=lambda m: (m.y, m.x))

            name_counts = defaultdict(int)
            for item in final_items:
                name_counts[item.name] += 1

            name_index = defaultdict(int)
            for item in final_items:
                if name_counts[item.name] == 1:
                    item.item_id = item.name
                else:
                    name_index[item.name] += 1
                    item.item_id = f"{item.name}_{name_index[item.name]}"

                spec = self.get_item_spec(item.item_id, item.name)
                item.width = int(round(spec.get("width", 0)))
                item.depth = int(round(spec.get("depth", 0)))
                item.height = int(round(spec.get("height", 0)))
                item.durability = int(spec.get("durability", 0))

            result.success = len(final_items) > 0
            result.items_scan = final_items
            goal_handle.succeed()
            self.get_logger().info(f"ScanItems done. count={len(final_items)}")
            return result

        except Exception as e:
            self.get_logger().error(f"execute_callback failed: {repr(e)}")
            self.get_logger().error(traceback.format_exc())
            try:
                goal_handle.abort()
            except Exception:
                pass
            return result
        finally:
            self.scan_in_progress = False

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

    def on_depth(self, msg: Image):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.latest_depth_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Depth conversion failed: {e}")

    def on_color(self, msg: Image):
        try:
            self.latest_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.latest_color_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Color conversion failed: {e}")

    def on_color_compressed(self, msg: CompressedImage):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            self.latest_color = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            self.latest_color_stamp = msg.header.stamp
        except Exception as e:
            self.get_logger().error(f"Compressed color conversion failed: {e}")

    # =====================================================
    # QR helpers (full-frame only, display only)
    # =====================================================
    def normalize_qr_text(self, raw: str) -> str:
        if raw is None:
            return ""
        return raw.strip()

    def map_qr_to_item_name(self, raw: str) -> str:
        raw = self.normalize_qr_text(raw)
        if not raw:
            return ""

        raw = raw.rstrip("/")
        normalized_map = {k.rstrip("/"): v for k, v in self.qr_map.items()}
        return normalized_map.get(raw, "")

    def decode_qr_candidates(self, img) -> List[str]:
        decoded = set()

        if img is None or img.size == 0:
            return []

        candidates = []

        # 1) 원본
        candidates.append(img)

        # 2) grayscale
        gray = None
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            candidates.append(gray)
        except Exception:
            pass

        # 3) 확대본
        try:
            big = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            candidates.append(big)
        except Exception:
            pass

        if gray is not None:
            try:
                gray_big = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                candidates.append(gray_big)
            except Exception:
                pass

            try:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                gray_clahe = clahe.apply(gray)
                candidates.append(gray_clahe)

                gray_clahe_big = cv2.resize(
                    gray_clahe, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
                )
                candidates.append(gray_clahe_big)
            except Exception:
                pass

            try:
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                candidates.append(otsu)

                otsu_big = cv2.resize(otsu, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                candidates.append(otsu_big)
            except Exception:
                pass

            try:
                adaptive = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    11,
                    2,
                )
                candidates.append(adaptive)

                adaptive_big = cv2.resize(
                    adaptive, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
                )
                candidates.append(adaptive_big)
            except Exception:
                pass

        # 너무 많은 중복 후보 방지
        unique_candidates = []
        seen = set()

        for c in candidates:
            if c is None or c.size == 0:
                continue

            try:
                shape_key = (c.shape[0], c.shape[1], 1 if len(c.shape) == 2 else c.shape[2])
            except Exception:
                continue

            if shape_key not in seen:
                unique_candidates.append(c)
                seen.add(shape_key)

        # pyzbar만 사용
        for c in unique_candidates:
            try:
                results = qr_decode(c)
            except Exception:
                results = []

            for r in results:
                try:
                    raw = r.data.decode("utf-8", errors="ignore").strip()
                    if raw:
                        decoded.add(raw)
                except Exception:
                    pass

        return list(decoded)

    def decode_qr_multi_stage_fullframe(self, frame) -> Tuple[List[str], Set[str]]:
        raw_list = self.decode_qr_candidates(frame)
        name_set = set()

        for raw in raw_list:
            item_name = self.map_qr_to_item_name(raw)
            if item_name:
                name_set.add(item_name)

        return raw_list, name_set

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

            if self.latest_depth is None:
                self.draw_text_with_bg(debug_img, "depth not ready", (20, 35), (0, 0, 255))
                self.latest_items = []
                self.publish_debug(debug_img, stamp)
                return

            depth = self.latest_depth.copy()

            # 전체 프레임 QR: 화면 표시용만 사용
            if self.enable_fullframe_qr:
                fullframe_qr_raws, fullframe_qr_names = self.decode_qr_multi_stage_fullframe(color)
            else:
                fullframe_qr_raws, fullframe_qr_names = [], set()

            self.latest_fullframe_qr_raws = fullframe_qr_raws
            self.latest_fullframe_qr_names = fullframe_qr_names

            H_img_to_ws, tag_boxes = self.compute_workspace_homography(color, debug_img)
            workspace_ready = H_img_to_ws is not None and self.H_ws_to_robot is not None

            if not workspace_ready:
                self.draw_text_with_bg(debug_img, "Workspace/Robot homography not ready", (20, 35), (0, 0, 255))

            results = self.model.predict(
                source=color,
                conf=self.conf_thres,
                iou=self.iou_thres,
                max_det=self.max_det,
                imgsz=self.imgsz,
                verbose=False,
            )

            items_with_score: List[Tuple[float, Item, str]] = []

            if not results:
                self.latest_items = []
                self.publish_debug(debug_img, stamp)
                return

            result = results[0]
            boxes = result.boxes
            names = result.names if hasattr(result, "names") else self.model.names

            if boxes is None:
                self.latest_items = []
                self.publish_debug(debug_img, stamp)
                return

            all_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1 = clamp(x1, 0, color.shape[1] - 1)
                y1 = clamp(y1, 0, color.shape[0] - 1)
                x2 = clamp(x2, 0, color.shape[1] - 1)
                y2 = clamp(y2, 0, color.shape[0] - 1)
                if x2 > x1 and y2 > y1:
                    all_boxes.append((box, x1, y1, x2, y2))

            all_boxes = self.filter_duplicate_and_small_boxes(all_boxes)

            for box, x1, y1, x2, y2 in all_boxes:
                try:
                    if self.is_overlapping_tag((x1, y1, x2, y2), tag_boxes, thr=0.15):
                        continue

                    cls_id = int(box.cls.item())
                    conf = float(box.conf.item())
                    cx, cy = int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)

                    z_mm = self.get_depth_median_mm(depth, cx, cy, self.depth_kernel)
                    if z_mm is None:
                        continue

                    box_w, box_h = x2 - x1, y2 - y1
                    pad = int(max(3, min(self.roi_pad_px, 0.08 * max(box_w, box_h))))
                    rx1 = clamp(x1 - pad, 0, color.shape[1] - 1)
                    ry1 = clamp(y1 - pad, 0, color.shape[0] - 1)
                    rx2 = clamp(x2 + pad, 0, color.shape[1] - 1)
                    ry2 = clamp(y2 + pad, 0, color.shape[0] - 1)
                    if rx2 <= rx1 or ry2 <= ry1:
                        continue

                    roi = color[ry1:ry2, rx1:rx2]

                    if isinstance(names, dict):
                        default_name = names.get(cls_id, f"class_{cls_id}")
                    else:
                        default_name = names[cls_id] if cls_id < len(names) else f"class_{cls_id}"

                    cls_name = self.class_name_map.get(cls_id, str(default_name)).strip()

                    if workspace_ready:
                        ws_center = self.transform_point(H_img_to_ws, (cx, cy))
                        if ws_center is None:
                            continue

                        x_ws, y_ws = ws_center
                        robot_xy = self.workspace_to_robot(x_ws, y_ws)
                        if robot_xy is None:
                            continue

                        x_mm, y_mm = robot_xy
                    else:
                        x_ws, y_ws = 0.0, 0.0
                        x_mm, y_mm = 0.0, 0.0

                    item_id = cls_name
                    track_key = self.make_track_key("", cls_name, x_ws, y_ws)
                    spec = self.get_item_spec(item_id, cls_name)

                    yaw_deg, obb_ws = None, None

                    other_boxes = [
                        (ox1, oy1, ox2, oy2)
                        for _, ox1, oy1, ox2, oy2 in all_boxes
                        if not (ox1 == x1 and oy1 == y1 and ox2 == x2 and oy2 == y2)
                    ]

                    if workspace_ready:
                        yaw_deg, obb_ws, _ = self.estimate_yaw_deg_from_roi_ws(
                            roi_bgr=roi,
                            roi_origin_xy=(rx1, ry1),
                            H_img_to_ws=H_img_to_ws,
                            other_boxes=other_boxes,
                            return_masked_roi=True,
                        )
                    else:
                        yaw_deg, _ = self.estimate_yaw_deg_from_roi_img(
                            roi_bgr=roi,
                            other_boxes=other_boxes,
                            return_masked_roi=True,
                        )
                        obb_ws = None

                    if yaw_deg is None:
                        miss_cnt = self.last_good_yaw_miss_count[track_key]
                        yaw_deg = self.get_fallback_yaw(track_key) if miss_cnt < self.yaw_hold_frames else 0.0
                        self.last_good_yaw_miss_count[track_key] += 1
                    else:
                        self.last_good_yaw_by_track[track_key] = float(yaw_deg)
                        self.last_good_yaw_miss_count[track_key] = 0

                    item = Item()
                    item.item_id = item_id
                    item.name = cls_name
                    item.width = int(round(spec.get("width", 0)))
                    item.depth = int(round(spec.get("depth", 0)))
                    item.height = int(round(spec.get("height", 0)))
                    item.durability = int(spec.get("durability", 0))
                    item.x = float(round(x_mm, 1))
                    item.y = float(round(y_mm, 1))
                    item.z = float(z_mm)
                    item.roll = 0.0
                    item.pitch = 0.0
                    item.yaw = float(yaw_deg)

                    items_with_score.append((conf, item, track_key))

                    self.draw_item_debug(
                        debug_img=debug_img,
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        rx1=rx1, ry1=ry1, rx2=rx2, ry2=ry2,
                        cx=cx, cy=cy,
                        cls_name=cls_name,
                        robot_x_mm=x_mm,
                        robot_y_mm=y_mm,
                        z_mm=z_mm,
                        yaw_deg=yaw_deg,
                        spec=spec,
                        workspace_ready=workspace_ready,
                        H_img_to_ws=H_img_to_ws,
                        obb_ws=obb_ws,
                    )

                except Exception as box_e:
                    self.get_logger().error(f"box processing failed: {repr(box_e)}")

            self.latest_items = items_with_score

            if self.enable_fullframe_qr:
                ordered_qr_names = [n for n in self.item_order if n in fullframe_qr_names]
                if len(ordered_qr_names) > 0:
                    qr_top = f"QR_NOW:{ordered_qr_names[0]}"
                    qr_color = (0, 255, 255)
                elif len(fullframe_qr_raws) > 0:
                    qr_top = "QR_NOW:unknown"
                    qr_color = (0, 165, 255)
                else:
                    qr_top = "QR_NOW:none"
                    qr_color = (150, 150, 150)

                self.draw_text_with_bg(debug_img, qr_top, (20, 55), qr_color)

            self.publish_debug(debug_img, stamp)

        except Exception as e:
            now = time.time()
            if self.latest_color is not None:
                debug_img = self.latest_color.copy()
                self.draw_text_with_bg(debug_img, f"process error: {type(e).__name__}", (20, 35), (0, 0, 255))
                self.publish_debug(debug_img, self.latest_color_stamp)

            if now - self._last_process_error_t > 1.0:
                self.get_logger().error(f"process_latest_frame failed: {repr(e)}")
                self.get_logger().error(traceback.format_exc())
                self._last_process_error_t = now

    def draw_item_debug(
        self,
        debug_img,
        x1, y1, x2, y2,
        rx1, ry1, rx2, ry2,
        cx, cy,
        cls_name,
        robot_x_mm, robot_y_mm, z_mm, yaw_deg,
        spec,
        workspace_ready,
        H_img_to_ws,
        obb_ws,
    ):
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), (255, 200, 0), self.box_thickness)

        if self.draw_center:
            cv2.circle(debug_img, (cx, cy), 3, (0, 0, 255), -1)

        if self.draw_roi:
            cv2.rectangle(debug_img, (rx1, ry1), (rx2, ry2), (180, 180, 0), 1)

        if self.draw_obb and workspace_ready and obb_ws is not None and H_img_to_ws is not None:
            obb_img = []
            for p in obb_ws:
                pi = self.transform_point_inv(H_img_to_ws, (float(p[0]), float(p[1])))
                if pi is not None:
                    obb_img.append((int(pi[0]), int(pi[1])))
            if len(obb_img) == 4:
                for i in range(4):
                    cv2.line(debug_img, obb_img[i], obb_img[(i + 1) % 4], (0, 0, 255), self.obb_thickness)

        pose_parts = []
        if workspace_ready and self.draw_xy:
            pose_parts.append(f"x={robot_x_mm:.1f}")
            pose_parts.append(f"y={robot_y_mm:.1f}")
        if self.draw_z:
            pose_parts.append(f"z={z_mm:.1f}")
        if self.draw_yaw:
            pose_parts.append(f"yaw={yaw_deg:.1f}")

        pose_text = " ".join(pose_parts)
        text = f"{cls_name}  {pose_text}" if self.draw_label else pose_text

        if text:
            self.draw_text_with_bg(debug_img, text, (x1, max(16, y1 - 6)), (0, 255, 0))

        next_y = y2 + 16

        if self.draw_spec and spec:
            spec_text = (
                f"w={float(spec.get('width', 0.0)):.1f} "
                f"d={float(spec.get('depth', 0.0)):.1f} "
                f"h={float(spec.get('height', 0.0)):.1f} "
                f"dur={int(spec.get('durability', 0))}"
            )
            self.draw_text_with_bg(
                debug_img,
                spec_text,
                (x1, min(debug_img.shape[0] - 8, next_y)),
                (255, 255, 0)
            )

    def compute_workspace_homography(self, color_bgr, debug_img):
        gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
        tags = self.detector.detect(gray, estimate_tag_pose=False)

        img_pts, ws_pts, tag_boxes = [], [], []

        for tag in tags:
            tag_id = int(tag.tag_id)
            corners = tag.corners.astype(int)

            min_x = int(np.min(corners[:, 0]))
            min_y = int(np.min(corners[:, 1]))
            max_x = int(np.max(corners[:, 0]))
            max_y = int(np.max(corners[:, 1]))
            tag_boxes.append((min_x, min_y, max_x, max_y))

            if self.draw_tag_boxes:
                for i in range(4):
                    cv2.line(debug_img, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)

                center = tuple(tag.center.astype(int))
                cv2.circle(debug_img, center, 4, (0, 0, 255), -1)
                self.draw_text_with_bg(debug_img, f"ID:{tag_id}", (center[0] + 8, center[1] - 8), (255, 0, 0))

            if tag_id in self.tag_world_points:
                img_pts.append([float(tag.center[0]), float(tag.center[1])])
                wx, wy = self.tag_world_points[tag_id]
                ws_pts.append([wx, wy])

        if self.draw_workspace_text:
            self.draw_text_with_bg(
                debug_img,
                f"workspace tags used: {len(img_pts)}",
                (20, 35),
                (0, 255, 0) if len(img_pts) >= 4 else (0, 0, 255)
            )

        if len(img_pts) < 4:
            return None, tag_boxes

        H, _ = cv2.findHomography(
            np.array(img_pts, dtype=np.float32),
            np.array(ws_pts, dtype=np.float32),
            method=cv2.RANSAC
        )
        return H, tag_boxes

    def compute_workspace_to_robot_homography(self):
        try:
            robot = self.robot_corner_points

            ws_pts = np.array([
                [self.workspace_size_mm, self.workspace_size_mm],
                [self.workspace_size_mm, 0.0],
                [0.0, self.workspace_size_mm],
                [0.0, 0.0],
            ], dtype=np.float32)

            robot_pts = np.array([
                [float(robot["lt"][0]), float(robot["lt"][1])],
                [float(robot["rt"][0]), float(robot["rt"][1])],
                [float(robot["lb"][0]), float(robot["lb"][1])],
                [float(robot["rb"][0]), float(robot["rb"][1])],
            ], dtype=np.float32)

            H, _ = cv2.findHomography(ws_pts, robot_pts, method=0)

            if H is None:
                self.get_logger().error("Failed to compute H_ws_to_robot")
            else:
                self.get_logger().info("Computed workspace->robot homography successfully.")

            return H

        except Exception as e:
            self.get_logger().error(f"compute_workspace_to_robot_homography failed: {e}")
            return None

    def workspace_to_robot(self, x_ws: float, y_ws: float):
        if self.H_ws_to_robot is None:
            return None
        return self.transform_point(self.H_ws_to_robot, (x_ws, y_ws))

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

    def bbox_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
        inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return 0.0 if union <= 0 else inter / union

    def is_overlapping_tag(self, box, tag_boxes, thr=0.15):
        return any(self.bbox_iou(box, tb) > thr for tb in tag_boxes)

    def filter_duplicate_and_small_boxes(self, box_items):
        filtered = []

        for box, x1, y1, x2, y2 in box_items:
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area < self.min_box_area_px:
                continue
            filtered.append((box, x1, y1, x2, y2))

        keep = [True] * len(filtered)

        for i in range(len(filtered)):
            if not keep[i]:
                continue

            box_i, x1_i, y1_i, x2_i, y2_i = filtered[i]
            cls_i = int(box_i.cls.item())
            area_i = max(0, x2_i - x1_i) * max(0, y2_i - y1_i)
            conf_i = float(box_i.conf.item())

            for j in range(i + 1, len(filtered)):
                if not keep[j]:
                    continue

                box_j, x1_j, y1_j, x2_j, y2_j = filtered[j]
                cls_j = int(box_j.cls.item())

                if cls_i != cls_j:
                    continue

                iou = self.bbox_iou((x1_i, y1_i, x2_i, y2_i), (x1_j, y1_j, x2_j, y2_j))
                if iou < self.dup_iou_thres:
                    continue

                area_j = max(0, x2_j - x1_j) * max(0, y2_j - y1_j)
                conf_j = float(box_j.conf.item())

                smaller = min(area_i, area_j)
                larger = max(area_i, area_j)

                if larger <= 0:
                    continue

                if smaller / larger <= self.small_box_area_ratio:
                    if area_i < area_j:
                        keep[i] = False
                        break
                    else:
                        keep[j] = False
                else:
                    if conf_i < conf_j:
                        keep[i] = False
                        break
                    else:
                        keep[j] = False

        return [filtered[i] for i in range(len(filtered)) if keep[i]]

    def normalize_yaw_deg_180(self, yaw_deg: float) -> float:
        while yaw_deg > 90.0:
            yaw_deg -= 180.0
        while yaw_deg <= -90.0:
            yaw_deg += 180.0
        return yaw_deg

    def apply_yaw_reference_shift(self, yaw_deg: float) -> float:
        return self.normalize_yaw_deg_180(yaw_deg + 90.0)

    def yaw_to_axis_0_180(self, yaw_deg: float) -> float:
        y = self.normalize_yaw_deg_180(yaw_deg)
        return y + 180.0 if y < 0.0 else y

    def robust_mode_yaw(self, yaw_list: List[float], bin_deg: float = 5.0) -> Optional[float]:
        if not yaw_list:
            return None

        bins = defaultdict(list)
        for yaw in yaw_list:
            axis_yaw = self.yaw_to_axis_0_180(float(yaw))
            bins[int(axis_yaw // bin_deg)].append(axis_yaw)

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

    def make_track_key(self, item_id: str, name: str, x: float, y: float) -> str:
        if item_id:
            return f"id:{item_id}"
        grid = max(self.track_grid_mm, 1.0)
        return f"{name}_{int(round(x / grid))}_{int(round(y / grid))}"

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
        return float(self.last_good_yaw_by_track.get(track_key, 0.0))

    def get_depth_median_mm(self, depth_img, cx, cy, k=2) -> Optional[float]:
        h, w = depth_img.shape[:2]
        x1, x2 = clamp(cx - k, 0, w - 1), clamp(cx + k + 1, 0, w)
        y1, y2 = clamp(cy - k, 0, h - 1), clamp(cy + k + 1, 0, h)

        patch = depth_img[y1:y2, x1:x2].astype(np.float32)
        vals = patch[np.isfinite(patch)]
        vals = vals[vals > 0]
        if len(vals) == 0:
            return None

        return float(np.median(vals)) * self.depth_value_scale_to_m * 1000.0

    def build_object_mask(self, roi_bgr, other_boxes=None):
        if roi_bgr is None or roi_bgr.size == 0:
            return None, None

        if other_boxes is None:
            other_boxes = []

        roi_h, roi_w = roi_bgr.shape[:2]
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(
            hsv,
            np.array([self.yellow_h_min, self.yellow_s_min, self.yellow_v_min], dtype=np.uint8),
            np.array([self.yellow_h_max, self.yellow_s_max, self.yellow_v_max], dtype=np.uint8),
        )
        fg_mask = 255 - yellow_mask

        protect_cx, protect_cy = roi_w * 0.5, roi_h * 0.5
        protect_r = min(roi_w, roi_h) * 0.22

        for rx1, ry1, rx2, ry2 in other_boxes:
            if rx2 <= rx1 or ry2 <= ry1:
                continue

            ob_cx, ob_cy = 0.5 * (rx1 + rx2), 0.5 * (ry1 + ry2)
            if math.hypot(ob_cx - protect_cx, ob_cy - protect_cy) < protect_r:
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
            return fg_mask, None

        cx_roi, cy_roi = roi_w * 0.5, roi_h * 0.5
        valid_contours, center_contours = [], []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 120:
                continue
            valid_contours.append(cnt)
            if cv2.pointPolygonTest(cnt, (cx_roi, cy_roi), False) >= 0:
                center_contours.append(cnt)

        if not valid_contours:
            return fg_mask, None

        def contour_score(cnt):
            area = cv2.contourArea(cnt)
            ccx, ccy = self.contour_center(cnt)
            return area - 4.0 * math.hypot(ccx - cx_roi, ccy - cy_roi)

        best_cnt = max(center_contours, key=contour_score) if center_contours else max(valid_contours, key=contour_score)

        selected_mask = np.zeros_like(fg_mask)
        cv2.drawContours(selected_mask, [best_cnt], -1, 255, thickness=-1)

        return selected_mask, best_cnt

    def masked_roi_from_mask(self, roi_bgr, mask):
        if roi_bgr is None or roi_bgr.size == 0 or mask is None:
            return roi_bgr
        out = np.zeros_like(roi_bgr)
        out[mask > 0] = roi_bgr[mask > 0]
        return out

    def estimate_yaw_deg_from_roi_ws(
        self,
        roi_bgr,
        roi_origin_xy,
        H_img_to_ws,
        other_boxes=None,
        return_masked_roi=False,
    ):
        if roi_bgr is None or roi_bgr.size == 0 or H_img_to_ws is None:
            return (None, None, roi_bgr) if return_masked_roi else (None, None)

        if other_boxes is None:
            other_boxes = []

        roi_h, roi_w = roi_bgr.shape[:2]
        ox, oy = roi_origin_xy

        other_boxes_local = []
        for bx1, by1, bx2, by2 in other_boxes:
            lx1 = clamp(bx1 - ox, 0, roi_w - 1)
            ly1 = clamp(by1 - oy, 0, roi_h - 1)
            lx2 = clamp(bx2 - ox, 0, roi_w - 1)
            ly2 = clamp(by2 - oy, 0, roi_h - 1)
            other_boxes_local.append((lx1, ly1, lx2, ly2))

        selected_mask, cnt = self.build_object_mask(roi_bgr, other_boxes_local)
        masked_roi = self.masked_roi_from_mask(roi_bgr, selected_mask)

        if cnt is None:
            return (None, None, masked_roi) if return_masked_roi else (None, None)

        pts_img = cnt.reshape(-1, 2).astype(np.float32)
        pts_img[:, 0] += float(ox)
        pts_img[:, 1] += float(oy)

        pts_ws = []
        for p in pts_img:
            pw = self.transform_point(H_img_to_ws, (float(p[0]), float(p[1])))
            if pw is not None:
                pts_ws.append([pw[0], pw[1]])

        if len(pts_ws) < 5:
            return (None, None, masked_roi) if return_masked_roi else (None, None)

        pts_ws = np.array(pts_ws, dtype=np.float32)
        rect = cv2.minAreaRect(pts_ws)
        (_, _), (rw, rh), _ = rect

        if rw < 1e-6 or rh < 1e-6:
            return (None, None, masked_roi) if return_masked_roi else (None, None)

        long_side, short_side = max(rw, rh), min(rw, rh)
        if long_side / max(short_side, 1e-6) < self.yaw_min_aspect_ratio:
            return (None, None, masked_roi) if return_masked_roi else (None, None)

        box_ws = cv2.boxPoints(rect).astype(np.float32)

        best_len, best_vec = -1.0, None
        for i in range(4):
            p0, p1 = box_ws[i], box_ws[(i + 1) % 4]
            vec = p1 - p0
            length = float(np.linalg.norm(vec))
            if length > best_len:
                best_len, best_vec = length, vec

        if best_vec is None or best_len < 1e-6:
            return (None, None, masked_roi) if return_masked_roi else (None, None)

        yaw_deg = -math.degrees(math.atan2(float(best_vec[1]), float(best_vec[0])))
        yaw_deg = self.apply_yaw_reference_shift(yaw_deg)

        if return_masked_roi:
            return yaw_deg, box_ws, masked_roi
        return yaw_deg, box_ws

    def estimate_yaw_deg_from_roi_img(
        self,
        roi_bgr,
        other_boxes=None,
        return_masked_roi=False,
    ):
        if roi_bgr is None or roi_bgr.size == 0:
            return (None, roi_bgr) if return_masked_roi else None

        if other_boxes is None:
            other_boxes = []

        roi_h, roi_w = roi_bgr.shape[:2]

        other_boxes_local = []
        for bx1, by1, bx2, by2 in other_boxes:
            lx1 = clamp(bx1, 0, roi_w - 1)
            ly1 = clamp(by1, 0, roi_h - 1)
            lx2 = clamp(bx2, 0, roi_w - 1)
            ly2 = clamp(by2, 0, roi_h - 1)
            other_boxes_local.append((lx1, ly1, lx2, ly2))

        selected_mask, cnt = self.build_object_mask(roi_bgr, other_boxes_local)
        masked_roi = self.masked_roi_from_mask(roi_bgr, selected_mask)

        if cnt is None:
            return (None, masked_roi) if return_masked_roi else None

        pts = cnt.reshape(-1, 2).astype(np.float32)
        if len(pts) < 5:
            return (None, masked_roi) if return_masked_roi else None

        rect = cv2.minAreaRect(pts)
        (_, _), (rw, rh), _ = rect

        if rw < 1e-6 or rh < 1e-6:
            return (None, masked_roi) if return_masked_roi else None

        long_side, short_side = max(rw, rh), min(rw, rh)
        if long_side / max(short_side, 1e-6) < self.yaw_min_aspect_ratio:
            return (None, masked_roi) if return_masked_roi else None

        box = cv2.boxPoints(rect).astype(np.float32)

        best_len, best_vec = -1.0, None
        for i in range(4):
            p0, p1 = box[i], box[(i + 1) % 4]
            vec = p1 - p0
            length = float(np.linalg.norm(vec))
            if length > best_len:
                best_len, best_vec = length, vec

        if best_vec is None or best_len < 1e-6:
            return (None, masked_roi) if return_masked_roi else None

        yaw_deg = -math.degrees(math.atan2(-float(best_vec[1]), float(best_vec[0])))
        yaw_deg = self.apply_yaw_reference_shift(yaw_deg)

        if return_masked_roi:
            return yaw_deg, masked_roi
        return yaw_deg

    def publish_debug(self, img_bgr, stamp):
        try:
            if img_bgr is None:
                return

            if self.show_cv:
                try:
                    display_img = img_bgr
                    cv2.imshow("vision_debug", display_img)
                    cv2.resizeWindow("vision_debug", self.cv_window_width, self.cv_window_height)
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

        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()