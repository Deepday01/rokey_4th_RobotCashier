#!/usr/bin/env python3
import json
from typing import Set, List

import cv2
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from cv_bridge import CvBridge
from pyzbar.pyzbar import decode
from ultralytics import YOLO


class VisionScanItemsNode(Node):

    def __init__(self):
        super().__init__('vision_scan_items_node')

        # -----------------------------
        # parameters
        # -----------------------------
        self.declare_parameter('color_topic', '/camera/camera/color/image_raw/compressed')
        self.declare_parameter('use_compressed', True)
        self.declare_parameter('show_window', True)

        self.declare_parameter('model_path', '/home/daehyuk/Downloads/best.pt')

        # detection params
        self.declare_parameter('conf_thres', 0.35)
        self.declare_parameter('iou_thres', 0.35)
        self.declare_parameter('max_det', 10)
        self.declare_parameter('imgsz', 960)

        self.declare_parameter('target_count', 5)

        self.color_topic = self.get_parameter('color_topic').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.show_window = self.get_parameter('show_window').value

        self.model_path = self.get_parameter('model_path').value
        self.conf_thres = self.get_parameter('conf_thres').value
        self.iou_thres = self.get_parameter('iou_thres').value
        self.max_det = self.get_parameter('max_det').value
        self.imgsz = self.get_parameter('imgsz').value

        self.target_count = self.get_parameter('target_count').value

        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)

        self.item_order = ["halls", "insect", "dino", "candy", "caramel"]

        self.qr_map = {
            "https://m.site.naver.com/20lZW": "halls",
            "https://m.site.naver.com/20m1A": "insect",
            "https://m.site.naver.com/20suF": "caramel",
            "https://m.site.naver.com/20svK": "candy",
            "https://m.site.naver.com/20swh": "dino",
        }

        self.model_names = self.model.names
        self.target_class_ids = self.get_target_class_ids(self.item_order)

        self.found_items: Set[str] = set()
        self.published_once = False

        self.scan_pub = self.create_publisher(String, '/vision/scan_items', 10)
        self.debug_pub = self.create_publisher(Image, '/qr/debug_image', 10)

        if self.use_compressed:
            self.sub = self.create_subscription(
                CompressedImage,
                self.color_topic,
                self.callback_compressed,
                10
            )
        else:
            self.sub = self.create_subscription(
                Image,
                self.color_topic,
                self.callback_raw,
                10
            )

        self.get_logger().info(f"Loaded model: {self.model_path}")
        self.get_logger().info(f"Model names: {self.model_names}")
        self.get_logger().info(f"Target class ids: {self.target_class_ids}")
        self.get_logger().info("YOLO visualize + FULL FRAME QR decode started")

    # ------------------------------------------------

    def get_target_class_ids(self, target_names: List[str]) -> List[int]:
        ids = []
        for cls_id, cls_name in self.model_names.items():
            if cls_name in target_names:
                ids.append(int(cls_id))
        return ids

    # ------------------------------------------------

    def decode_qr(self, img):
        decoded = set()

        if img is None or img.size == 0:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(2.0, (8, 8))
        gray_clahe = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray_clahe, (3, 3), 0)

        _, otsu = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        adaptive = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )

        candidates = [img, gray, gray_clahe, otsu, adaptive]

        for c in candidates:
            results = decode(c)
            for r in results:
                try:
                    raw = r.data.decode("utf-8").strip()
                    decoded.add(raw)
                except Exception:
                    pass

        return list(decoded)

    # ------------------------------------------------

    def decode_qr_multi_stage_fullframe(self, frame):
        raw_list = self.decode_qr(frame)
        if raw_list:
            return raw_list

        scale = 2
        if min(frame.shape[:2]) < 720:
            scale = 3

        big = cv2.resize(
            frame, None,
            fx=scale, fy=scale,
            interpolation=cv2.INTER_CUBIC
        )
        raw_list = self.decode_qr(big)
        if raw_list:
            return raw_list

        return []

    # ------------------------------------------------

    def run_yolo(self, frame):
        return self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=self.imgsz,
            max_det=self.max_det,
            classes=self.target_class_ids if len(self.target_class_ids) > 0 else None,
            verbose=False
        )

    # ------------------------------------------------

    def process(self, frame, header):
        vis = frame.copy()

        # 1) 프레임 전체에서 QR 읽기
        raw_list = self.decode_qr_multi_stage_fullframe(frame)

        qr_found_now = []
        for raw in raw_list:
            name = self.qr_map.get(raw, "unknown")
            if name != "unknown":
                qr_found_now.append(name)
                self.found_items.add(name)

        # 2) YOLO는 박스 시각화/보조용
        results = self.run_yolo(frame)

        if results and results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                det_name = self.model_names.get(cls_id, str(cls_id))

                box_w = x2 - x1
                box_h = y2 - y1
                area = box_w * box_h
                if area < 2500:
                    continue

                # QR 전체프레임 성공 여부만 표시
                matched = det_name in qr_found_now or det_name in self.found_items
                color = (0, 255, 0) if matched else (0, 165, 255)

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

                if matched:
                    label = f"{det_name} {conf:.2f} | qr_ok"
                else:
                    label = f"{det_name} {conf:.2f} | qr_wait"

                cv2.putText(
                    vis,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

        ordered = [i for i in self.item_order if i in self.found_items]

        cv2.putText(
            vis,
            f'QR_NOW:{qr_found_now}',
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            vis,
            f'FOUND:{ordered}',
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        if not self.published_once and len(self.found_items) >= self.target_count:
            msg = String()
            msg.data = json.dumps(ordered)
            self.scan_pub.publish(msg)
            self.published_once = True
            self.get_logger().info(f"Published scan items: {ordered}")

        debug = self.bridge.cv2_to_imgmsg(vis, "bgr8")
        debug.header = header
        self.debug_pub.publish(debug)

        if self.show_window:
            cv2.imshow("vision_scan_items_node", vis)
            cv2.waitKey(1)

    # ------------------------------------------------

    def callback_raw(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.process(frame, msg.header)

    def callback_compressed(self, msg):
        frame = self.bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        self.process(frame, msg.header)

    # ------------------------------------------------


def main():
    rclpy.init()
    node = VisionScanItemsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == "__main__":
    main()