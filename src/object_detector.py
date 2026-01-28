"""
YOLOv8による物体検出
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Detection:
    """検出結果を格納するデータクラス"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: Tuple[int, int]  # (cx, cy)
    width: int  # バウンディングボックスの幅 (pixels)
    height: int  # バウンディングボックスの高さ (pixels)


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5, device="cpu"):
        """
        Args:
            model_path: YOLOモデルのパス
            confidence: 信頼度の閾値
            device: "cpu", "cuda", or "mps"
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.device = device

        # COCO クラス名
        self.class_names = self.model.names

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        フレームから物体を検出

        Args:
            frame: BGR画像（OpenCV形式）

        Returns:
            検出結果のリスト
        """
        results = self.model(frame, conf=self.confidence, device=self.device, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                # バウンディングボックス座標
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # クラス情報
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]
                confidence = float(box.conf[0])

                # 中心座標とサイズ
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                width = x2 - x1
                height = y2 - y1

                detection = Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    center=(cx, cy),
                    width=width,
                    height=height
                )
                detections.append(detection)

        return detections

    def detect_specific_classes(self, frame: np.ndarray, target_classes: List[str]) -> List[Detection]:
        """
        特定のクラスのみを検出

        Args:
            frame: BGR画像
            target_classes: 検出したいクラス名のリスト

        Returns:
            検出結果のリスト
        """
        all_detections = self.detect(frame)
        return [d for d in all_detections if d.class_name in target_classes]

    def draw_detections(self, frame: np.ndarray, detections: List[Detection],
                        show_distance: bool = False, distances: Optional[List[float]] = None) -> np.ndarray:
        """
        検出結果を画像に描画

        Args:
            frame: 元画像
            detections: 検出結果
            show_distance: 距離を表示するか
            distances: 各検出物体の距離 (mm)

        Returns:
            描画された画像
        """
        output = frame.copy()

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det.bbox

            # バウンディングボックス
            color = (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # ラベル
            label = f"{det.class_name}: {det.confidence:.2f}"
            if show_distance and distances and i < len(distances):
                dist_m = distances[i] / 1000  # mm → m
                label += f" | {dist_m:.2f}m"

            # ラベル背景
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # 中心点
            cv2.circle(output, det.center, 5, (0, 0, 255), -1)

        return output


def main():
    """テスト用"""
    detector = ObjectDetector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("ESCキーで終了")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        output = detector.draw_detections(frame, detections)

        # 検出情報を表示
        for det in detections:
            print(f"{det.class_name}: center={det.center}, size=({det.width}x{det.height})")

        cv2.imshow("Object Detection", output)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
