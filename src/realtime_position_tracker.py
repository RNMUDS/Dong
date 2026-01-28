"""
リアルタイム位置推定システム
WebカメラとYOLOを使用して物体の実空間位置を推定
"""

import cv2
import numpy as np
import yaml
import os
import sys
from pathlib import Path
from typing import Optional, List

from object_detector import ObjectDetector, Detection
from position_estimator import (
    PositionEstimator,
    MultiObjectTracker,
    Position3D,
    create_default_camera_matrix
)
from camera_calibration import CameraCalibrator


class RealtimePositionTracker:
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定読み込み
        self.config = self._load_config(config_path)

        # カメラ行列の取得
        self.camera_matrix = self._get_camera_matrix()

        # 物体検出器
        yolo_config = self.config.get("yolo", {})
        self.detector = ObjectDetector(
            model_path=yolo_config.get("model", "yolov8n.pt"),
            confidence=yolo_config.get("confidence", 0.5),
            device=yolo_config.get("device", "cpu")
        )

        # 位置推定器
        known_sizes = self.config.get("objects", {})
        cam_config = self.config.get("camera", {})
        image_size = (cam_config.get("width", 1280), cam_config.get("height", 720))

        self.estimator = PositionEstimator(
            self.camera_matrix,
            known_sizes,
            image_size
        )

        # マルチオブジェクトトラッカー
        self.tracker = MultiObjectTracker(self.estimator, smoothing_alpha=0.3)

        # カメラ設定
        self.camera_index = cam_config.get("index", 0)
        self.frame_width = cam_config.get("width", 1280)
        self.frame_height = cam_config.get("height", 720)

    def _load_config(self, config_path: str) -> dict:
        """設定ファイルを読み込み"""
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}

    def _get_camera_matrix(self) -> np.ndarray:
        """カメラ行列を取得（キャリブレーション済みまたはデフォルト）"""
        calib_config = self.config.get("calibration", {})
        calib_file = calib_config.get("calibration_file", "config/camera_calibration.yaml")

        if os.path.exists(calib_file):
            print(f"キャリブレーションデータを読み込み: {calib_file}")
            calibrator = CameraCalibrator()
            if calibrator.load_calibration(calib_file):
                return calibrator.camera_matrix

        # キャリブレーションファイルがない場合はデフォルト値を使用
        print("キャリブレーションデータがありません。デフォルト値を使用します。")
        print("精度向上のため、カメラキャリブレーションを推奨します。")
        cam_config = self.config.get("camera", {})
        return create_default_camera_matrix(
            cam_config.get("width", 1280),
            cam_config.get("height", 720)
        )

    def run(self, target_classes: Optional[List[str]] = None, show_gui: bool = True):
        """
        リアルタイム位置推定を実行

        Args:
            target_classes: 検出対象のクラス名リスト（Noneで全クラス）
            show_gui: GUIを表示するか
        """
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        if not cap.isOpened():
            raise RuntimeError(f"カメラ {self.camera_index} を開けませんでした")

        print("\n=== リアルタイム位置推定 ===")
        print("ESC: 終了 / R: トラッカーリセット")
        if target_classes:
            print(f"検出対象: {target_classes}")
        print()

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            # 物体検出
            if target_classes:
                detections = self.detector.detect_specific_classes(frame, target_classes)
            else:
                detections = self.detector.detect(frame)

            # 3D位置推定
            positions = self.tracker.update(detections)

            # 結果を表示
            if show_gui:
                output = self._draw_results(frame, detections, positions)
                cv2.imshow("Position Tracker", output)

            # コンソール出力
            self._print_positions(positions)

            # キー入力処理
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r') or key == ord('R'):
                self.tracker.reset()
                print("トラッカーをリセットしました")

        cap.release()
        cv2.destroyAllWindows()

    def _draw_results(self, frame: np.ndarray, detections: List[Detection],
                      positions: List[Position3D]) -> np.ndarray:
        """結果を画像に描画"""
        output = frame.copy()

        for det, pos in zip(detections, positions):
            x1, y1, x2, y2 = det.bbox

            # バウンディングボックス
            color = (0, 255, 0)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # 情報表示
            dist_m = pos.z / 1000
            label1 = f"{det.class_name}: {det.confidence:.2f}"
            label2 = f"Z: {dist_m:.2f}m"
            label3 = f"X: {pos.x/1000:.2f}m Y: {pos.y/1000:.2f}m"

            # ラベル背景
            y_offset = y1 - 10
            for label in [label3, label2, label1]:
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(output, (x1, y_offset - h - 5), (x1 + w + 5, y_offset + 5), color, -1)
                cv2.putText(output, label, (x1 + 2, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                y_offset -= h + 10

            # 中心点
            cv2.circle(output, det.center, 5, (0, 0, 255), -1)

        # 座標系の説明
        cv2.putText(output, "Coordinate: X(right+) Y(down+) Z(depth)",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return output

    def _print_positions(self, positions: List[Position3D]):
        """位置情報をコンソールに出力"""
        if not positions:
            return

        # カーソルを上に移動して上書き（見やすさのため）
        print("\033[2K", end="")  # 現在行をクリア
        for pos in positions:
            x_m, y_m, z_m = pos.to_meters()
            print(f"\r{pos.class_name}: X={x_m:+.2f}m, Y={y_m:+.2f}m, Z={z_m:.2f}m  ", end="")
        print("", end="\r")

    def get_positions(self, frame: np.ndarray,
                      target_classes: Optional[List[str]] = None) -> List[Position3D]:
        """
        単一フレームから位置を取得（外部連携用）

        Args:
            frame: BGR画像
            target_classes: 検出対象クラス

        Returns:
            Position3Dのリスト
        """
        if target_classes:
            detections = self.detector.detect_specific_classes(frame, target_classes)
        else:
            detections = self.detector.detect(frame)

        return self.tracker.update(detections)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="リアルタイム位置推定")
    parser.add_argument("--config", default="config/settings.yaml", help="設定ファイル")
    parser.add_argument("--classes", nargs="+", help="検出対象クラス（例: bottle cup）")
    parser.add_argument("--no-gui", action="store_true", help="GUIを表示しない")

    args = parser.parse_args()

    # プロジェクトルートに移動
    script_dir = Path(__file__).parent.parent
    os.chdir(script_dir)

    tracker = RealtimePositionTracker(args.config)
    tracker.run(target_classes=args.classes, show_gui=not args.no_gui)


if __name__ == "__main__":
    main()
