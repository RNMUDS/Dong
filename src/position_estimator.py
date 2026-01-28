"""
実空間位置推定
単眼カメラ + 既知サイズから3D位置を推定
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from object_detector import Detection


@dataclass
class Position3D:
    """3D位置情報"""
    x: float  # 左右方向 (mm) - 右が正
    y: float  # 上下方向 (mm) - 下が正
    z: float  # 奥行き/距離 (mm)
    class_name: str
    confidence: float

    def to_meters(self) -> Tuple[float, float, float]:
        """メートル単位に変換"""
        return (self.x / 1000, self.y / 1000, self.z / 1000)

    def __str__(self):
        return f"{self.class_name}: X={self.x:.0f}mm, Y={self.y:.0f}mm, Z={self.z:.0f}mm"


class PositionEstimator:
    def __init__(self, camera_matrix: np.ndarray, known_sizes: Dict[str, Dict[str, float]],
                 image_size: Tuple[int, int] = (1280, 720)):
        """
        Args:
            camera_matrix: カメラ内部パラメータ行列 (3x3)
            known_sizes: 物体の既知サイズ {"class_name": {"height": mm, "width": mm}}
            image_size: 画像サイズ (width, height)
        """
        self.camera_matrix = camera_matrix
        self.known_sizes = known_sizes
        self.image_size = image_size

        # 焦点距離
        self.fx = camera_matrix[0, 0]
        self.fy = camera_matrix[1, 1]

        # 光学中心
        self.cx = camera_matrix[0, 2]
        self.cy = camera_matrix[1, 2]

    def estimate_distance(self, detection: Detection, use_height: bool = True) -> float:
        """
        検出物体の距離を推定

        距離 = (実際のサイズ × 焦点距離) / 検出されたピクセルサイズ

        Args:
            detection: 検出結果
            use_height: Trueなら高さ、Falseなら幅を使用

        Returns:
            推定距離 (mm)
        """
        # 既知サイズを取得
        class_name = detection.class_name
        if class_name in self.known_sizes:
            sizes = self.known_sizes[class_name]
        else:
            sizes = self.known_sizes.get("default", {"height": 100.0, "width": 100.0})

        if use_height:
            real_size = sizes["height"]
            pixel_size = detection.height
            focal_length = self.fy
        else:
            real_size = sizes["width"]
            pixel_size = detection.width
            focal_length = self.fx

        if pixel_size <= 0:
            return float('inf')

        # 距離を計算
        distance = (real_size * focal_length) / pixel_size

        return distance

    def estimate_3d_position(self, detection: Detection, distance: Optional[float] = None) -> Position3D:
        """
        検出物体の3D位置を推定

        Args:
            detection: 検出結果
            distance: 事前計算した距離（省略時は自動計算）

        Returns:
            3D位置情報
        """
        if distance is None:
            distance = self.estimate_distance(detection)

        # 画像座標から正規化座標へ
        cx_pixel, cy_pixel = detection.center

        # カメラ座標系での位置を計算
        # X = (u - cx) * Z / fx
        # Y = (v - cy) * Z / fy
        x = (cx_pixel - self.cx) * distance / self.fx
        y = (cy_pixel - self.cy) * distance / self.fy
        z = distance

        return Position3D(
            x=x,
            y=y,
            z=z,
            class_name=detection.class_name,
            confidence=detection.confidence
        )

    def estimate_distance_with_smoothing(self, detection: Detection,
                                         prev_distance: Optional[float] = None,
                                         alpha: float = 0.3) -> float:
        """
        距離推定にスムージングを適用（ノイズ低減）

        Args:
            detection: 検出結果
            prev_distance: 前フレームの距離
            alpha: スムージング係数（0-1、大きいほど新しい値を重視）

        Returns:
            スムージングされた距離 (mm)
        """
        current_distance = self.estimate_distance(detection)

        if prev_distance is None:
            return current_distance

        # 指数移動平均
        smoothed = alpha * current_distance + (1 - alpha) * prev_distance

        return smoothed


class MultiObjectTracker:
    """複数物体の位置をトラッキング"""

    def __init__(self, position_estimator: PositionEstimator, smoothing_alpha: float = 0.3):
        self.estimator = position_estimator
        self.smoothing_alpha = smoothing_alpha

        # 各物体の前フレーム距離を保持
        self.prev_distances: Dict[int, float] = {}

    def update(self, detections: list) -> list:
        """
        複数の検出結果から3D位置を推定

        Args:
            detections: 検出結果のリスト

        Returns:
            Position3Dのリスト
        """
        positions = []
        new_prev_distances = {}

        for i, det in enumerate(detections):
            # スムージング用の前フレーム距離を取得
            prev_dist = self.prev_distances.get(i)

            # 距離推定（スムージング付き）
            distance = self.estimator.estimate_distance_with_smoothing(
                det, prev_dist, self.smoothing_alpha
            )

            # 3D位置を計算
            pos = self.estimator.estimate_3d_position(det, distance)
            positions.append(pos)

            new_prev_distances[i] = distance

        self.prev_distances = new_prev_distances

        return positions

    def reset(self):
        """トラッキング状態をリセット"""
        self.prev_distances.clear()


def create_default_camera_matrix(image_width: int = 1280, image_height: int = 720) -> np.ndarray:
    """
    キャリブレーションなしで使用するデフォルトのカメラ行列を生成

    一般的なWebカメラの焦点距離を仮定（精度は低い）
    """
    # 一般的なWebカメラの水平FOVは約60度と仮定
    fov_horizontal = 60  # degrees
    fov_rad = np.radians(fov_horizontal)

    # 焦点距離を計算
    fx = image_width / (2 * np.tan(fov_rad / 2))
    fy = fx  # アスペクト比1:1と仮定

    # 光学中心は画像中心
    cx = image_width / 2
    cy = image_height / 2

    camera_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    return camera_matrix


if __name__ == "__main__":
    # テスト
    camera_matrix = create_default_camera_matrix()
    print("デフォルトカメラ行列:")
    print(camera_matrix)

    known_sizes = {
        "bottle": {"height": 200.0, "width": 65.0},
        "cell phone": {"height": 150.0, "width": 70.0},
        "default": {"height": 100.0, "width": 100.0},
    }

    estimator = PositionEstimator(camera_matrix, known_sizes)

    # ダミーの検出結果でテスト
    dummy_detection = Detection(
        class_id=0,
        class_name="bottle",
        confidence=0.9,
        bbox=(500, 300, 550, 400),
        center=(525, 350),
        width=50,
        height=100
    )

    distance = estimator.estimate_distance(dummy_detection)
    print(f"\n推定距離: {distance:.0f}mm ({distance/1000:.2f}m)")

    pos = estimator.estimate_3d_position(dummy_detection, distance)
    print(f"3D位置: {pos}")
