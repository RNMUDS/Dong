"""
カメラキャリブレーション
チェスボード画像を使用してカメラの内部パラメータを取得
"""

import cv2
import numpy as np
import yaml
import os
from pathlib import Path


class CameraCalibrator:
    def __init__(self, chessboard_size=(9, 6), square_size=25.0):
        """
        Args:
            chessboard_size: チェスボードの内側コーナー数 (columns, rows)
            square_size: 各マスの実際のサイズ (mm)
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # キャリブレーション結果
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None

        # 3D点の準備（チェスボードの実世界座標）
        self.objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= square_size

    def collect_calibration_images(self, camera_index=0, save_dir="calibration_images", num_images=15):
        """
        Webカメラからキャリブレーション画像を収集

        Args:
            camera_index: カメラデバイス番号
            save_dir: 画像保存ディレクトリ
            num_images: 必要な画像枚数
        """
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        if not cap.isOpened():
            raise RuntimeError(f"カメラ {camera_index} を開けませんでした")

        collected = 0
        print(f"チェスボードを様々な角度で撮影してください（{num_images}枚必要）")
        print("スペースキー: 撮影 / ESC: 終了")

        while collected < num_images:
            ret, frame = cap.read()
            if not ret:
                continue

            # チェスボード検出を試みる
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret_corners, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            display = frame.copy()

            if ret_corners:
                cv2.drawChessboardCorners(display, self.chessboard_size, corners, ret_corners)
                cv2.putText(display, "Chessboard detected! Press SPACE to capture",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Looking for chessboard...",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(display, f"Collected: {collected}/{num_images}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Calibration", display)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == 32 and ret_corners:  # SPACE
                filename = os.path.join(save_dir, f"calib_{collected:02d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"保存: {filename}")
                collected += 1

        cap.release()
        cv2.destroyAllWindows()

        print(f"\n{collected}枚の画像を収集しました")
        return collected >= num_images

    def calibrate_from_images(self, image_dir="calibration_images"):
        """
        保存された画像からキャリブレーションを実行

        Args:
            image_dir: キャリブレーション画像のディレクトリ

        Returns:
            bool: キャリブレーション成功/失敗
        """
        image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))

        if len(image_paths) < 5:
            print(f"画像が不足しています（最低5枚必要、現在{len(image_paths)}枚）")
            return False

        objpoints = []  # 3D点
        imgpoints = []  # 2D点
        img_size = None

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_size = gray.shape[::-1]

            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            if ret:
                # サブピクセル精度でコーナーを精緻化
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                objpoints.append(self.objp)
                imgpoints.append(corners_refined)

        if len(objpoints) < 5:
            print(f"有効な画像が不足しています（{len(objpoints)}枚）")
            return False

        print(f"{len(objpoints)}枚の画像でキャリブレーションを実行中...")

        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )

        if ret:
            print("キャリブレーション成功!")
            print(f"\nカメラ行列:\n{self.camera_matrix}")
            print(f"\n歪み係数:\n{self.dist_coeffs.ravel()}")

            # 再投影誤差を計算
            total_error = 0
            for i in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[i], self.rvecs[i],
                                                   self.tvecs[i], self.camera_matrix,
                                                   self.dist_coeffs)
                error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                total_error += error

            print(f"\n平均再投影誤差: {total_error / len(objpoints):.4f} pixels")

        return ret

    def save_calibration(self, filepath="config/camera_calibration.yaml"):
        """キャリブレーション結果をYAMLファイルに保存"""
        if self.camera_matrix is None:
            print("キャリブレーションデータがありません")
            return False

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        data = {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "focal_length_x": float(self.camera_matrix[0, 0]),
            "focal_length_y": float(self.camera_matrix[1, 1]),
            "principal_point_x": float(self.camera_matrix[0, 2]),
            "principal_point_y": float(self.camera_matrix[1, 2]),
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"キャリブレーションデータを保存: {filepath}")
        return True

    def load_calibration(self, filepath="config/camera_calibration.yaml"):
        """キャリブレーション結果をYAMLファイルから読み込み"""
        if not os.path.exists(filepath):
            print(f"ファイルが見つかりません: {filepath}")
            return False

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        self.camera_matrix = np.array(data["camera_matrix"])
        self.dist_coeffs = np.array(data["dist_coeffs"])

        print(f"キャリブレーションデータを読み込み: {filepath}")
        return True

    def get_focal_length(self):
        """焦点距離を取得 (pixels)"""
        if self.camera_matrix is None:
            return None
        return (self.camera_matrix[0, 0], self.camera_matrix[1, 1])


def main():
    """キャリブレーション実行用スクリプト"""
    import argparse

    parser = argparse.ArgumentParser(description="カメラキャリブレーション")
    parser.add_argument("--collect", action="store_true", help="キャリブレーション画像を収集")
    parser.add_argument("--calibrate", action="store_true", help="キャリブレーションを実行")
    parser.add_argument("--camera", type=int, default=0, help="カメラ番号")
    parser.add_argument("--images", type=int, default=15, help="収集する画像枚数")
    parser.add_argument("--cols", type=int, default=9, help="チェスボードの列数")
    parser.add_argument("--rows", type=int, default=6, help="チェスボードの行数")
    parser.add_argument("--square", type=float, default=25.0, help="マスのサイズ(mm)")

    args = parser.parse_args()

    calibrator = CameraCalibrator(
        chessboard_size=(args.cols, args.rows),
        square_size=args.square
    )

    if args.collect:
        calibrator.collect_calibration_images(
            camera_index=args.camera,
            num_images=args.images
        )

    if args.calibrate:
        if calibrator.calibrate_from_images():
            calibrator.save_calibration()


if __name__ == "__main__":
    main()
