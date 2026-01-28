#!/usr/bin/env python3
"""
実空間位置推定システム - メインエントリーポイント

使用方法:
    # 1. キャリブレーション画像の収集
    python run.py calibrate --collect

    # 2. キャリブレーションの実行
    python run.py calibrate --run

    # 3. リアルタイム位置推定（全物体）
    python run.py track

    # 4. 特定の物体のみ追跡
    python run.py track --classes bottle cup
"""

import sys
import os

# プロジェクトルートをパスに追加
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "src"))
os.chdir(project_root)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="実空間位置推定システム",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python run.py calibrate --collect   # キャリブレーション画像を収集
  python run.py calibrate --run       # キャリブレーションを実行
  python run.py track                 # リアルタイム位置推定を開始
  python run.py track --classes bottle cup  # 特定クラスのみ追跡
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="コマンド")

    # calibrate コマンド
    calib_parser = subparsers.add_parser("calibrate", help="カメラキャリブレーション")
    calib_parser.add_argument("--collect", action="store_true", help="キャリブレーション画像を収集")
    calib_parser.add_argument("--run", action="store_true", help="キャリブレーションを実行")
    calib_parser.add_argument("--camera", type=int, default=0, help="カメラ番号")
    calib_parser.add_argument("--images", type=int, default=15, help="収集する画像枚数")

    # track コマンド
    track_parser = subparsers.add_parser("track", help="リアルタイム位置推定")
    track_parser.add_argument("--classes", nargs="+", help="検出対象クラス")
    track_parser.add_argument("--no-gui", action="store_true", help="GUIを表示しない")
    track_parser.add_argument("--config", default="config/settings.yaml", help="設定ファイル")

    # detect コマンド（YOLO検出のみ）
    detect_parser = subparsers.add_parser("detect", help="物体検出のみ（位置推定なし）")
    detect_parser.add_argument("--camera", type=int, default=0, help="カメラ番号")

    args = parser.parse_args()

    if args.command == "calibrate":
        from camera_calibration import CameraCalibrator

        calibrator = CameraCalibrator(
            chessboard_size=(9, 6),
            square_size=25.0
        )

        if args.collect:
            calibrator.collect_calibration_images(
                camera_index=args.camera,
                num_images=args.images
            )

        if args.run:
            if calibrator.calibrate_from_images():
                calibrator.save_calibration()

    elif args.command == "track":
        from realtime_position_tracker import RealtimePositionTracker

        tracker = RealtimePositionTracker(args.config)
        tracker.run(target_classes=args.classes, show_gui=not args.no_gui)

    elif args.command == "detect":
        from object_detector import ObjectDetector
        import cv2

        detector = ObjectDetector()
        cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("ESCキーで終了")

        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            detections = detector.detect(frame)
            output = detector.draw_detections(frame, detections)

            cv2.imshow("Object Detection", output)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
