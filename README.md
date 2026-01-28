# Dong - 実空間位置推定システム

WebカメラとYOLOv8を使用して、物体の実空間での3D位置をリアルタイムに推定するシステムです。

## 仕組み

### 原理：単眼カメラによる距離推定

単眼カメラでは直接的に深度（奥行き）を取得できませんが、**物体の実際のサイズが既知**であれば、以下の原理で距離を推定できます。

```
距離 = (実際のサイズ × 焦点距離) / 検出されたピクセルサイズ
```

![原理図](https://latex.codecogs.com/svg.image?Z=\frac{H_{real}\times&space;f}{H_{pixel}})

- `Z`: 物体までの距離 (mm)
- `H_real`: 物体の実際のサイズ (mm)
- `f`: カメラの焦点距離 (pixels)
- `H_pixel`: 画像上での物体のサイズ (pixels)

### 処理フロー

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Webカメラ   │ -> │  YOLOv8     │ -> │  距離推定    │ -> │  3D位置計算  │
│  画像取得    │    │  物体検出    │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                  │                  │
                          v                  v                  v
                   バウンディング       既知サイズと         カメラ座標系
                   ボックス取得        ピクセルサイズ比較    (X, Y, Z)
```

### 座標系

出力される座標はカメラ座標系です：

- **X軸**: 右方向が正 (mm)
- **Y軸**: 下方向が正 (mm)
- **Z軸**: カメラからの奥行き/距離 (mm)

```
        Y (下+)
        │
        │
        └───── X (右+)
       /
      /
     Z (奥+)
   📷 カメラ
```

## インストール

### 必要条件

- Python 3.8以上
- Webカメラ

### セットアップ

```bash
git clone https://github.com/RNMUDS/Dong.git
cd Dong
pip install -r requirements.txt
```

## 使い方

### 1. リアルタイム位置推定（基本）

```bash
python run.py track
```

- ESCキーで終了
- Rキーでトラッカーリセット

### 2. 特定の物体のみ追跡

```bash
# ボトルとカップのみ
python run.py track --classes bottle cup

# スマートフォンのみ
python run.py track --classes "cell phone"
```

### 3. カメラキャリブレーション（精度向上）

チェスボードパターン（9x6の内部コーナー）を用意してください。

```bash
# キャリブレーション画像を収集（15枚）
python run.py calibrate --collect

# キャリブレーションを実行
python run.py calibrate --run
```

### 4. 物体検出のみ（位置推定なし）

```bash
python run.py detect
```

## 設定

### 物体サイズの設定

`config/settings.yaml` で検出したい物体の実際のサイズを設定します：

```yaml
objects:
  # ペットボトル
  bottle:
    height: 200.0  # mm
    width: 65.0

  # スマートフォン
  cell phone:
    height: 150.0
    width: 70.0

  # カップ
  cup:
    height: 95.0
    width: 80.0

  # 人物（全身の高さ）
  person:
    height: 1700.0  # mm
    width: 500.0

  # 未登録の物体用デフォルト
  default:
    height: 100.0
    width: 100.0
```

### カメラ設定

```yaml
camera:
  index: 0        # カメラデバイス番号
  width: 1280     # 解像度
  height: 720

yolo:
  model: "yolov8n.pt"  # n/s/m/l/x から選択
  confidence: 0.5       # 検出閾値
  device: "cpu"         # cpu / cuda / mps
```

## 検出可能な物体

YOLOv8はCOCOデータセットの80クラスを検出できます：

| カテゴリ | クラス名 |
|---------|---------|
| 人物 | person |
| 乗り物 | bicycle, car, motorcycle, bus, truck |
| 動物 | cat, dog, bird, horse |
| 日用品 | bottle, cup, chair, couch, bed, dining table |
| 電子機器 | cell phone, laptop, tv, keyboard, mouse |
| 食品 | banana, apple, orange, pizza |

全クラス一覧は[COCOデータセット](https://cocodataset.org/#explore)を参照してください。

## 精度について

| 条件 | 推定誤差 |
|------|---------|
| キャリブレーションなし | ±20-30% |
| キャリブレーションあり | ±10-15% |
| 物体サイズが正確 | さらに向上 |

### 精度向上のポイント

1. **カメラキャリブレーション**を実行する
2. **物体の正確なサイズ**を `settings.yaml` に設定する
3. **照明条件**を一定に保つ
4. **物体が正面を向いている**状態で測定する

### 制限事項

- 物体が傾いていると誤差が大きくなる
- 遠距離（5m以上）では精度が低下
- 部分的に隠れた物体は正確に測定できない

## ファイル構成

```
Dong/
├── run.py                  # メインエントリーポイント
├── requirements.txt        # 依存パッケージ
├── config/
│   ├── settings.yaml       # 設定ファイル
│   └── camera_calibration.yaml  # キャリブレーション結果（生成）
├── calibration_images/     # キャリブレーション画像（生成）
└── src/
    ├── camera_calibration.py      # カメラキャリブレーション
    ├── object_detector.py         # YOLOv8物体検出
    ├── position_estimator.py      # 距離・3D位置計算
    └── realtime_position_tracker.py  # 統合システム
```

## 外部連携（API使用例）

```python
from src.realtime_position_tracker import RealtimePositionTracker
import cv2

tracker = RealtimePositionTracker("config/settings.yaml")

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

# 位置を取得
positions = tracker.get_positions(frame, target_classes=["bottle"])

for pos in positions:
    print(f"{pos.class_name}: X={pos.x:.0f}mm, Y={pos.y:.0f}mm, Z={pos.z:.0f}mm")

cap.release()
```

## ライセンス

MIT License
