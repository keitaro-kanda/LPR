# Visualization Tools

LPRデータの可視化・プロットツール群です。

## 主要ツール

### B-scan可視化
- **`plot_Bscan.py`**: 基本的なB-scanプロット
- **`plot_Bscan_detected_rocks.py`**: 岩石検出結果付きB-scanプロット
- **`plot_Bscan_trim.py`**: トリミングされたB-scanプロット

### A-scan・スペクトル可視化
- **`plot_Ascan.py`**: A-scan（単一トレース）プロット
- **`plot_spectrum.py`**: 周波数スペクトル解析・プロット

### 位置・軌道データ
- **`plot_position.py`**: ローバー位置・軌道の可視化

### インタラクティブビューア (`plot_viewer/`)
- **`plot_viewer.py`**: 基本的なインタラクティブプロットビューア
- **`plot_viewer_add_label.py`**: ラベル追加機能付きビューア
- **`plot_viewer_depth_measurement.py`**: 深さ計測機能付きビューア

## 使用例

```bash
# B-scanプロット
python visualization/plot_Bscan.py

# 岩石検出結果表示
python visualization/plot_Bscan_detected_rocks.py

# インタラクティブビューア
python visualization/plot_viewer/plot_viewer_add_label.py

# 深さ計測ビューア
python visualization/plot_viewer/plot_viewer_depth_measurement.py
```

## 可視化機能

### データ表示
- **時間-距離軸**: ns/m単位での表示
- **深度-距離軸**: 地形補正対応の深度表示
- **カラーマップ**: seismic, viridis, turbo等のカラーマップ

### ラベル・注釈
- **岩石ラベル**: 6種類の岩石タイプの色分け表示
- **統計情報**: ヒストグラム・統計値の重畳表示
- **スケールバー**: 距離・時間スケールの表示

### 出力フォーマット
- **高解像度画像**: PNG (120-300 DPI) + PDF (600 DPI)
- **インタラクティブ表示**: PyQtGraphによるリアルタイム操作
- **複数フォーマット**: データ・画像・PDFの同時出力
- **計測データ**: JSON形式での深さ計測結果出力

## 深さ計測機能

### plot_viewer_depth_measurement.py
B-scan上で縦方向の深さを計測するインタラクティブツールです。

**主な機能:**
- **2点クリック計測**: B-scan上で2回クリックして縦方向の距離を計測
- **複数計測管理**: 複数の計測結果を同時に表示・管理
- **視覚的表示**: 始点と終点を結ぶ直線と時間差[ns]の表示
- **計測結果削除**: 計測線をダブルクリックで削除
- **JSON出力**: 計測データの自動保存（終了時にx座標順でソート）

**計測結果データ構造:**
```json
{
  "measurements": [
    {
      "id": 1,
      "x_position": 10.5,
      "start_time_ns": 100.0,
      "end_time_ns": 200.0,
      "time_diff_ns": 100.0,
      "depth_m": 5.2
    }
  ]
}
```

**物理定数:**
- sample_interval: 0.312500e-9 [s]
- trace_interval: 3.6e-2 [m]
- epsilon_r: 4.5 (相対誘電率)
- 深さ計算: `depth = (time_diff * c) / (2 * sqrt(epsilon_r))`
- t=0補正: x=0でのNaN以外の最初の値をt=0として設定

**使用方法:**
1. B-scan上で2回クリックして深さを計測
2. 測定線をダブルクリックで削除
3. 測定データは自動的にJSONファイルに保存
4. 終了時にx座標順でIDを振り直して最終保存