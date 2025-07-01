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

## 使用例

```bash
# B-scanプロット
python visualization/plot_Bscan.py

# 岩石検出結果表示
python visualization/plot_Bscan_detected_rocks.py

# インタラクティブビューア
python visualization/plot_viewer/plot_viewer_add_label.py
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