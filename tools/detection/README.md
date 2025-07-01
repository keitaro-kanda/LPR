# Detection Algorithms

LPRデータからの特徴検出・物体検出アルゴリズム群です。

## 主要ツール

### 双曲線検出 (`detect_hyperbola/`)
- **`detect_hyperbola.py`**: 基本的な双曲線検出アルゴリズム
- **`detect_hyperbola_Hough.py`**: Hough変換による双曲線検出
- **`robust_hyperbola_fitting.py`**: ロバスト双曲線フィッティング
- **`robust_hyperbola_main.py`**: ロバスト検出のメインスクリプト

### その他の検出アルゴリズム
- **`detect_peak.py`**: ピーク検出アルゴリズム
- **`hyperbole_detection_plot_boundingbox.py`**: バウンディングボックス表示

## 使用例

```bash
# 双曲線検出
python detection/detect_hyperbola/detect_hyperbola.py

# Hough変換による双曲線検出
python detection/detect_hyperbola/detect_hyperbola_Hough.py

# ピーク検出
python detection/detect_peak.py
```

## 検出対象

- **地下岩石**: 双曲線状の反射パターン
- **層構造**: 水平方向の連続性
- **異常構造**: 通常パターンからの逸脱

## アルゴリズム特徴

- **マルチスケール処理**: 複数の空間スケールでの検出
- **ノイズ耐性**: メディアンフィルタ等による前処理
- **統計的検証**: 検出結果の信頼性評価